"""``adcore-run``: few-shot experiments from Hydra configs, on one or several GPUs.

    adcore-run model=patchcore_wrn50 experiment=mvtec_fewshot
    adcore-run model=patchcore_wrn50 experiment=visa_fewshot launcher=gpus launcher.devices=[0,1]
    adcore-run --config-dir configs model=my_model experiment=mvtec_fewshot   # downstream repo
    adcore-run -m model=patchcore_wrn50,my_model experiment=mvtec_fewshot     # one after another
    adcore-run -m model.module.sampling_ratio=0.1,0.01                         # a hyperparameter grid

The config has a ``model`` (``shared`` objects built once per process, a ``module`` built
per run, transforms, Trainer kwargs), an ``experiment`` (dataset, categories, shots,
seeds) and a ``launcher``; ``adcore/conf/adcore.yaml`` documents every key. The module
can come from any package: the only requirement is that it is an `AnomalyModule`.

With several GPU slots, each slot is a spawned worker process that sees only its GPU and
takes whole categories off a queue, so a category's test split is still decoded once.
In a multirun (``-m``) the jobs are queued as Hydra launches them and run together once
the last is queued (`PooledMultirun`), so a GPU that finishes one variant moves on to the
next instead of waiting for the slowest GPU. Workers send their outcomes back and this process alone writes ``results.jsonl``, so
resuming works as with `FewShotExperiment.run`. Each combination of ``model.*`` overrides is a *variant*
with its own output directory and MLflow sweep, tagged ``variant``. Workers log to MLflow themselves
(including training curves) only when the store is a server; with SQLite this process
records the final metrics instead.

Needs the ``run`` extra (``hydra-core``).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import traceback
from collections.abc import Sequence
from typing import Any

import pandas as pd
from hydra.experimental.callback import Callback
from hydra.utils import get_object, instantiate
from lightning.pytorch.utilities.seed import isolate_rng
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from adcore.experiment import FewShotExperiment, Run
from adcore.module import AnomalyModule
from adcore.mvtec import MVTEC, MVTEC_AD2, REALIAD, VISA, DatasetSpec

log = logging.getLogger(__name__)

_SPECS = {spec.name: spec for spec in (MVTEC, VISA, MVTEC_AD2, REALIAD)}
# Experiment keys that may change between sessions of one output directory: the grid
# grows, and loader settings don't change results.
_UNHASHED = ("name", "categories", "shots", "seeds", "batch_size", "num_workers", "cache_test_set")
# MLflow stores that tolerate writes from several processes.
_CONCURRENT_SCHEMES = ("http", "https", "databricks", "postgresql", "mysql", "mssql")


def _variant_from_overrides(overrides: Sequence[str]) -> str:
    """The label of a job's ``model.*`` overrides, e.g. ``model_name=resnet18,sampling_ratio=0.1``.

    Keys are the last path segment, or the whole path after ``model.`` when two overrides
    share one. Group choices (``model=...``), ``model.name`` and other overrides are left
    out, so a growing grid (``experiment.shots``) keeps the label.
    """
    pairs = {}
    for override in overrides:
        key, sep, value = override.lstrip("+~").partition("=")
        # model.name is already the arch and a directory level.
        if sep and key.startswith("model.") and key != "model.name":
            pairs[key.removeprefix("model.")] = value
    last = [key.rsplit(".", 1)[-1] for key in pairs]
    labels = {
        (short if last.count(short) == 1 else key): value
        for key, short, value in zip(pairs, last, pairs.values())
    }
    # A value may be a path or hub id; keep the label one directory level.
    return ",".join(f"{k}={v}".replace("/", "_") for k, v in sorted(labels.items()))


def _variant() -> str:
    from hydra.core.hydra_config import HydraConfig

    if not HydraConfig.initialized():
        return ""
    return _variant_from_overrides(HydraConfig.get().overrides.task)


# Used by adcore.yaml: `variant` and the per-variant output directory.
OmegaConf.register_new_resolver("adcore.variant", _variant, replace=True)
OmegaConf.register_new_resolver("adcore.subdir", lambda name: f"/{name}" if name else "", replace=True)


def _build(node: Any) -> Any:
    if node is None:
        return None
    return instantiate(node, _convert_="all")


class ModuleFactory:
    """``module_factory`` for `FewShotExperiment` from a model config.

    ``model.shared`` is built on the first call and passed by name to every module, so
    a backbone loads once per process. ``model.module`` is built fresh for each run.
    ``fpr_limit``, when given, replaces every module's AUPRO FPR limit.
    """

    def __init__(self, model: DictConfig, *, fpr_limit: float | None = None) -> None:
        self.module = model.module
        self.fpr_limit = fpr_limit
        self.shared = model.get("shared") or {}
        self._objects: dict[str, Any] | None = None
        target = get_object(self.module._target_)
        if inspect.isclass(target) and not issubclass(target, AnomalyModule):
            raise TypeError(f"model.module {self.module._target_} is not an AnomalyModule")
        self.takes_run = "run" in inspect.signature(target).parameters

    def __call__(self, run: Run) -> AnomalyModule:
        if self._objects is None:
            # Building draws random numbers (e.g. timm's init before the pretrained weights
            # load); without isolation the first run of each process would see another RNG
            # state than the rest, so results would depend on the worker and on resuming.
            with isolate_rng():
                self._objects = {key: _build(node) for key, node in self.shared.items()}
        kwargs = dict(self._objects)
        if self.takes_run:
            kwargs["run"] = run
        # Called rather than passed to `instantiate`, which would turn a `Run` into a config.
        module = instantiate(self.module, _convert_="all", _partial_=True)(**kwargs)
        if not isinstance(module, AnomalyModule):
            raise TypeError(f"model.module built a {type(module).__name__}, not an AnomalyModule")
        if self.fpr_limit is not None:
            # Set rather than passed: a subclass need not take `fpr_limit` itself.
            module.fpr_limit = self.fpr_limit
        return module


def config_hash(cfg: DictConfig) -> str:
    """Hash of the settings that change results: the model, and the experiment without its grid."""
    model = OmegaConf.to_container(cfg.model, resolve=True)
    experiment = {
        k: v
        for k, v in OmegaConf.to_container(cfg.experiment, resolve=True).items()
        if k not in _UNHASHED
    }
    text = json.dumps({"model": model, "experiment": experiment}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _plain(node: DictConfig | None) -> dict[str, Any]:
    return {} if node is None else OmegaConf.to_container(node, resolve=True)


def _flatten(node: Any, prefix: str) -> dict[str, Any]:
    if isinstance(node, dict) and node:
        out = {}
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}.{key}"))
        return out
    return {prefix: node}


def _spec(value: Any) -> DatasetSpec:
    if isinstance(value, str):
        return _SPECS[value]
    return _build(value)


def _tracking(cfg: DictConfig):
    settings = cfg.get("tracking")
    if settings is None or not settings.enabled:
        return None
    from adcore.tracking import MLflowTracking, default_tracking_uri

    model = OmegaConf.to_container(cfg.model, resolve=True)
    name = model.pop("name", None)
    tags = dict(settings.tags)
    sweep = settings.sweep
    if variant := cfg.get("variant"):
        tags["variant"] = variant
        # The output directory is named after the variant alone; keep the model in the sweep.
        sweep = sweep or f"{name}[{variant}]"
    return MLflowTracking(
        experiment=settings.experiment,
        arch=settings.arch,
        tracking_uri=settings.tracking_uri or default_tracking_uri(),
        params=_flatten(model, "model"),
        tags=tags,
        sweep=sweep,
    )


def build_experiment(
    cfg: DictConfig, *, tracking: bool = True, device: int | None = None
) -> FewShotExperiment:
    """The `FewShotExperiment` a resolved config describes; nothing is loaded until it runs."""
    model, experiment = cfg.model, cfg.experiment
    trainer = {
        **_plain(experiment.get("trainer")),
        **_plain(model.get("trainer")),
    }
    if device is not None:
        trainer["devices"] = [device]
    categories = experiment.get("categories")
    return FewShotExperiment(
        experiment.root,
        module_factory=ModuleFactory(model, fpr_limit=experiment.get("fpr_limit")),
        categories=list(categories) if categories is not None else None,
        shots=list(experiment.shots),
        seeds=list(experiment.seeds),
        output_dir=cfg.output_dir,
        trainer_kwargs=trainer,
        anomalous_fraction=experiment.anomalous_fraction,
        image_size=tuple(model.image_size),
        transform=_build(model.get("transform")),
        train_transform=_build(model.get("train_transform")),
        support_aug=_build(model.get("support_aug")),
        batch_size=experiment.batch_size,
        num_workers=experiment.num_workers,
        cache_test_set=experiment.cache_test_set,
        spec=_spec(experiment.spec),
        tracking=_tracking(cfg) if tracking else None,
        metadata={"config_hash": config_hash(cfg)},
    )


def _check_resume(experiment: FewShotExperiment, force: bool) -> None:
    """Refuse to add runs to an output directory made with other settings."""
    if experiment.output_dir is None or not experiment.results_path.exists():
        return
    config_path = experiment.output_dir / "config.json"
    previous = json.loads(config_path.read_text()).get("config_hash") if config_path.exists() else None
    current = experiment.metadata["config_hash"]
    if previous is None or previous == current:
        return
    message = (
        f"{experiment.output_dir} holds results from other settings "
        f"(config hash {previous}, now {current})"
    )
    if not force:
        raise RuntimeError(f"{message}; use another output_dir, or force=true to add to it")
    log.warning("%s; adding to it anyway (force=true)", message)


def _slots(launcher: DictConfig | None) -> list[int]:
    """Indices into the visible GPUs, one per worker; empty runs everything in this process."""
    if launcher is None or launcher.devices is None:
        return []
    if launcher.devices == "all":
        import torch

        devices = list(range(torch.cuda.device_count()))
        if not devices:
            raise RuntimeError("launcher.devices=all, but no GPU is visible")
    else:
        devices = list(launcher.devices)
    return [d for d in devices for _ in range(launcher.workers_per_device)]


def _units(categories: Sequence[str], todo: Sequence[Run]) -> list[tuple[str, list[Run]]]:
    """Work for the queue: the runs of each category."""
    units = [(c, [run for run in todo if run.category == c]) for c in categories]
    return [unit for unit in units if unit[1]]


def _by_shots(units: Sequence[tuple[str, list[Run]]]) -> list[tuple[str, list[Run]]]:
    """Units split into (category, shots), for when categories are too few for the workers."""
    return [
        (category, [run for run in runs if run.shots == shots])
        for category, runs in units
        for shots in dict.fromkeys(run.shots for run in runs)
    ]


def _concurrent_store(uri: str) -> bool:
    return uri.split(":", 1)[0].split("+", 1)[0] in _CONCURRENT_SCHEMES


def _worker(
    containers: list[dict], sweep_ids: list[str | None], device: int, work, results
) -> None:
    # Only this GPU, set before anything touches CUDA; `device` indexes the visible ones.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[device] if visible else str(device)
    import torch

    current, experiment = None, None
    while (unit := work.get()) is not None:
        job, category, runs = unit
        try:
            if job != current:
                # One job's experiment at a time, so one variant's shared objects on the GPU.
                current, experiment = None, None
                gc.collect()
                torch.cuda.empty_cache()
                experiment = build_experiment(
                    OmegaConf.create(containers[job]), tracking=sweep_ids[job] is not None
                )
                if sweep_ids[job] is not None:
                    experiment.use_sweep(sweep_ids[job])
                current = job
            for outcome in experiment.run_category(category, runs):
                results.put(("outcome", job, outcome))
        except Exception:
            results.put(("failed", job, (category, traceback.format_exc())))
        else:
            results.put(("done", job, category))


def _run_parallel(
    jobs: Sequence[tuple[dict, FewShotExperiment]], slots: list[int], progress: bool
) -> pd.DataFrame:
    """Run every job's missing runs on one pool of workers; returns the last job's results."""
    containers = [container for container, _ in jobs]
    experiments = [experiment for _, experiment in jobs]
    todos, seen = [], set()
    for experiment in experiments:
        experiment.write_config()
        # Jobs that differ only outside model.* share an output directory, and maybe runs.
        where = experiment.output_dir or id(experiment)
        todo = [run for run in experiment.todo() if (where, run.key) not in seen]
        seen.update((where, run.key) for run in todo)
        todos.append(todo)
    per_job = [_units(e.categories, todo) for e, todo in zip(experiments, todos)]
    if sum(map(len, per_job)) < len(slots):
        per_job = [_by_shots(units) for units in per_job]
    units = [(job, *unit) for job, job_units in enumerate(per_job) for unit in job_units]
    if not units:
        return experiments[-1].results()

    sweep_ids = [
        e.start_sweep() if e.tracking is not None and todo else None
        for e, todo in zip(experiments, todos)
    ]
    workers_track = [
        e.tracking is not None and _concurrent_store(e.tracking.tracking_uri)
        for e in experiments
    ]
    for e, sweep_id, tracked in zip(experiments, sweep_ids, workers_track):
        if sweep_id is not None and not tracked:
            log.warning(
                "%s is not safe for several writers: recording final metrics from the main "
                "process, without training curves. Set MLFLOW_TRACKING_URI to a tracking "
                "server to get them.",
                e.tracking.tracking_uri,
            )
            break

    context = mp.get_context("spawn")
    work, results = context.Queue(), context.Queue()
    for unit in units:
        work.put(unit)
    for _ in slots:
        work.put(None)
    worker_sweeps = [s if tracked else None for s, tracked in zip(sweep_ids, workers_track)]
    workers = [
        context.Process(
            target=_worker,
            args=(containers, worker_sweeps, device, work, results),
            name=f"adcore-worker{i}-gpu{device}",
        )
        for i, device in enumerate(slots)
    ]
    n_runs = sum(map(len, todos))
    log.info(
        "%d runs in %d units from %d jobs on %d workers (GPUs %s)",
        n_runs, len(units), sum(1 for todo in todos if todo), len(slots),
        ",".join(map(str, slots)),
    )

    failures: list[tuple[Any, str, str]] = []
    remaining = len(units)
    bar = tqdm(total=n_runs, desc="Runs", disable=not progress)
    try:
        for worker in workers:
            worker.start()
        while remaining:
            try:
                kind, job, payload = results.get(timeout=10)
            except queue.Empty:
                if not any(worker.is_alive() for worker in workers):
                    break
                continue
            experiment = experiments[job]
            if kind == "outcome":
                experiment.record(payload)
                if experiment.tracking is not None and not workers_track[job]:
                    experiment.track(payload)
                bar.update()
                bar.set_postfix_str(payload["run"].name)
            else:
                remaining -= 1
                if kind == "failed":
                    failures.append((experiment.output_dir, *payload))
                    log.error("%s: category %s failed:\n%s", *failures[-1])
        for worker in workers:
            worker.join()
    finally:
        bar.close()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        for e, sweep_id in zip(experiments, sweep_ids):
            if sweep_id is not None:
                e.tracking.end_sweep(sweep_id)

    if failures or remaining:
        failed = ", ".join(f"{d}:{c}" if len(jobs) > 1 else c for d, c, _ in failures)
        raise RuntimeError(
            f"{len(failures)} work units failed ({failed}) and "
            f"{remaining} were lost to dead workers; rerun to resume the missing runs"
        )
    return experiments[-1].results()


# Multi-GPU jobs of the running multirun, waiting for one shared pool; None when no
# `PooledMultirun` is watching this process.
_pending: list[tuple[dict, FewShotExperiment, list[int], bool]] | None = None


class PooledMultirun(Callback):
    """Hydra callback: runs a multirun's multi-GPU jobs on one worker pool.

    Each job queues itself in `run`; the pool starts once the last job is queued. Jobs
    launched in other processes (e.g. by the joblib launcher) never see the queue and
    run on their own.
    """

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        global _pending
        _pending = []

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        global _pending
        jobs, _pending = _pending or [], None
        if jobs:
            _, _, slots, progress = jobs[0]
            _run_parallel([(c, e) for c, e, _, _ in jobs], slots, progress)


def run(cfg: DictConfig) -> pd.DataFrame | None:
    """Run the experiment ``cfg`` describes; call it from your own ``@hydra.main`` if you like.

    Within a multirun watched by `PooledMultirun`, a multi-GPU job is only queued and
    None is returned; it runs with the others when the multirun ends.
    """
    container = OmegaConf.to_container(cfg, resolve=True)  # resolves ${hydra:...} here
    cfg = OmegaConf.create(container)
    slots = _slots(cfg.get("launcher"))
    if len(slots) <= 1:
        experiment = build_experiment(cfg, device=slots[0] if slots else None)
        _check_resume(experiment, cfg.get("force", False))
        return experiment.run(progress=cfg.get("progress", True))
    experiment = build_experiment(cfg)
    _check_resume(experiment, cfg.get("force", False))
    progress = cfg.get("progress", True)
    if _pending is not None:
        if _pending and _pending[0][2] != slots:
            raise ValueError(
                f"the jobs of a multirun share one worker pool, but this job's launcher "
                f"slots {slots} differ from {_pending[0][2]}"
            )
        _pending.append((container, experiment, slots, progress))
        log.info("queued %s for the shared worker pool", experiment.output_dir)
        return None
    return _run_parallel([(container, experiment)], slots, progress)


def _patch_hydra_args_parser() -> None:
    """hydra-core 1.3 gives argparse a non-str help, which Python 3.14 rejects."""
    import hydra  # noqa: F401  (loads hydra.main)

    hydra_main = sys.modules["hydra.main"]

    build = hydra_main.get_args_parser
    if getattr(build, "_adcore_patched", False):
        return

    def get_args_parser() -> argparse.ArgumentParser:
        check = argparse.ArgumentParser._check_help
        argparse.ArgumentParser._check_help = lambda self, action: (
            check(self, action) if isinstance(action.help, str | None) else None
        )
        try:
            return build()
        finally:
            argparse.ArgumentParser._check_help = check

    get_args_parser._adcore_patched = True
    hydra_main.get_args_parser = get_args_parser


def main() -> None:
    import hydra

    if hasattr(argparse.ArgumentParser, "_check_help"):
        _patch_hydra_args_parser()
    hydra.main(config_path="conf", config_name="adcore", version_base="1.3")(run)()
