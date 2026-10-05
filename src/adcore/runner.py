"""``adcore-run``: few-shot experiments from Hydra configs, on one or several GPUs.

    adcore-run model=patchcore_wrn50 experiment=mvtec_fewshot
    adcore-run model=patchcore_wrn50 experiment=visa_fewshot launcher=gpus launcher.devices=[0,1]
    adcore-run --config-dir configs model=my_model experiment=mvtec_fewshot   # downstream repo
    adcore-run -m model=patchcore_wrn50,my_model experiment=mvtec_fewshot     # one after another

The config has a ``model`` (``shared`` objects built once per process, a ``module`` built
per run, transforms, Trainer kwargs), an ``experiment`` (dataset, categories, shots,
seeds) and a ``launcher``; ``adcore/conf/adcore.yaml`` documents every key. The module
can come from any package: the only requirement is that it is an `AnomalyModule`.

With several GPU slots, each slot is a spawned worker process that sees only its GPU and
takes whole categories off a queue, so a category's test split is still decoded once.
Workers send their outcomes back and this process alone writes ``results.jsonl``, so
resuming works as with `FewShotExperiment.run`. Workers log to MLflow themselves
(including training curves) only when the store is a server; with SQLite this process
records the final metrics instead.

Needs the ``run`` extra (``hydra-core``).
"""

from __future__ import annotations

import argparse
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
from hydra.utils import get_object, instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from adcore.experiment import FewShotExperiment, Run
from adcore.module import AnomalyModule
from adcore.mvtec import MVTEC, VISA, DatasetSpec

log = logging.getLogger(__name__)

_SPECS = {spec.name: spec for spec in (MVTEC, VISA)}
# Experiment keys that may change between sessions of one output directory: the grid
# grows, and loader settings don't change results.
_UNHASHED = ("name", "categories", "shots", "seeds", "batch_size", "num_workers", "cache_test_set")
# MLflow stores that tolerate writes from several processes.
_CONCURRENT_SCHEMES = ("http", "https", "databricks", "postgresql", "mysql", "mssql")


def _build(node: Any, **kwargs: Any) -> Any:
    if node is None:
        return None
    return instantiate(node, _convert_="all", **kwargs)


class ModuleFactory:
    """``module_factory`` for `FewShotExperiment` from a model config.

    ``model.shared`` is built on the first call and passed by name to every module, so
    a backbone loads once per process. ``model.module`` is built fresh for each run.
    """

    def __init__(self, model: DictConfig) -> None:
        self.module = model.module
        self.shared = model.get("shared") or {}
        self._objects: dict[str, Any] | None = None
        target = get_object(self.module._target_)
        if inspect.isclass(target) and not issubclass(target, AnomalyModule):
            raise TypeError(f"model.module {self.module._target_} is not an AnomalyModule")
        self.takes_run = "run" in inspect.signature(target).parameters

    def __call__(self, run: Run) -> AnomalyModule:
        if self._objects is None:
            self._objects = {key: _build(node) for key, node in self.shared.items()}
        kwargs = dict(self._objects)
        if self.takes_run:
            kwargs["run"] = run
        module = _build(self.module, **kwargs)
        if not isinstance(module, AnomalyModule):
            raise TypeError(f"model.module built a {type(module).__name__}, not an AnomalyModule")
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
    model.pop("name", None)
    return MLflowTracking(
        experiment=settings.experiment,
        arch=settings.arch,
        tracking_uri=settings.tracking_uri or default_tracking_uri(),
        params=_flatten(model, "model"),
        tags=dict(settings.tags),
        sweep=settings.sweep,
    )


def build_experiment(
    cfg: DictConfig, *, tracking: bool = True, device: int | None = None
) -> FewShotExperiment:
    """The `FewShotExperiment` a resolved config describes; nothing is loaded until it runs."""
    model, experiment = cfg.model, cfg.experiment
    trainer = {
        **OmegaConf.to_container(experiment.get("trainer") or {}),
        **OmegaConf.to_container(model.get("trainer") or {}),
    }
    if device is not None:
        trainer["devices"] = [device]
    categories = experiment.get("categories")
    return FewShotExperiment(
        experiment.root,
        module_factory=ModuleFactory(model),
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


def _units(
    categories: Sequence[str], todo: Sequence[Run], n_workers: int
) -> list[tuple[str, list[Run]]]:
    """Work for the queue: whole categories, or (category, shots) when categories are too few."""
    units = [(c, [run for run in todo if run.category == c]) for c in categories]
    units = [unit for unit in units if unit[1]]
    if len(units) >= n_workers:
        return units
    return [
        (category, [run for run in runs if run.shots == shots])
        for category, runs in units
        for shots in dict.fromkeys(run.shots for run in runs)
    ]


def _concurrent_store(uri: str) -> bool:
    return uri.split(":", 1)[0].split("+", 1)[0] in _CONCURRENT_SCHEMES


def _worker(container: dict, device: int, sweep_id: str | None, work, results) -> None:
    # Only this GPU, set before anything touches CUDA; `device` indexes the visible ones.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[device] if visible else str(device)
    cfg = OmegaConf.create(container)
    experiment = build_experiment(cfg, tracking=sweep_id is not None)
    if sweep_id is not None:
        experiment.use_sweep(sweep_id)
    while (unit := work.get()) is not None:
        category, runs = unit
        try:
            for outcome in experiment.run_category(category, runs):
                results.put(("outcome", outcome))
        except Exception:
            results.put(("failed", (category, traceback.format_exc())))
        else:
            results.put(("done", category))


def _run_parallel(
    container: dict, experiment: FewShotExperiment, slots: list[int], progress: bool
) -> pd.DataFrame:
    experiment.write_config()
    todo = experiment.todo()
    units = _units(experiment.categories, todo, len(slots))
    tracking = experiment.tracking
    sweep_id = experiment.start_sweep() if tracking is not None else None
    workers_track = tracking is not None and _concurrent_store(tracking.tracking_uri)
    if tracking is not None and not workers_track:
        log.warning(
            "%s is not safe for several writers: recording final metrics from the main "
            "process, without training curves. Set MLFLOW_TRACKING_URI to a tracking "
            "server to get them.",
            tracking.tracking_uri,
        )

    context = mp.get_context("spawn")
    work, results = context.Queue(), context.Queue()
    for unit in units:
        work.put(unit)
    for _ in slots:
        work.put(None)
    workers = [
        context.Process(
            target=_worker,
            args=(container, device, sweep_id if workers_track else None, work, results),
            name=f"adcore-worker{i}-gpu{device}",
        )
        for i, device in enumerate(slots)
    ]
    log.info(
        "%d runs in %d units on %d workers (GPUs %s)",
        len(todo), len(units), len(slots), ",".join(map(str, slots)),
    )

    failures: list[tuple[str, str]] = []
    remaining = len(units)
    bar = tqdm(total=len(todo), desc="Runs", disable=not progress)
    try:
        for worker in workers:
            worker.start()
        while remaining:
            try:
                kind, payload = results.get(timeout=10)
            except queue.Empty:
                if not any(worker.is_alive() for worker in workers):
                    break
                continue
            if kind == "outcome":
                experiment.record(payload)
                if tracking is not None and not workers_track:
                    experiment.track(payload)
                bar.update()
                bar.set_postfix_str(payload["run"].name)
            else:
                remaining -= 1
                if kind == "failed":
                    failures.append(payload)
                    log.error("category %s failed:\n%s", *payload)
        for worker in workers:
            worker.join()
    finally:
        bar.close()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        if tracking is not None:
            tracking.end_sweep(sweep_id)

    if failures or remaining:
        raise RuntimeError(
            f"{len(failures)} work units failed ({', '.join(c for c, _ in failures)}) and "
            f"{remaining} were lost to dead workers; rerun to resume the missing runs"
        )
    return experiment.results()


def run(cfg: DictConfig) -> pd.DataFrame:
    """Run the experiment ``cfg`` describes; call it from your own ``@hydra.main`` if you like."""
    container = OmegaConf.to_container(cfg, resolve=True)  # resolves ${hydra:...} here
    cfg = OmegaConf.create(container)
    slots = _slots(cfg.get("launcher"))
    if len(slots) <= 1:
        experiment = build_experiment(cfg, device=slots[0] if slots else None)
        _check_resume(experiment, cfg.get("force", False))
        return experiment.run(progress=cfg.get("progress", True))
    experiment = build_experiment(cfg)
    _check_resume(experiment, cfg.get("force", False))
    return _run_parallel(container, experiment, slots, cfg.get("progress", True))


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
