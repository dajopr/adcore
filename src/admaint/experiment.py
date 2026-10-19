"""Maintenance experiments: fit a `Maintainable` module, evaluate it, update it step by
step and evaluate it after every step.

    extractor = TimmExtractor("wide_resnet50_2", out_indices=(2, 3))
    experiment = MaintenanceExperiment(
        root="/data/shared-data/public_datasets/raw/MVTec",
        module_factory=lambda run: MaintainablePatchCore(extractor, sampling_ratio=1.0),
        scenario="contamination",
        shots=(8,), n_contam=(0, 1, 2), n_update=(1, 2), seeds=(0, 1, 2),
        output_dir="runs/mvtec-contamination/patchcore-wrn50",
    )
    results = experiment.run()     # one row per (run, phase, eval set, view, defect type)
    before_after(results, "image_auroc", defect_type="target")

A run is a `MaintRun`; its data comes from the scenario's builder in `admaint.protocol`.
Its ``seed`` draws what the module is fitted on and its ``update_seed`` the update frames,
so ``update_seeds`` adds runs that reuse a fit: the experiment fits once per distinct
`fit_key`, keeps the module's `Maintainable.fit_state` and restores it into a fresh module
for the other runs of that fit (``cache_fits``).
Phase 0 is the fitted module, phase i the module after step i. Every phase is scored on
each eval set (``"test"``, or ``"source"`` / ``"target"`` under drift) and under each label
view (``"original"``, plus ``"extended"`` for extension). The ``defect_type == "all"`` row
also carries the detection threshold calibrated on held-out good frames at
``calibration_fpr``, recalibrated every phase, the rates of frames above it (see
`admaint.protocol.threshold_metrics`) and the module's `maintenance_stats`.

With an ``output_dir`` every run is appended to ``results.jsonl`` as soon as it ends, with
per-image scores in ``scores/<run>/phase<i>_<eval set>.csv`` and the module's files in
``artifacts/<run>/phase<i>/<eval set>/``; a rerun skips the runs already there.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.model_helpers import is_overridden
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from adcore.datamodule import CachedDataset
from adcore.evaluation import STAT, score
from adcore.experiment import _jsonable, _quiet_lightning, results_frame
from adcore.module import AnomalyModule
from adcore.mvtec import MVTEC, DatasetSpec, MVTecDataset, default_transform, find_records
from admaint import protocol as P
from admaint.maintainable import Maintainable

if TYPE_CHECKING:
    from adcore.tracking import MLflowTracking

log = logging.getLogger(__name__)


# Repeated on every test / predict pass of a run, on top of adcore's.
_QUIET = (
    r".*isinstance\(treespec, LeafSpec\).*",
    r".*module\(s\) in eval mode at the start of training.*",
    r".*defined a `validation_step` but have no `val_dataloader`.*",
)


@contextmanager
def _quiet():
    cuda = logging.getLogger("lightning.pytorch.accelerators.cuda")  # LOCAL_RANK lines
    level = cuda.level
    cuda.setLevel(logging.WARNING)
    with _quiet_lightning(), warnings.catch_warnings():
        for message in _QUIET:
            warnings.filterwarnings("ignore", message=message)
        try:
            yield
        finally:
            cuda.setLevel(level)


@dataclass(frozen=True)
class MaintRun:
    """One maintenance run. Fields a scenario does not use are ``""`` or 0."""

    category: str
    target: str  # the defect type (contamination, extension)
    shift: str  # the shift's name (drift)
    shots: int | None  # None: every good training frame but the calibration set
    n_contam: int
    n_update: int
    seed: int  # draws the frames the module is fitted on
    update_seed: int = 0  # draws the update frames

    @property
    def key(self) -> tuple:
        return tuple(asdict(self).values())

    @property
    def name(self) -> str:
        parts = [self.category, self.target, self.shift]
        parts.append(f"{'full' if self.shots is None else self.shots}shot")
        if self.n_contam:
            parts.append(f"c{self.n_contam}")
        if self.n_update:
            parts.append(f"u{self.n_update}")
        parts.append(f"seed{self.seed}")
        parts.append(f"upd{self.update_seed}")
        return "_".join(p for p in parts if p)


RUN_FIELDS = tuple(MaintRun.__dataclass_fields__)
_GRID = (
    "categories", "target_defects", "shots", "n_contam", "n_update", "shifts", "seeds",
    "update_seeds",
)


class MaintenanceExperiment:
    """Every run of one scenario's grid on one dataset.

    Args:
        root: Dataset directory, one subdirectory per category.
        module_factory: Builds a fresh, unfitted `Maintainable` `AnomalyModule` for a
            `MaintRun`.
        scenario: ``"contamination"``, ``"extension"``, ``"stepwise"`` or ``"drift"``.
        categories: Defaults to all of ``spec.categories``.
        target_defects: Defect types to target (contamination, extension); None means
            every defect type of each category. Types with fewer than ``min_eval`` frames
            left to evaluate are skipped with a warning.
        shots: Good frames the module is fitted on; None means all but the calibration set.
        n_contam: Contaminants in the training set (contamination only).
        n_update: Frames in the update (contamination, extension, drift).
        steps: Cumulative good-frame counts after each step (stepwise only).
        shifts: Named domain shifts, torchvision v2 transforms over ``(image, mask)``
            (drift only); see `admaint.protocol.Shifted`.
        retire: Under drift, also remove the initial frames in a second step.
        seeds: Fit seeds: each draws the frames the module is fitted on (initial frames,
            contaminants, calibration set) and goes to ``L.seed_everything`` before the
            module is built.
        update_seeds: Update seeds, one run each per fit seed: each draws the update
            frames (and so which target frames are evaluated) and seeds the updates.
            Runs that differ only in their update seed share one fit.
        cache_fits: Fit once per `fit_key` and restore that fit for the other runs with
            the same key (`Maintainable.fit_state`). The module factory must then not
            depend on run fields outside the key.
        min_eval: Fewest frames of a target defect type left to evaluate.
        n_calibration: Good training frames held out to calibrate the threshold.
        calibration_fpr: Fraction of calibration frames above the threshold.
        output_dir: Where ``config.json``, ``results.jsonl``, ``scores/``, ``artifacts/``
            and Lightning ``logs/`` go. None keeps results in memory only.
        trainer_kwargs: ``Trainer`` arguments, or a function of the run returning them,
            over the defaults of `adcore.FewShotExperiment`.
        image_size, transform, train_transform, batch_size, num_workers, spec: As for
            `adcore.FewShotExperiment`. ``train_transform`` applies to the good frames
            the module is fitted on or updated with; everything else uses ``transform``.
        tracking: Also record every run in MLflow; its sweep name defaults to the name
            of ``output_dir``.
        metadata: Extra entries for ``config.json`` and the sweep's MLflow params.
    """

    def __init__(
        self,
        root: str | Path,
        module_factory: Callable[[MaintRun], AnomalyModule],
        scenario: str,
        *,
        categories: Sequence[str] | None = None,
        target_defects: Sequence[str] | None = None,
        shots: Sequence[int | None] = (8,),
        n_contam: Sequence[int] = (0, 1, 2),
        n_update: Sequence[int] = (1,),
        steps: Sequence[int] = (),
        shifts: dict[str, Callable] | None = None,
        retire: bool = False,
        seeds: Sequence[int] = (0, 1, 2),
        update_seeds: Sequence[int] = (0,),
        cache_fits: bool = True,
        min_eval: int = 3,
        n_calibration: int = 20,
        calibration_fpr: float = 0.05,
        output_dir: str | Path | None = None,
        trainer_kwargs: dict[str, Any] | Callable[[MaintRun], dict[str, Any]] | None = None,
        image_size: tuple[int, int] = (224, 224),
        transform=None,
        train_transform=None,
        batch_size: int = 32,
        num_workers: int = 4,
        spec: DatasetSpec = MVTEC,
        tracking: MLflowTracking | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if scenario not in P.SCENARIOS:
            raise ValueError(f"scenario must be one of {P.SCENARIOS}, got {scenario!r}")
        self.root = Path(root)
        self.module_factory = module_factory
        self.scenario = scenario
        self.spec = spec
        self.categories = list(categories) if categories is not None else list(spec.categories)
        self.target_defects = list(target_defects) if target_defects is not None else None
        self.shots = list(shots)
        uses = {
            "n_contam": scenario == P.CONTAMINATION,
            "n_update": scenario != P.STEPWISE,
            "steps": scenario == P.STEPWISE,
            "shifts": scenario == P.DRIFT,
        }
        self.n_contam = [int(n) for n in n_contam] if uses["n_contam"] else [0]
        self.n_update = [int(n) for n in n_update] if uses["n_update"] else [0]
        self.steps = [int(n) for n in steps] if uses["steps"] else []
        self.shifts = dict(shifts or {}) if uses["shifts"] else {}
        if uses["shifts"] and not self.shifts:
            raise ValueError("drift needs at least one shift")
        if uses["steps"] and not self.steps:
            raise ValueError("stepwise needs at least one step")
        if uses["n_update"] and min(self.n_update) < 1:
            raise ValueError(f"n_update must be at least 1, got {self.n_update}")
        self.retire = retire and scenario == P.DRIFT
        self.seeds = list(seeds)
        self.update_seeds = list(update_seeds)
        self.cache_fits = cache_fits
        self.min_eval = min_eval
        self.n_calibration = n_calibration
        self.calibration_fpr = calibration_fpr
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.trainer_kwargs = trainer_kwargs or {}
        self.image_size = tuple(image_size)
        self.transform = transform or default_transform(self.image_size)
        self.train_transform = train_transform or self.transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tracking = tracking
        self.metadata = dict(metadata or {})
        self._rows: list[dict[str, Any]] = []
        self._sweep_id: str | None = None
        self._targets: dict[str, list[str]] = {}
        self._fit: dict[str, Any] | None = None  # the last fit, while its runs go
        if tracking is not None and tracking.sweep is None and self.output_dir is None:
            raise ValueError("tracking without output_dir needs MLflowTracking(sweep=...)")

    # --- grid ------------------------------------------------------------------------

    def targets(self, category: str) -> list[str]:
        """The defect types of ``category`` this experiment targets, leaving out those too
        small to keep ``min_eval`` frames for evaluation."""
        if self.scenario not in (P.CONTAMINATION, P.EXTENSION):
            return [""]
        if category not in self._targets:
            counts: dict[str, int] = {}
            for record in find_records(self.root, [category], "test", self.spec):
                if record["label"]:
                    counts[record["defect_type"]] = counts.get(record["defect_type"], 0) + 1
            wanted = self.target_defects if self.target_defects is not None else sorted(counts)
            kept = []
            for target in wanted:
                left = P.n_target_eval(counts.get(target, 0), max(self.n_update), max(self.n_contam))
                if left < self.min_eval:
                    log.warning(
                        "skipping %s/%s: %d frames leave %d to evaluate, fewer than min_eval=%d",
                        category, target, counts.get(target, 0), left, self.min_eval,
                    )
                else:
                    kept.append(target)
            self._targets[category] = kept
        return self._targets[category]

    @property
    def runs(self) -> list[MaintRun]:
        return [
            MaintRun(category, target, shift, shots, n_contam, n_update, seed, update_seed)
            for category in self.categories
            for target, shift, shots, n_contam, n_update, seed, update_seed in product(
                self.targets(category),
                list(self.shifts) or [""],
                self.shots,
                self.n_contam,
                self.n_update,
                self.seeds,
                self.update_seeds,
            )
        ]

    def fit_key(self, run: MaintRun) -> tuple:
        """What the fitted module of ``run`` depends on: runs with equal keys share a fit.

        The category, shots and fit seed draw the initial frames; contaminants add the
        target and their number. Clean runs of every target share one fit.
        """
        key = (run.category, run.shots, run.seed)
        if run.n_contam:
            key += (run.target, run.n_contam)
        return key

    # --- bookkeeping -----------------------------------------------------------------

    @property
    def results_path(self) -> Path | None:
        return self.output_dir / "results.jsonl" if self.output_dir else None

    def config(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "spec": self.spec.name,
            "scenario": self.scenario,
            "categories": self.categories,
            "target_defects": self.target_defects,
            "shots": self.shots,
            "n_contam": self.n_contam,
            "n_update": self.n_update,
            "steps": self.steps,
            "shifts": {name: repr(shift) for name, shift in self.shifts.items()},
            "retire": self.retire,
            "seeds": self.seeds,
            "update_seeds": self.update_seeds,
            "min_eval": self.min_eval,
            "n_calibration": self.n_calibration,
            "calibration_fpr": self.calibration_fpr,
            "image_size": list(self.image_size),
            "transform": repr(self.transform),
            "train_transform": repr(self.train_transform),
            "trainer_kwargs": repr(self.trainer_kwargs),
            **self.metadata,
        }

    def write_config(self) -> None:
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "config.json").write_text(json.dumps(self.config(), indent=2))

    def _load_rows(self) -> list[dict[str, Any]]:
        if self.results_path is None or not self.results_path.exists():
            return list(self._rows)
        with self.results_path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if rows and "update_seed" not in rows[0]:
            raise RuntimeError(
                f"{self.results_path} is from an admaint without update seeds, whose draws "
                "differ; use another output_dir"
            )
        return rows

    def results(self) -> pd.DataFrame:
        """Every recorded row, including those from earlier sessions in ``output_dir``."""
        return results_frame(self._load_rows())

    def todo(self) -> list[MaintRun]:
        """The runs not already in ``output_dir``."""
        done = {tuple(row[f] for f in RUN_FIELDS) for row in self._load_rows()}
        return [run for run in self.runs if run.key not in done]

    @property
    def sweep(self) -> str | None:
        if self.tracking is None:
            return None
        return self.tracking.sweep or self.output_dir.name

    def start_sweep(self) -> str:
        """The MLflow parent run. The grid grows between sessions, so its children carry
        it and the parent does not."""
        if self._sweep_id is None:
            config = {k: v for k, v in self.config().items() if k not in _GRID}
            self._sweep_id = self.tracking.start_sweep(self.sweep, config)
        return self._sweep_id

    # --- running ---------------------------------------------------------------------

    def run(self, progress: bool = True) -> pd.DataFrame:
        """Run everything not already in ``output_dir`` and return all results."""
        self.write_config()
        todo = self.todo()
        if self.tracking is not None:
            self.start_sweep()
        bar = tqdm(total=len(todo), desc="Runs", disable=not progress)
        for category in self.categories:
            category_runs = [run for run in todo if run.category == category]
            if not category_runs:
                continue
            bar.set_postfix_str(category)
            for outcome in self.run_category(category, category_runs):
                self.record(outcome)
                bar.update()
        bar.close()
        if self.tracking is not None:
            self.tracking.end_sweep(self.start_sweep())
        return self.results()

    def _cache(self, dataset: Dataset) -> CachedDataset:
        return CachedDataset(dataset, num_workers=self.num_workers, batch_size=self.batch_size)

    def run_category(self, category: str, runs: Sequence[MaintRun]) -> Iterator[dict[str, Any]]:
        """Run ``runs``, all of ``category``; its test split (and each shifted copy) is
        decoded once for all of them."""
        test = MVTecDataset(self.root, category, "test", self.transform, spec=self.spec)
        data = {
            "train": MVTecDataset(self.root, category, "train", self.train_transform, spec=self.spec),
            "calibration_source": MVTecDataset(
                self.root, category, "train", self.transform, spec=self.spec
            ),
            "test": self._cache(test),
        }
        shifted: dict[str, CachedDataset] = {}
        # Runs sharing a fit go one after another, so one cached fit is enough.
        by_fit: dict[tuple, list[MaintRun]] = {}
        for run in runs:
            by_fit.setdefault(self.fit_key(run), []).append(run)
        self._fit = None
        for run in (run for group in by_fit.values() for run in group):
            if run.shift and run.shift not in shifted:
                shifted[run.shift] = self._cache(P.Shifted(test, self.shifts[run.shift]))
            yield self.run_one(run, {**data, "shifted_test": shifted.get(run.shift)})
        self._fit = None

    def protocol(self, run: MaintRun, data: dict[str, Dataset]) -> P.Protocol:
        """The data of ``run`` from its category's datasets."""
        common = (data["train"], data["calibration_source"], data["test"])
        if self.scenario == P.CONTAMINATION:
            return P.contamination(
                *common, run.category, run.target, run.seed, update_seed=run.update_seed,
                shots=run.shots, n_contam=run.n_contam, n_update=run.n_update,
                max_contam=max(self.n_contam), max_update=max(self.n_update),
                n_calibration=self.n_calibration,
            )
        if self.scenario == P.EXTENSION:
            return P.extension(
                *common, run.category, run.target, run.seed, update_seed=run.update_seed,
                shots=run.shots, n_update=run.n_update, max_update=max(self.n_update),
                n_calibration=self.n_calibration,
            )
        if self.scenario == P.STEPWISE:
            return P.stepwise(
                *common, run.category, run.seed, update_seed=run.update_seed,
                shots=run.shots, steps=self.steps, n_calibration=self.n_calibration,
            )
        return P.drift(
            *common, data["shifted_test"], run.category, run.seed, update_seed=run.update_seed,
            shift=self.shifts[run.shift], shots=run.shots, n_update=run.n_update,
            n_calibration=self.n_calibration, retire=self.retire,
        )

    def trainer(self, run: MaintRun) -> L.Trainer:
        overrides = self.trainer_kwargs(run) if callable(self.trainer_kwargs) else self.trainer_kwargs
        logger = CSVLogger(self.output_dir / "logs", name=run.name) if self.output_dir else False
        return L.Trainer(
            **{
                "accelerator": "auto",
                "devices": 1,
                "max_epochs": 1,
                "logger": logger,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "num_sanity_val_steps": 0,
                **overrides,
            }
        )

    def _loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        base = dataset
        while isinstance(base, Subset):
            base = base.dataset
        # Worker start-up dominates for a handful of frames; workers would copy a cache.
        cached = isinstance(base, CachedDataset)
        workers = 0 if cached or len(dataset) <= self.batch_size else self.num_workers
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=workers)

    def run_one(self, run: MaintRun, data: dict[str, Dataset]) -> dict[str, Any]:
        """Fit, update and evaluate one run; returns its rows, scores and artifacts."""
        mlflow_run_id = None
        if self.tracking is not None:
            mlflow_run_id = self.tracking.start_child(
                self.start_sweep(), self.sweep, run.name, _run_params(run), self.output_dir
            )
        try:
            outcome = self._run(run, data)
        except BaseException:
            if mlflow_run_id is not None:
                self.tracking.fail_run(mlflow_run_id)
            raise
        if mlflow_run_id is not None:
            self.tracking.log_artifacts(mlflow_run_id, outcome["artifacts"])
            self.tracking.finish_child(mlflow_run_id, mlflow_metrics(outcome["rows"]))
        return outcome

    def _run(self, run: MaintRun, data: dict[str, Dataset]) -> dict[str, Any]:
        with _quiet():
            L.seed_everything(run.seed, verbose=False)
            protocol = self.protocol(run, data)
            module = self.module_factory(run)
            if not isinstance(module, Maintainable):
                raise TypeError(f"{type(module).__name__} is not admaint.Maintainable")
            trainer = self.trainer(run)

            fit_seconds, fit_reused = self._fit_or_restore(run, module, trainer, protocol)
            # Updates draw from their own seed, whether the fit was run or restored.
            L.seed_everything(_update_seed(run), verbose=False)

            outcome = {"run": run, "rows": [], "scores": {}, "artifacts": {}}
            run_fields = {
                **asdict(run),
                "scenario": self.scenario,
                **protocol.sizes,
                "fit_seconds": fit_seconds,
                "fit_reused": fit_reused,
                "pro_connectivity": module.pro_connectivity,
            }
            self._evaluate(trainer, module, protocol, 0, None, 0.0, run_fields, outcome)
            for phase, step in enumerate(protocol.steps, start=1):
                seconds = self._apply(module, step, trainer.strategy.root_device)
                self._evaluate(trainer, module, protocol, phase, step, seconds, run_fields, outcome)

        del module, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return outcome

    def _fit_or_restore(
        self, run: MaintRun, module: AnomalyModule, trainer: L.Trainer, protocol: P.Protocol
    ) -> tuple[float, bool]:
        """Fit ``module``, or restore the cached fit of an earlier run with the same
        `fit_key`; returns the seconds the fit took and whether it was restored."""
        if not is_overridden("training_step", module):  # zero-shot: nothing to fit
            return 0.0, False
        key = self.fit_key(run)
        if self.cache_fits and self._fit is not None and self._fit["key"] == key:
            module.load_fit_state(self._fit["state"])
            return self._fit["seconds"], True
        start = perf_counter()
        trainer.fit(module, train_dataloaders=self._loader(protocol.initial, shuffle=True))
        seconds = perf_counter() - start
        if self.cache_fits:
            self._fit = {"key": key, "state": module.fit_state(), "seconds": seconds}
        return seconds, False

    def _apply(self, module: AnomalyModule, step: P.Step, device: torch.device) -> float:
        """Run ``step`` on ``module``; returns the seconds it took."""
        module.to(device).eval()
        update = getattr(module, f"{step.op}_representation")
        start = perf_counter()
        with torch.no_grad():
            for batch in self._loader(step.dataset):
                masks = batch["mask"].to(device) if step.use_masks else None
                update(batch["image"].to(device), masks)
        return perf_counter() - start

    def _threshold(self, trainer: L.Trainer, module: AnomalyModule, dataset: Dataset) -> float:
        outputs = trainer.predict(module, dataloaders=self._loader(dataset))
        scores = torch.cat([o["pred_score"].detach().double().reshape(-1).cpu() for o in outputs])
        return float(np.quantile(scores.numpy(), 1 - self.calibration_fpr))

    def _evaluate(
        self,
        trainer: L.Trainer,
        module: AnomalyModule,
        protocol: P.Protocol,
        phase: int,
        step: P.Step | None,
        update_seconds: float,
        run_fields: dict[str, Any],
        outcome: dict[str, Any],
    ) -> None:
        maintenance = module.maintenance_stats()
        thresholds: dict[int, float] = {}  # by calibration set, which eval sets may share
        for eval_set, dataset in protocol.evals.items():
            start = perf_counter()
            trainer.test(module, dataloaders=self._loader(dataset), verbose=False)
            test_seconds = perf_counter() - start
            result = module.test_result
            calibration = protocol.calibration[eval_set]
            if id(calibration) not in thresholds:
                thresholds[id(calibration)] = self._threshold(trainer, module, calibration)
            threshold = thresholds[id(calibration)]
            overall = {
                **P.threshold_metrics(result.predictions, threshold, protocol.target),
                **{f"{STAT}{name}": value for name, value in result.stats.items()},
                **maintenance,
            }
            phase_fields = {
                **run_fields,
                "phase": phase,
                "step": step.name if step else "initial",
                "op": step.op if step else "",
                "n_step": len(step.dataset) if step else 0,
                "update_seconds": update_seconds,
                "test_seconds": test_seconds,
                "eval_set": eval_set,
            }
            for view_name, view in protocol.views.items():
                metrics = (
                    result.metrics
                    if view is P.original
                    else score(
                        view(result.predictions),
                        per_defect=module.per_defect,
                        n_bins=module.n_bins,
                        fpr_limit=module.fpr_limit,
                        pro_connectivity=module.pro_connectivity,
                    )
                )
                for row in metrics.to_dict("records"):
                    outcome["rows"].append(
                        {
                            **phase_fields,
                            "view": view_name,
                            **row,
                            **(overall if row["defect_type"] == "all" else {}),
                        }
                    )
            outcome["scores"][f"phase{phase}_{eval_set}"] = result.predictions.scores_frame()
            outcome["artifacts"].update(
                {f"phase{phase}/{eval_set}/{name}": data for name, data in result.artifacts.items()}
            )

    def record(self, outcome: dict[str, Any]) -> None:
        """Append an outcome of `run_one` to ``results.jsonl``, ``scores/`` and ``artifacts/``."""
        rows = [_jsonable(row) for row in outcome["rows"]]
        if self.output_dir is None:
            self._rows += rows
            return
        name = outcome["run"].name
        # Before results.jsonl, so a run counted as done always has its files.
        for path, data in outcome["artifacts"].items():
            target = self.output_dir / "artifacts" / name / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        scores_dir = self.output_dir / "scores" / name
        scores_dir.mkdir(parents=True, exist_ok=True)
        for phase, frame in outcome["scores"].items():
            frame.to_csv(scores_dir / f"{phase}.csv", index=False)
        with self.results_path.open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def _update_seed(run: MaintRun) -> int:
    """The global seed for ``run``'s updates, from both of its seeds."""
    return int(np.random.SeedSequence([run.seed, run.update_seed]).generate_state(1)[0] % 2**31)


def _run_params(run: MaintRun) -> dict[str, Any]:
    params = asdict(run)
    params["shots"] = "full" if run.shots is None else str(run.shots)
    return params


# Columns of a row that are not metrics.
_NOT_METRICS = (*RUN_FIELDS, "scenario", "phase", "step", "op", "eval_set", "view", "defect_type")


def mlflow_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """A run's rows as MLflow metrics named ``phase<i>/<eval set>/<view>/<defect type>/<column>``."""
    import re

    metrics = {}
    for row in rows:
        defect = re.sub(r"[^\w\-. ]", "_", row["defect_type"])
        prefix = f"phase{row['phase']}/{row['eval_set']}/{row['view']}/{defect}"
        for column, value in row.items():
            if column in _NOT_METRICS or isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                metrics[f"{prefix}/{column}"] = float(value)
    return metrics


def before_after(
    results: pd.DataFrame,
    metric: str = "image_auroc",
    *,
    view: str = "original",
    defect_type: str = "all",
) -> pd.DataFrame:
    """Mean and std over seeds of ``metric`` at phase 0, at the last phase and of their
    difference, per run setting and eval set; update seeds are averaged within each seed.

    ``defect_type="target"`` takes each run's own target defect type.
    """
    rows = results[results["view"] == view]
    if defect_type == "target":
        rows = rows[rows["defect_type"] == rows["target"]]
    else:
        rows = rows[rows["defect_type"] == defect_type]
    keys = [*RUN_FIELDS, "eval_set"]
    last = rows.groupby(keys, dropna=False)["phase"].transform("max")
    before = rows[rows["phase"] == 0][[*keys, metric]]
    after = rows[rows["phase"] == last][[*keys, metric]]
    merged = before.merge(after, on=keys, suffixes=("_before", "_after"))
    merged = merged.rename(columns={f"{metric}_before": "before", f"{metric}_after": "after"})
    merged["delta"] = merged["after"] - merged["before"]
    values = ["before", "after", "delta"]
    per_seed = merged.groupby([k for k in keys if k != "update_seed"], dropna=False)[values].mean()
    group = [k for k in keys if k not in ("seed", "update_seed")]
    return per_seed.groupby(group, dropna=False).agg(["mean", "std"])
