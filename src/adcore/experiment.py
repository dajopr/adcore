"""Few-shot experiments on Lightning: every (category, shots, seed) builds a fresh
`AnomalyModule`, runs ``Trainer.fit`` on an `MVTecDataModule` and ``Trainer.test`` on the
category's test split.

    extractor = TimmExtractor("wide_resnet50_2", out_indices=(2, 3))
    experiment = FewShotExperiment(
        root="/data/shared-data/public_datasets/raw/MVTec",
        module_factory=lambda run: PatchCore(extractor, sampling_ratio=1.0),
        shots=(1, 2, 4, 8),
        seeds=(0, 1, 2),
        output_dir="runs/patchcore-wrn50",
    )
    results = experiment.run()      # one row per (category, shots, seed, defect_type)
    summarize(results)              # mean / std over seeds, plus the category mean
    table(results, "image_auroc")   # categories x shots, "mean ± std"

A trainable module works the same way; give the Trainer its epochs with
``trainer_kwargs={"max_epochs": 5}`` and, for supervision, ``anomalous_fraction``.

With an ``output_dir`` every finished run is appended to ``results.jsonl`` straight away,
and a rerun skips the runs already there, so an interrupted experiment resumes. Each
run's Lightning logs (training losses, test metrics) go to ``logs/<run>/``.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.model_helpers import is_overridden
from tqdm.auto import tqdm

from adcore.datamodule import CachedDataset, MVTecDataModule
from adcore.evaluation import METRICS
from adcore.module import AnomalyModule
from adcore.mvtec import CATEGORIES, MVTecDataset, default_transform


@dataclass(frozen=True)
class Run:
    category: str
    shots: int | None  # None: every training frame
    seed: int

    @property
    def key(self) -> tuple[str, int | None, int]:
        return (self.category, self.shots, self.seed)

    @property
    def name(self) -> str:
        return f"{self.category}_{'full' if self.shots is None else self.shots}shot_seed{self.seed}"


# Expected for few-shot memory-bank modules, and repeated once per run otherwise.
_QUIET = (
    r".*configure_optimizers.*returned.*None.*",
    r".*does not have many workers.*",
    r".*smaller than the logging interval.*",
)


@contextmanager
def _quiet_lightning():
    rank_zero = logging.getLogger("lightning.pytorch.utilities.rank_zero")
    level = rank_zero.level
    rank_zero.setLevel(logging.WARNING)
    with warnings.catch_warnings():
        for message in _QUIET:
            warnings.filterwarnings("ignore", message=message)
        try:
            yield
        finally:
            rank_zero.setLevel(level)


class FewShotExperiment:
    """Every combination of ``categories`` x ``shots`` x ``seeds`` on MVTec AD.

    Args:
        root: MVTec AD directory, one subdirectory per category.
        module_factory: Builds a fresh, unfitted `AnomalyModule` for a `Run`. It gets the
            run so a module can depend on its category (e.g. text prompts). Close over a
            shared extractor rather than building one per run.
        categories: Defaults to all fifteen.
        shots: Defect-free training frames per run; None means all of them.
        seeds: One run per seed. The seed picks the shots (and anomalous training
            frames) and goes to ``L.seed_everything`` before the module is built.
        output_dir: Where ``config.json``, ``results.jsonl``, per-image
            ``scores/<run>.csv`` and Lightning ``logs/<run>/`` go. None keeps results in
            memory only.
        trainer_kwargs: ``Trainer`` arguments, or a function of the `Run` returning them,
            over the defaults: one device, ``max_epochs=1``, CSV logging into
            ``output_dir``, no checkpointing, no progress bar.
        anomalous_fraction: Fraction of each defect type's test frames moved into
            training; see `MVTecDataModule`. 0 keeps the test split intact.
        image_size: Resolution images and masks are resized to, and so the resolution
            pixel metrics are computed at. Ignored when ``transform`` is given.
        transform: A torchvision v2 transform over ``(image, mask)``.
        cache_test_set: Decode each category's test split once and reuse it for every run.
    """

    def __init__(
        self,
        root: str | Path,
        module_factory: Callable[[Run], AnomalyModule],
        *,
        categories: Sequence[str] | None = None,
        shots: Sequence[int | None] = (1, 2, 4, 8),
        seeds: Sequence[int] = (0, 1, 2),
        output_dir: str | Path | None = None,
        trainer_kwargs: dict[str, Any] | Callable[[Run], dict[str, Any]] | None = None,
        anomalous_fraction: float = 0.0,
        image_size: tuple[int, int] = (224, 224),
        transform=None,
        batch_size: int = 32,
        num_workers: int = 4,
        cache_test_set: bool = True,
    ) -> None:
        self.root = Path(root)
        self.module_factory = module_factory
        self.categories = list(categories) if categories is not None else list(CATEGORIES)
        self.shots = list(shots)
        self.seeds = list(seeds)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.trainer_kwargs = trainer_kwargs or {}
        self.anomalous_fraction = anomalous_fraction
        self.image_size = tuple(image_size)
        self.transform = transform or default_transform(self.image_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_test_set = cache_test_set
        self._rows: list[dict[str, Any]] = []

    @property
    def runs(self) -> list[Run]:
        return [
            Run(category, shots, seed)
            for category in self.categories
            for shots in self.shots
            for seed in self.seeds
        ]

    @property
    def results_path(self) -> Path | None:
        return self.output_dir / "results.jsonl" if self.output_dir else None

    def config(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "categories": self.categories,
            "shots": self.shots,
            "seeds": self.seeds,
            "anomalous_fraction": self.anomalous_fraction,
            "image_size": list(self.image_size),
            "transform": repr(self.transform),
            "trainer_kwargs": repr(self.trainer_kwargs),
        }

    def _load_rows(self) -> list[dict[str, Any]]:
        if self.results_path is None or not self.results_path.exists():
            return list(self._rows)
        with self.results_path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def results(self) -> pd.DataFrame:
        """Every recorded row, including those from earlier sessions in ``output_dir``."""
        return results_frame(self._load_rows())

    def trainer(self, run: Run) -> L.Trainer:
        overrides = (
            self.trainer_kwargs(run) if callable(self.trainer_kwargs) else self.trainer_kwargs
        )
        logger = (
            CSVLogger(self.output_dir / "logs", name=run.name)
            if self.output_dir is not None
            else False
        )
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

    def run(self, progress: bool = True) -> pd.DataFrame:
        """Run everything not already in ``output_dir`` and return all results."""
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "config.json").write_text(
                json.dumps(self.config(), indent=2)
            )
        done = {
            (row["category"], row["shots"], row["seed"]) for row in self._load_rows()
        }
        todo = [run for run in self.runs if run.key not in done]

        bar = tqdm(total=len(todo), desc="Runs", disable=not progress)
        for category in self.categories:
            category_runs = [run for run in todo if run.category == category]
            if not category_runs:
                continue
            test = None
            if self.cache_test_set:
                test = CachedDataset(
                    MVTecDataset(self.root, category, "test", self.transform),
                    num_workers=self.num_workers,
                    batch_size=self.batch_size,
                )
            for run in category_runs:
                bar.set_postfix_str(run.name)
                self._record(self.run_one(run, test))
                bar.update()
        bar.close()
        return self.results()

    def datamodule(self, run: Run, test_dataset=None) -> MVTecDataModule:
        return MVTecDataModule(
            self.root,
            run.category,
            shots=run.shots,
            seed=run.seed,
            anomalous_fraction=self.anomalous_fraction,
            transform=self.transform,
            test_dataset=test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def run_one(self, run: Run, test_dataset=None) -> dict[str, Any]:
        """Fit and test one run; returns its metric rows and per-image scores."""
        with _quiet_lightning():
            L.seed_everything(run.seed, verbose=False)
            datamodule = self.datamodule(run, test_dataset)
            module = self.module_factory(run)
            trainer = self.trainer(run)

            start = perf_counter()
            # Zero-shot modules have nothing to fit.
            if is_overridden("training_step", module):
                trainer.fit(module, datamodule=datamodule)
            fit_seconds = perf_counter() - start

            start = perf_counter()
            trainer.test(module, datamodule=datamodule, verbose=False)
            test_seconds = perf_counter() - start

        result = module.test_result
        del module, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows = [
            {
                **asdict(run),
                "n_train": len(datamodule.train),
                "n_anomalous_train": len(datamodule.train) - len(datamodule.reference),
                **row,
                "fit_seconds": fit_seconds,
                "test_seconds": test_seconds,
            }
            for row in result.metrics.to_dict("records")
        ]
        return {"run": run, "rows": rows, "scores": result.predictions.scores_frame()}

    def _record(self, outcome: dict[str, Any]) -> None:
        rows = [_jsonable(row) for row in outcome["rows"]]
        if self.output_dir is None:
            self._rows += rows
            return
        with self.results_path.open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        scores_dir = self.output_dir / "scores"
        scores_dir.mkdir(exist_ok=True)
        outcome["scores"].to_csv(scores_dir / f"{outcome['run'].name}.csv", index=False)


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    """Plain python scalars, with NaN metrics as null so the file stays valid JSON."""
    out = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and np.isnan(value):
            value = None
        out[key] = value
    return out


def results_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["shots"] = frame["shots"].astype("Int64")  # <NA> is full-shot
    for metric in METRICS:
        frame[metric] = frame[metric].astype(float)
    return frame


def _shot_labels(shots: pd.Series) -> pd.Categorical:
    """Shot counts as ordered labels, full-shot ("full") last."""
    counts = sorted(int(s) for s in shots.dropna().unique())
    labels = shots.astype("string").fillna("full")
    order = [str(c) for c in counts] + (["full"] if shots.isna().any() else [])
    return pd.Categorical(labels, categories=order, ordered=True)


def summarize(
    results: pd.DataFrame, metrics: Sequence[str] = METRICS
) -> pd.DataFrame:
    """Mean and std over seeds per (shots, category), on each category's whole test set.

    The ``"mean"`` category averages over categories within each seed first and then
    takes mean and std over seeds, so its std is seed variance of the benchmark mean
    rather than spread between categories.
    """
    overall = results[results["defect_type"] == "all"].copy()
    overall["shots"] = _shot_labels(overall["shots"])
    categories = list(dict.fromkeys(overall["category"]))

    per_seed_mean = (
        overall.groupby(["shots", "seed"], observed=True)[list(metrics)]
        .mean()
        .reset_index()
        .assign(category="mean")
    )
    combined = pd.concat(
        [overall[["category", "shots", "seed", *metrics]], per_seed_mean],
        ignore_index=True,
    )
    combined["category"] = pd.Categorical(
        combined["category"], categories=[*categories, "mean"], ordered=True
    )
    return combined.groupby(["shots", "category"], observed=True)[list(metrics)].agg(
        ["mean", "std"]
    )


def table(
    results: pd.DataFrame, metric: str = "image_auroc", decimals: int = 1
) -> pd.DataFrame:
    """Categories x shots of ``"mean ± std"`` in percent for one metric."""
    stats = summarize(results, [metric])[metric]
    text = (stats["mean"] * 100).round(decimals).map(
        lambda m: f"{m:.{decimals}f}"
    ) + (stats["std"] * 100).map(
        lambda s: "" if pd.isna(s) else f" ± {s:.{decimals}f}"
    )
    return text.unstack("shots")
