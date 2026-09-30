"""Core anomaly detection on Lightning: PatchCore, MVTec AD / VisA loading, metrics, evaluation
and few-shot experiments."""

from adcore.datamodule import (
    AugmentedSupport,
    AugmentSpec,
    CachedDataset,
    MVTecDataModule,
    default_rotation,
)
from adcore.detectors import PatchCore, upsample_and_blur
from adcore.evaluation import EvalResult, Predictions, evaluate, score
from adcore.experiment import FewShotExperiment, Run, summarize, table
from adcore.extractors import TimmExtractor
from adcore.metrics import ADMetrics, compute_metrics
from adcore.module import AnomalyModule
from adcore.mvtec import (
    CATEGORIES,
    MVTEC,
    VISA,
    VISA_CATEGORIES,
    DatasetSpec,
    MVTecDataset,
    default_transform,
    few_shot_subset,
    split_by_defect_type,
)
from adcore.patchcore import PatchcoreModel

__all__ = [
    "ADMetrics",
    "AnomalyModule",
    "AugmentSpec",
    "AugmentedSupport",
    "CATEGORIES",
    "CachedDataset",
    "DatasetSpec",
    "EvalResult",
    "FewShotExperiment",
    "MVTEC",
    "MVTecDataModule",
    "MVTecDataset",
    "PatchCore",
    "PatchcoreModel",
    "Predictions",
    "Run",
    "TimmExtractor",
    "VISA",
    "VISA_CATEGORIES",
    "compute_metrics",
    "default_rotation",
    "default_transform",
    "evaluate",
    "few_shot_subset",
    "score",
    "split_by_defect_type",
    "summarize",
    "table",
    "upsample_and_blur",
]
