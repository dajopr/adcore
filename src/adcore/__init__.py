"""Core anomaly detection on Lightning: PatchCore, MVTec AD loading, metrics, evaluation
and few-shot experiments."""

from adcore.datamodule import CachedDataset, MVTecDataModule
from adcore.detectors import PatchCore, upsample_and_blur
from adcore.evaluation import EvalResult, Predictions, evaluate, score
from adcore.experiment import FewShotExperiment, Run, summarize, table
from adcore.extractors import TimmExtractor
from adcore.metrics import ADMetrics, compute_metrics
from adcore.module import AnomalyModule
from adcore.mvtec import CATEGORIES, MVTecDataset, few_shot_subset, split_by_defect_type
from adcore.patchcore import PatchcoreModel

__all__ = [
    "ADMetrics",
    "AnomalyModule",
    "CATEGORIES",
    "CachedDataset",
    "EvalResult",
    "FewShotExperiment",
    "MVTecDataModule",
    "MVTecDataset",
    "PatchCore",
    "PatchcoreModel",
    "Predictions",
    "Run",
    "TimmExtractor",
    "compute_metrics",
    "evaluate",
    "few_shot_subset",
    "score",
    "split_by_defect_type",
    "summarize",
    "table",
    "upsample_and_blur",
]
