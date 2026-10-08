"""Core anomaly detection on Lightning: PatchCore, MVTec AD / VisA loading, metrics, evaluation,
few-shot experiments and their tracking in MLflow."""

from adcore.datamodule import (
    AugmentedSupport,
    AugmentSpec,
    CachedDataset,
    MVTecDataModule,
    default_rotation,
)
from adcore.detectors import PatchCore, upsample_and_blur
from adcore.evaluation import EvalResult, Predictions, evaluate, score
from adcore.experiment import FewShotExperiment, Run, read_artifacts, summarize, table
from adcore.extractors import TimmExtractor
from adcore.metrics import ADMetrics, compute_metrics, image_aupr, region_structure
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
from adcore.stats import OutputMean
from adcore.tracking import MLflowTracking, load_artifacts, load_runs, upload_sweep

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
    "MLflowTracking",
    "MVTEC",
    "MVTecDataModule",
    "MVTecDataset",
    "OutputMean",
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
    "image_aupr",
    "load_artifacts",
    "load_runs",
    "score",
    "split_by_defect_type",
    "read_artifacts",
    "region_structure",
    "summarize",
    "table",
    "upload_sweep",
    "upsample_and_blur",
]
