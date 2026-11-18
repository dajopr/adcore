"""Core anomaly detection on Lightning: PatchCore, MVTec AD / VisA / MVTec AD 2 loading, metrics, evaluation,
few-shot experiments and their tracking in MLflow."""

from adcore.bootstrap import paired_bootstrap
from adcore.datamodule import (
    AugmentedSupport,
    AugmentSpec,
    CachedDataset,
    MVTecDataModule,
    default_rotation,
)
from adcore.detectors import PatchCore, pool_image_score, upsample_and_blur
from adcore.evaluation import EvalResult, Predictions, best_f1_scores, evaluate, score
from adcore.experiment import FewShotExperiment, Run, read_artifacts, summarize, table
from adcore.extractors import TimmExtractor
from adcore.hf_extractor import HFViTExtractor
from adcore.metrics import ADMetrics, compute_metrics, image_aupr, region_structure
from adcore.module import AnomalyModule
from adcore.mvtec import (
    CATEGORIES,
    MVTEC,
    MVTEC_AD2,
    MVTEC_AD2_CATEGORIES,
    REALIAD,
    REALIAD_CATEGORIES,
    VISA,
    VISA_CATEGORIES,
    DatasetSpec,
    MVTecDataset,
    ResizeShortSide,
    default_transform,
    few_shot_subset,
    split_by_defect_type,
)
from adcore.patchcore import PatchcoreModel
from adcore.stats import OutputMean
from adcore.thresholds import QuantileThreshold, Thresholds
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
    "HFViTExtractor",
    "MLflowTracking",
    "MVTEC",
    "MVTEC_AD2",
    "MVTEC_AD2_CATEGORIES",
    "REALIAD",
    "REALIAD_CATEGORIES",
    "MVTecDataModule",
    "MVTecDataset",
    "OutputMean",
    "PatchCore",
    "PatchcoreModel",
    "QuantileThreshold",
    "ResizeShortSide",
    "Predictions",
    "Run",
    "Thresholds",
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
    "paired_bootstrap",
    "best_f1_scores",
    "pool_image_score",
    "score",
    "split_by_defect_type",
    "read_artifacts",
    "region_structure",
    "summarize",
    "table",
    "upload_sweep",
    "upsample_and_blur",
]
