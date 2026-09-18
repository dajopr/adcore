"""Predictions collected over an eval set, and their metrics per category and defect type.

The loop itself is Lightning's: `adcore.module.AnomalyModule` feeds every test / val batch
into a `PredictionCollector` and scores the result at epoch end. `evaluate` is the
one-call version for a module that is already fitted:

    result = evaluate(module, test_loader)
    result.metrics          # one row per (category, defect_type), "all" first
    result.predictions      # the collected labels, scores, masks and maps
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from adcore.metrics import (
    DEFAULT_FPR_LIMIT,
    DEFAULT_N_BINS,
    build_pixel_histograms,
    image_auroc,
    pixel_metrics,
)

if TYPE_CHECKING:
    from adcore.module import AnomalyModule

METRICS = ("image_auroc", "pixel_auroc", "pixel_aupr", "aupro")


@dataclass
class Predictions:
    """Everything a detector produced on an eval set, aligned by row."""

    labels: np.ndarray  # (N,) int — 0 good, 1 anomalous
    image_scores: np.ndarray  # (N,) float64
    masks: np.ndarray  # (N, H, W) uint8
    anomaly_maps: np.ndarray  # (N, H, W) float32
    categories: np.ndarray  # (N,) str
    defect_types: np.ndarray  # (N,) str
    image_paths: np.ndarray  # (N,) str

    def __len__(self) -> int:
        return len(self.labels)

    def scores_frame(self) -> pd.DataFrame:
        """Per-image scores, without the maps."""
        return pd.DataFrame(
            {
                "image_path": self.image_paths,
                "category": self.categories,
                "defect_type": self.defect_types,
                "label": self.labels,
                "image_score": self.image_scores,
            }
        )


@dataclass
class EvalResult:
    metrics: pd.DataFrame
    predictions: Predictions

    @property
    def overall(self) -> pd.DataFrame:
        """One row per category: the metrics over its whole test set."""
        return self.metrics[self.metrics["defect_type"] == "all"].reset_index(drop=True)


def _as_maps(anomaly_map: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """``(B, 1, h, w)`` or ``(B, h, w)`` maps as float32 ``(B, *size)``."""
    anomaly_map = anomaly_map.detach().float()
    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.unsqueeze(1)
    if tuple(anomaly_map.shape[-2:]) != tuple(size):
        anomaly_map = F.interpolate(
            anomaly_map, size=size, mode="bilinear", align_corners=False
        )
    return anomaly_map[:, 0].cpu().numpy()


@dataclass
class PredictionCollector:
    """Accumulates batches and their outputs on the CPU until `finish`.

    Anomaly maps are resized to the mask resolution when they differ.
    """

    _parts: dict[str, list] = field(default_factory=dict)

    def add(self, batch: dict[str, Any], output: dict[str, torch.Tensor]) -> None:
        masks = batch["mask"]
        if masks.ndim == 4:
            masks = masks[:, 0]
        part = {
            "labels": torch.as_tensor(batch["label"]).cpu().numpy().astype(np.int64),
            "image_scores": output["pred_score"].detach().double().reshape(-1).cpu().numpy(),
            "masks": (masks > 0).to(torch.uint8).cpu().numpy(),
            "anomaly_maps": _as_maps(output["anomaly_map"], tuple(masks.shape[-2:])),
            "categories": np.asarray(batch["category"]),
            "defect_types": np.asarray(batch["defect_type"]),
            "image_paths": np.asarray(batch["image_path"]),
        }
        for key, value in part.items():
            self._parts.setdefault(key, []).append(value)

    def finish(self) -> Predictions:
        if not self._parts:
            raise ValueError("no batches were collected")
        predictions = Predictions(
            **{key: np.concatenate(parts) for key, parts in self._parts.items()}
        )
        self._parts = {}
        return predictions


def score(
    predictions: Predictions,
    *,
    per_defect: bool = True,
    n_bins: int = DEFAULT_N_BINS,
    fpr_limit: float = DEFAULT_FPR_LIMIT,
) -> pd.DataFrame:
    """Metrics per category over its whole test set (``defect_type == "all"``) and, with
    ``per_defect``, over each defect type together with the category's good frames.

    Histograms are built per category so every category gets the full bin resolution
    over its own score range.
    """
    rows = []
    for category in dict.fromkeys(predictions.categories):
        in_category = np.flatnonzero(predictions.categories == category)
        histograms = build_pixel_histograms(
            predictions.masks[in_category],
            predictions.anomaly_maps[in_category],
            n_bins=n_bins,
        )
        defect_types = predictions.defect_types[in_category]
        labels = predictions.labels[in_category]
        image_scores = predictions.image_scores[in_category]

        groups = [("all", np.arange(len(in_category)))]
        if per_defect:
            for defect_type in sorted(set(defect_types) - {"good"}):
                groups.append(
                    (
                        defect_type,
                        np.flatnonzero(
                            (defect_types == defect_type) | (defect_types == "good")
                        ),
                    )
                )

        for defect_type, rows_in in groups:
            pixel = pixel_metrics(histograms, rows_in, fpr_limit=fpr_limit)
            rows.append(
                {
                    "category": category,
                    "defect_type": defect_type,
                    "n_images": len(rows_in),
                    "n_anomalous": int(labels[rows_in].sum()),
                    "image_auroc": image_auroc(labels[rows_in], image_scores[rows_in]),
                    "pixel_auroc": pixel.auroc,
                    "pixel_aupr": pixel.aupr,
                    "aupro": pixel.aupro,
                }
            )
    return pd.DataFrame(rows)


def evaluate(
    module: AnomalyModule,
    dataloaders: Iterable[dict[str, Any]] | Any,
    **trainer_kwargs: Any,
) -> EvalResult:
    """``Trainer.test`` a fitted module and return its `EvalResult`.

    ``dataloaders`` is a ``DataLoader`` or a ``LightningDataModule``; ``trainer_kwargs``
    go to the ``Trainer`` (defaults: one device, no logger, no progress bar).
    """
    trainer = L.Trainer(
        **{
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            **trainer_kwargs,
        }
    )
    if isinstance(dataloaders, L.LightningDataModule):
        trainer.test(module, datamodule=dataloaders, verbose=False)
    else:
        trainer.test(module, dataloaders=dataloaders, verbose=False)
    return module.test_result
