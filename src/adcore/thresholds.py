"""Image and pixel thresholds chosen on a holdout split, per category.

A threshold rule is any callable taking the `adcore.evaluation.Predictions` of a fitted
module on the holdout and returning a `Thresholds` per category. `adcore.module.AnomalyModule`
applies its rule after every validation pass and scores the test split at the result:

    module = PatchCore(extractor, threshold_rule=QuantileThreshold(image_fpr=0.05))
    trainer.fit(module, datamodule=datamodule)
    trainer.validate(module, datamodule=datamodule)   # module.thresholds
    trainer.test(module, datamodule=datamodule)       # + image_f1, seg_f1, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from adcore.evaluation import Predictions

# What an image threshold is compared with: the module's ``pred_score``, or the maximum
# of its anomaly map.
ImageSource = Literal["score", "max_pixel"]


@dataclass(frozen=True)
class Thresholds:
    """A frame is anomalous when its image value is at or above ``image``, a pixel when
    its anomaly score is at or above ``pixel``."""

    image: float
    pixel: float
    image_source: ImageSource = "score"


def image_values(
    image_scores: np.ndarray, anomaly_maps: np.ndarray, source: ImageSource
) -> np.ndarray:
    """Per frame, what an image threshold is compared with."""
    if source == "score":
        return np.asarray(image_scores, dtype=np.float64)
    if source == "max_pixel":
        return anomaly_maps.reshape(len(anomaly_maps), -1).max(axis=1).astype(np.float64)
    raise ValueError(f"image_source must be 'score' or 'max_pixel', got {source!r}")


@dataclass(frozen=True)
class QuantileThreshold:
    """Thresholds at target false-alarm rates on the holdout's defect-free data.

    Only defect-free frames, and their defect-free pixels, are used, so a holdout that
    holds anomalies too still sets false-alarm rates.

    Args:
        image_fpr: Target fraction of defect-free frames flagged: the image threshold is
            the ``1 - image_fpr`` quantile of their image values.
        pixel_fpr: Target fraction of defect-free pixels flagged: the pixel threshold is
            the ``1 - pixel_fpr`` quantile of every such pixel's score.
        image_source: ``"score"`` thresholds ``pred_score``; ``"max_pixel"`` each frame's
            maximum anomaly-map value.
    """

    image_fpr: float = 0.05
    pixel_fpr: float = 0.01
    image_source: ImageSource = "score"

    def __post_init__(self) -> None:
        for name in ("image_fpr", "pixel_fpr"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")

    def __call__(self, predictions: Predictions) -> dict[str, Thresholds]:
        thresholds = {}
        for category in dict.fromkeys(predictions.categories):
            good = np.flatnonzero(
                (predictions.categories == category) & (predictions.labels == 0)
            )
            if len(good) == 0:
                continue
            maps = predictions.anomaly_maps[good]
            values = image_values(predictions.image_scores[good], maps, self.image_source)
            pixels = maps[predictions.masks[good] == 0]
            thresholds[category] = Thresholds(
                image=float(np.quantile(values, 1.0 - self.image_fpr)),
                pixel=float(np.quantile(pixels, 1.0 - self.pixel_fpr)),
                image_source=self.image_source,
            )
        return thresholds
