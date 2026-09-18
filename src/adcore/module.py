"""`AnomalyModule`: a ``LightningModule`` that knows how to be evaluated.

A subclass implements ``forward(images) -> {"anomaly_map", "pred_score"}`` and trains
however Lightning trains it — ``training_step`` + ``configure_optimizers`` for gradient
training, or a no-optimizer pass that fills a memory bank (see `adcore.detectors.PatchCore`).
``validation_step`` / ``test_step`` are provided: they collect every batch, and at epoch
end the predictions are scored per category and defect type, logged, and kept as
``val_result`` / ``test_result``.

Logged metrics are ``{stage}/{metric}`` averaged over categories, and
``{stage}/{category}/{metric}`` for each category — so a checkpoint callback can monitor
``val/image_auroc``.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch

from adcore.evaluation import (
    METRICS,
    EvalResult,
    PredictionCollector,
    score,
)
from adcore.metrics import DEFAULT_FPR_LIMIT, DEFAULT_N_BINS


class AnomalyModule(L.LightningModule):
    """Base class for anomaly detectors trained and evaluated with Lightning.

    Args:
        per_defect: Also score each defect type (with the good frames).
        n_bins: Histogram bins for the pixel metrics.
        fpr_limit: FPR limit for AUPRO.
    """

    def __init__(
        self,
        per_defect: bool = True,
        n_bins: int = DEFAULT_N_BINS,
        fpr_limit: float = DEFAULT_FPR_LIMIT,
    ) -> None:
        super().__init__()
        self.per_defect = per_defect
        self.n_bins = n_bins
        self.fpr_limit = fpr_limit
        self.val_result: EvalResult | None = None
        self.test_result: EvalResult | None = None
        self._collectors: dict[str, PredictionCollector] = {}

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Score ``(B, 3, H, W)`` images.

        Returns at least ``"anomaly_map"`` — ``(B, 1, h, w)`` or ``(B, h, w)``, resized
        to the mask resolution by the evaluator if it differs — and ``"pred_score"``, ``(B,)``.
        """
        raise NotImplementedError

    # --- evaluation -----------------------------------------------------------------

    def _collect(self, stage: str, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        output = self(batch["image"])
        self._collectors.setdefault(stage, PredictionCollector()).add(batch, output)
        return output

    def _finish(self, stage: str) -> EvalResult | None:
        collector = self._collectors.pop(stage, None)
        if collector is None:
            return None
        if self.trainer.world_size > 1:
            raise NotImplementedError(
                "AnomalyModule scores on a single device; evaluate with devices=1"
            )
        predictions = collector.finish()
        metrics = score(
            predictions,
            per_defect=self.per_defect,
            n_bins=self.n_bins,
            fpr_limit=self.fpr_limit,
        )
        result = EvalResult(metrics=metrics, predictions=predictions)

        overall = result.overall
        logged = {f"{stage}/{m}": float(overall[m].mean()) for m in METRICS}
        if len(overall) > 1:
            for row in overall.to_dict("records"):
                logged.update(
                    {f"{stage}/{row['category']}/{m}": float(row[m]) for m in METRICS}
                )
        self.log_dict(logged, batch_size=len(predictions))
        return result

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._collect("val", batch)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self._collectors.pop("val", None)
            return
        self.val_result = self._finish("val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._collect("test", batch)

    def on_test_epoch_end(self) -> None:
        self.test_result = self._finish("test")

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, torch.Tensor]:
        return self(batch["image"])
