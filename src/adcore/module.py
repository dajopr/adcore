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

A module can also track its own statistics with ``stats``: a torchmetrics
``MetricCollection`` whose metrics take ``update(batch, output)`` — the batch dict and the
full ``forward`` output — and return one scalar from ``compute()``. They are updated every
val / test batch, computed over the whole eval set, logged as ``{stage}/stat/{name}`` and
kept on ``val_result.stats`` / ``test_result.stats``.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torchmetrics import Metric, MetricCollection

from adcore.evaluation import (
    METRICS,
    STAT,
    EvalResult,
    PredictionCollector,
    score,
)
from adcore.metrics import (
    DEFAULT_FPR_LIMIT,
    DEFAULT_N_BINS,
    DEFAULT_PRO_CONNECTIVITY,
    region_structure,
)


class AnomalyModule(L.LightningModule):
    """Base class for anomaly detectors trained and evaluated with Lightning.

    Args:
        per_defect: Also score each defect type (with the good frames).
        n_bins: Histogram bins for the pixel metrics.
        fpr_limit: FPR limit for AUPRO.
        pro_connectivity: 4 (MVTec's official convention) or 8 (SubspaceAD, anomalib)
            connected defect regions for AUPRO.
        stats: Statistics of the module's own, for both val and test: a
            ``MetricCollection`` (or a dict of ``Metric``) whose metrics take
            ``update(batch, output)`` and return one scalar from ``compute()``. Each stage
            gets its own copy.
        val_stats: Replaces ``stats`` for validation.
        test_stats: Replaces ``stats`` for testing.
    """

    def __init__(
        self,
        per_defect: bool = True,
        n_bins: int = DEFAULT_N_BINS,
        fpr_limit: float = DEFAULT_FPR_LIMIT,
        pro_connectivity: int = DEFAULT_PRO_CONNECTIVITY,
        stats: MetricCollection | dict[str, Metric] | None = None,
        val_stats: MetricCollection | dict[str, Metric] | None = None,
        test_stats: MetricCollection | dict[str, Metric] | None = None,
    ) -> None:
        super().__init__()
        region_structure(pro_connectivity)  # fail at construction, not after testing
        self.per_defect = per_defect
        self.n_bins = n_bins
        self.fpr_limit = fpr_limit
        self.pro_connectivity = pro_connectivity
        self.val_result: EvalResult | None = None
        self.test_result: EvalResult | None = None
        self._collectors: dict[str, PredictionCollector] = {}
        # Registered submodules, so their states follow the module's device.
        self.val_stats = _collection(val_stats if val_stats is not None else stats)
        self.test_stats = _collection(test_stats if test_stats is not None else stats)

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
        stats = self._stats(stage)
        if stats is not None:
            stats.update(batch, output)
        return output

    def _stats(self, stage: str) -> MetricCollection | None:
        return getattr(self, f"{stage}_stats", None)

    def _compute_stats(self, stage: str) -> dict[str, float]:
        stats = self._stats(stage)
        if stats is None:
            return {}
        values = {}
        for name, value in stats.compute().items():
            value = torch.as_tensor(value)
            if value.numel() != 1:
                raise ValueError(
                    f"stat {name!r} computed a {tuple(value.shape)} tensor; return one "
                    "scalar per metric (split a vector into one metric per reduction)"
                )
            values[name] = float(value)
        stats.reset()
        return values

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
            pro_connectivity=self.pro_connectivity,
        )
        result = EvalResult(
            metrics=metrics, predictions=predictions, stats=self._compute_stats(stage)
        )

        overall = result.overall
        logged = {f"{stage}/{m}": float(overall[m].mean()) for m in METRICS}
        if len(overall) > 1:
            for row in overall.to_dict("records"):
                logged.update(
                    {f"{stage}/{row['category']}/{m}": float(row[m]) for m in METRICS}
                )
        logged.update({f"{stage}/{STAT}{name}": value for name, value in result.stats.items()})
        self.log_dict(logged, batch_size=len(predictions))
        return result

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._collect("val", batch)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self._collectors.pop("val", None)
            if self.val_stats is not None:
                self.val_stats.reset()
            return
        self.val_result = self._finish("val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._collect("test", batch)

    def on_test_epoch_end(self) -> None:
        self.test_result = self._finish("test")

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, torch.Tensor]:
        return self(batch["image"])


def _collection(
    stats: MetricCollection | dict[str, Metric] | None,
) -> MetricCollection | None:
    """A fresh copy, so stages (and modules sharing a config) never share state."""
    if stats is None:
        return None
    if not isinstance(stats, MetricCollection):
        stats = MetricCollection(dict(stats))
    return stats.clone()
