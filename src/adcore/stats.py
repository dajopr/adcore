"""Ready-made statistics for `AnomalyModule` ``stats``.

A stat is a torchmetrics ``Metric`` whose ``update`` takes ``(batch, output)``: the batch
dict and everything ``forward`` returned. Write your own the same way:

    class EnergyFractionRatioMean(Metric):
        def __init__(self, n_components):
            super().__init__()
            self.add_state("sums", torch.zeros(2, n_components, dtype=torch.float64), dist_reduce_fx="sum")
            ...
        def update(self, batch, output): ...
        def compute(self): ...      # one scalar
"""

from __future__ import annotations

from typing import Any

import torch
from torchmetrics import Metric


class OutputMean(Metric):
    """Mean of ``output[key]`` over every element of the eval set.

    A ``(B,)`` output gives the mean over images; a ``(B, ...)`` one the mean over all its
    elements, e.g. patches.
    """

    full_state_update = False

    def __init__(self, key: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.key = key
        self.add_state("total", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, batch: dict[str, Any], output: dict[str, torch.Tensor]) -> None:
        values = output[self.key].detach().double()
        self.total += values.sum()
        self.count += values.numel()

    def compute(self) -> torch.Tensor:
        return self.total / self.count
