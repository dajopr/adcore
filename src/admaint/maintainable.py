"""The maintenance interface: a fitted `adcore.AnomalyModule` updated in place.

A detector that can be maintained mixes in `Maintainable` and implements two operations
on what it considers nominal:

- ``add_representation(images, masks)``: the masked regions are nominal from now on;
- ``remove_representation(images, masks)``: the masked regions are not nominal (any more).

``images`` are a ``(B, 3, H, W)`` batch on the module's device, transformed like the
eval set; ``masks`` are ``(B, 1, H, W)`` 0/1 masks at the same resolution, or None for the
whole image. A module maps them to its own resolution, e.g. with `patch_mask`.

    class MyDetector(AnomalyModule, Maintainable):
        def add_representation(self, images, masks=None): ...
        def remove_representation(self, images, masks=None): ...
        def maintenance_stats(self): return {"bank_size": ...}

A maintenance experiment fits a module once and restores the fit into fresh modules for
every run that shares it (see `Maintainable.fit_state`).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class Maintainable:
    """Mixin for an `adcore.AnomalyModule` that can be updated after fitting."""

    def add_representation(
        self, images: torch.Tensor, masks: torch.Tensor | None = None
    ) -> None:
        """Make the masked regions of ``images`` (all of them when ``masks`` is None) nominal."""
        raise NotImplementedError

    def remove_representation(
        self, images: torch.Tensor, masks: torch.Tensor | None = None
    ) -> None:
        """Stop treating the masked regions of ``images`` (all of them when ``masks`` is
        None) as nominal."""
        raise NotImplementedError

    def maintenance_stats(self) -> dict[str, float]:
        """Numbers describing the module's state, recorded after every phase (e.g. the
        size of a memory bank); none by default."""
        return {}

    def fit_state(self) -> dict[str, Any]:
        """What fitting produced, as copies that later updates cannot change.

        `load_fit_state` puts it into a fresh, unfitted module built for a run with the
        same fit, instead of fitting again. The default copies the whole ``state_dict``,
        frozen backbones included; override both to keep only what fitting changes, and
        include any fitted state that lives outside ``state_dict``.
        """
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_fit_state(self, state: dict[str, Any]) -> None:
        """Restore a `fit_state` into this freshly built module."""
        self.load_state_dict({k: v.clone() for k, v in state.items()})


def patch_mask(
    masks: torch.Tensor | None, size: tuple[int, int], overlap: float = 0.0
) -> torch.Tensor | None:
    """``(B, 1, H, W)`` or ``(B, H, W)`` pixel masks as ``(B, h, w)`` bool patch masks.

    A patch is selected when more than ``overlap`` of its pixels are masked; 0 selects
    every patch the mask touches. None stays None (every patch).
    """
    if masks is None:
        return None
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    return F.adaptive_avg_pool2d((masks > 0).float(), tuple(size))[:, 0] > overlap
