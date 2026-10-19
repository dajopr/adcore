"""PatchCore whose memory bank can be edited after fitting: the reference `Maintainable`.

- ``add_representation`` appends the embeddings of the masked patches to the bank
  (without coreset subsampling).
- ``remove_representation`` deletes, for every masked patch, its ``remove_k`` nearest
  bank entries.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from adcore.detectors import PatchCore
from adcore.patchcore import euclidean_dist, reshape_embedding
from admaint.maintainable import Maintainable, patch_mask


class MaintainablePatchCore(PatchCore, Maintainable):
    """`adcore.PatchCore` with mask-based edits of its memory bank.

    Args:
        extractor: As for `adcore.PatchCore`.
        remove_k: Bank entries deleted per masked patch by ``remove_representation``.
        mask_overlap: Fraction of a patch's pixels that must be masked for the patch to
            count; 0 counts every patch the mask touches. See `patch_mask`.
        chunk_size: Query patches per distance computation when removing.
        **kwargs: Passed to `adcore.PatchCore`.
    """

    def __init__(
        self,
        extractor: Callable[[torch.Tensor], torch.Tensor],
        remove_k: int = 1,
        mask_overlap: float = 0.0,
        chunk_size: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(extractor, **kwargs)
        if remove_k < 1:
            raise ValueError(f"remove_k must be at least 1, got {remove_k}")
        self.remove_k = remove_k
        self.mask_overlap = mask_overlap
        self.chunk_size = chunk_size
        self.n_added = 0
        self.n_removed = 0

    def _patches(self, images: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
        """Embeddings ``(n, C)`` of the masked patches of ``images``."""
        embedding = self.extractor(images)
        patches = reshape_embedding(embedding)  # rows in (b, h, w) order
        keep = patch_mask(masks, tuple(embedding.shape[-2:]), self.mask_overlap)
        if keep is None:
            return patches
        return patches[keep.reshape(-1).to(patches.device)]

    @torch.no_grad()
    def add_representation(
        self, images: torch.Tensor, masks: torch.Tensor | None = None
    ) -> None:
        bank = self.model.memory_bank
        patches = self._patches(images, masks).to(bank.device, bank.dtype)
        bank.data = torch.cat([bank.data, patches])
        self.n_added += len(patches)

    @torch.no_grad()
    def remove_representation(
        self, images: torch.Tensor, masks: torch.Tensor | None = None
    ) -> None:
        bank = self.model.memory_bank
        queries = self._patches(images, masks).to(bank.device, bank.dtype)
        if not len(queries) or not len(bank):
            return
        k = min(self.remove_k, len(bank))
        drop = torch.zeros(len(bank), dtype=torch.bool, device=bank.device)
        for chunk in queries.split(self.chunk_size):
            _, nearest = euclidean_dist(chunk, bank.data).topk(k, dim=1, largest=False)
            drop[nearest.reshape(-1)] = True
        bank.data = bank.data[~drop]
        self.n_removed += int(drop.sum())

    def fit_state(self) -> dict[str, Any]:
        # The extractor is frozen; only the memory bank comes from fitting.
        return {"memory_bank": self.model.memory_bank.detach().clone()}

    def load_fit_state(self, state: dict[str, Any]) -> None:
        self.model.memory_bank.data = state["memory_bank"].clone()

    def maintenance_stats(self) -> dict[str, float]:
        return {
            "bank_size": len(self.model.memory_bank),
            "n_added": self.n_added,
            "n_removed": self.n_removed,
        }
