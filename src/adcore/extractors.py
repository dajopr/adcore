"""Frozen backbones mapping images ``(B, 3, H, W)`` to a patch embedding ``(B, C, h, w)``.

Any callable with that signature can stand in for these: `adcore.detectors.PatchCore`
only ever calls ``extractor(images)``.
"""

from collections.abc import Sequence

import timm
import torch
import torch.nn.functional as F
from torch import nn


class TimmExtractor(nn.Module):
    """A frozen timm backbone's feature levels, resized to the first level's grid and
    concatenated along channels.

    Args:
        model_name: Any timm name, e.g. ``"wide_resnet50_2"`` or
            ``"hf-hub:timm/vit_small_plus_patch16_dinov3.lvd1689m"``.
        out_indices: Feature levels to keep, passed to timm's ``features_only``. For
            CNNs these are stages (PatchCore uses ``(2, 3)``); for ViTs they are blocks.
            None takes timm's default.
        pool_size: Side of PatchCore's local neighbourhood average applied to the
            embedding; ``1`` disables it.
        pretrained: Load pretrained weights.
    """

    def __init__(
        self,
        model_name: str,
        out_indices: Sequence[int] | None = None,
        pool_size: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if pool_size % 2 == 0:
            raise ValueError(f"pool_size must be odd to keep the grid size, got {pool_size}")
        kwargs = {} if out_indices is None else {"out_indices": tuple(out_indices)}
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, **kwargs
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.pool_size = pool_size

    def train(self, mode: bool = True) -> "TimmExtractor":
        # Frozen: batch-norm statistics and dropout stay in eval mode whatever the parent does.
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        grid = features[0].shape[-2:]
        embedding = torch.cat(
            [
                f
                if f.shape[-2:] == grid
                else F.interpolate(f, size=grid, mode="bilinear", align_corners=False)
                for f in features
            ],
            dim=1,
        )
        if self.pool_size > 1:
            embedding = F.avg_pool2d(
                embedding, self.pool_size, stride=1, padding=self.pool_size // 2
            )
        return embedding
