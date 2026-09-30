"""PatchCore as an `AnomalyModule`.

Fitting is one Lightning epoch without an optimizer: ``training_step`` collects the
embeddings of the defect-free frames in each batch and ``on_train_epoch_end`` builds the
memory bank from them.

    module = PatchCore(extractor)
    datamodule = MVTecDataModule(root, "bottle", shots=4)
    trainer = L.Trainer(max_epochs=1, devices=1)
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)   # then module.test_result
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.transforms.v2.functional import gaussian_blur

from adcore.module import AnomalyModule
from adcore.patchcore import PatchcoreModel, reshape_embedding


def upsample_and_blur(
    patch_scores: torch.Tensor,
    size: tuple[int, int],
    sigma: float,
    kernel_size: int | None = None,
) -> torch.Tensor:
    """PatchCore's anomaly map: bilinear upsampling, then Gaussian smoothing.

    ``kernel_size`` None is the full kernel, ``2 * int(4 * sigma + 0.5) + 1`` (33 at
    sigma 4). SubspaceAD uses ``cv2.GaussianBlur(map, (3, 3), 4)``, i.e. ``kernel_size=3``
    — close to a 3x3 box filter. Upsampling and reflect padding match cv2's
    ``INTER_LINEAR`` and ``BORDER_REFLECT_101``.
    """
    anomaly_map = F.interpolate(patch_scores, size=size, mode="bilinear", align_corners=False)
    if sigma > 0:
        kernel = kernel_size or 2 * int(4 * sigma + 0.5) + 1
        anomaly_map = gaussian_blur(anomaly_map, kernel_size=[kernel, kernel], sigma=[sigma])
    return anomaly_map


class PatchCore(AnomalyModule):
    """PatchCore over any feature extractor.

    Args:
        extractor: Maps ``(B, 3, H, W)`` images to a ``(B, C, h, w)`` embedding, e.g.
            `adcore.extractors.TimmExtractor`. Sharing one instance across modules
            avoids reloading the backbone on every run of an experiment.
        sampling_ratio: Coreset ratio (or absolute size when an int). ``1.0`` keeps
            every patch, the usual choice when few shots make the bank small anyway.
        num_neighbors: Neighbours used to reweight the image score; 1 is the raw max.
        blur_sigma: Gaussian smoothing of the upsampled anomaly map; 0 disables it.
        blur_kernel_size: Odd kernel size for that smoothing; None is the full kernel.
            3 matches SubspaceAD.
        **kwargs: Passed to `AnomalyModule`.
    """

    def __init__(
        self,
        extractor: Callable[[torch.Tensor], torch.Tensor],
        sampling_ratio: float | int = 0.1,
        num_neighbors: int = 9,
        blur_sigma: float = 4.0,
        blur_kernel_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.extractor = extractor
        self.model = PatchcoreModel(num_neighbors=num_neighbors)
        self.sampling_ratio = sampling_ratio
        self.blur_sigma = blur_sigma
        self.blur_kernel_size = blur_kernel_size
        self._embeddings: list[torch.Tensor] = []

    def configure_optimizers(self) -> None:
        return None

    @torch.no_grad()
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        # Anomalous training frames (from `MVTecDataModule(anomalous_fraction=...)`) must
        # not enter the memory bank.
        images = batch["image"][torch.as_tensor(batch["label"]) == 0]
        if len(images):
            self._embeddings.append(reshape_embedding(self.extractor(images)))

    def on_train_epoch_end(self) -> None:
        if not self._embeddings:
            raise ValueError("PatchCore saw no defect-free training frames")
        self.model.subsample_embedding(torch.cat(self._embeddings), self.sampling_ratio)
        self._embeddings = []

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.model(self.extractor(images))
        return {
            "anomaly_map": upsample_and_blur(
                output["anomaly_map"],
                tuple(images.shape[-2:]),
                self.blur_sigma,
                self.blur_kernel_size,
            ),
            "pred_score": output["pred_score"],
        }
