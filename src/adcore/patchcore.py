"""PatchCore memory bank and nearest-neighbour anomaly prediction.

Copied from pyanodet ``sherlock.models.hacked_patchcore.HackedPatchcoreModel``, with
the anomalib 2.2.0 ``PatchcoreModel`` static helpers it calls vendored in. Feature
extraction, foreground masking and rotation-invariant scoring are left out: ``forward``
takes an already extracted ``(B, C, H, W)`` embedding.
"""

import torch
import torch.nn as nn

from adcore.coreset import get_embedding_subsample


# --- anomalib.models.image.patchcore.torch_model.PatchcoreModel ---
# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Reshape ``(batch_size, embedding_dim, height, width)`` to
    ``(batch_size * height * width, embedding_dim)``."""
    embedding_size = embedding.size(1)
    return embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distances between ``(n, d)`` and ``(m, d)``, shape ``(n, m)``.

    Avoids ``torch.cdist()`` for better compatibility with ONNX export and OpenVINO
    conversion.
    """
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
    y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
    # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
    res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
    return res.clamp_min_(0).sqrt_()


class PatchcoreModel(nn.Module):
    def __init__(self, num_neighbors: int = 9):
        super().__init__()

        self.num_neighbors = num_neighbors

        self.memory_bank = torch.nn.Parameter(
            data=torch.Tensor([]), requires_grad=False
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The bank's size is only known once fitted, so a checkpoint's bank replaces the
        # empty placeholder instead of failing the shape check.
        bank = state_dict.get(prefix + "memory_bank")
        if bank is not None:
            self.memory_bank.data = torch.empty_like(bank, device=self.memory_bank.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, embedding: torch.Tensor):
        """Score an extracted embedding of shape (B, C, H, W) against the memory bank."""
        batch_size, _, width, height = embedding.shape

        embedding = reshape_embedding(embedding)

        # compare with memory bank via nearest neighbor search
        patch_scores, locations = self.nearest_neighbors(
            embedding=embedding, n_neighbors=1
        )

        # reshape to batch dimension
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))

        # compute anomaly score
        pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
        # reshape to w, h

        patch_scores = patch_scores.reshape((batch_size, 1, width, height))

        return {
            "anomaly_map": patch_scores,
            "pred_score": pred_score,
            "patch_scores": patch_scores,
            "nn_indices": locations,
            "embedding": embedding,
        }

    def subsample_embedding(
        self, embedding: torch.Tensor, sampling_ratio: float
    ) -> None:
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float | int): Coreset sampling ratio, or an absolute coreset
                size when an int. A bank as large as the embedding (e.g. ``1.0``) keeps
                every patch without the greedy search — the usual few-shot choice.
        """
        n_patches = embedding.shape[0]
        coreset_size = (
            int(n_patches * sampling_ratio)
            if isinstance(sampling_ratio, float)
            else sampling_ratio
        )
        if coreset_size >= n_patches:
            self.memory_bank.data = embedding
            return

        self.memory_bank.data = get_embedding_subsample(embedding, sampling_ratio)

    def compute_anomaly_score(
        self,
        patch_scores: torch.Tensor,
        locations: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Image-Level Anomaly Score.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores

        Returns:
            Tensor: Image-level anomaly scores
        """
        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to its nearest
        # neighbor in each image
        max_patches = torch.argmax(
            patch_scores, dim=1
        )  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[
            torch.arange(batch_size), max_patches
        ]
        # 2. Find the distance of the patch to it's nearest neighbor,
        # and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[
            torch.arange(batch_size), max_patches
        ]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        memory_bank_effective_size = self.memory_bank.shape[
            0
        ]  # edge case when memory bank is too small
        _, support_samples = self.nearest_neighbors(
            nn_sample,
            n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
        )
        # 4. Find the distance of the patch features to each of the support samples
        distances = euclidean_dist(
            max_patches_features.unsqueeze(1), self.memory_bank[support_samples]
        )
        # 5. Apply softmax to find the weights
        weights = (1 - nn.functional.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        return weights * score  # s in the paper

    def nearest_neighbors(
        self, embedding: torch.Tensor, n_neighbors: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest Neighbors using brute force method and euclidean norm.

        Args:
            embedding (torch.Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        distances = euclidean_dist(embedding, self.memory_bank)
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(
                k=n_neighbors, largest=False, dim=1
            )
        return patch_scores, locations
