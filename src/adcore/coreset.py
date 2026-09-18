"""k-center greedy coreset subsampling.

Copied from pyanodet (``sherlock.models.components.kcenter`` and
``HackedPatchcoreModel.get_embedding_subsample``), with the anomalib 2.2.0 base
classes ``KCenterGreedy`` and ``SparseRandomProjection`` it builds on vendored in.
"""

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.utils.random import sample_without_replacement
from tqdm import tqdm


# --- anomalib.models.components.dimensionality_reduction.random_projection ---
# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class NotFittedError(ValueError, AttributeError):
    """Exception raised when model is used before fitting."""


class SparseRandomProjection:
    """Sparse Random Projection using PyTorch operations.

    Args:
        eps (float, optional): Minimum distortion rate parameter for calculating
            Johnson-Lindenstrauss minimum dimensions. Defaults to ``0.1``.
        random_state (int | None, optional): Seed for random number generation.
            Used for reproducible results. Defaults to ``None``.

    References:
        .. [1] P. Li, T. Hastie and K. Church, "Very Sparse Random Projections,"
           KDD '06, 2006.
    """

    def __init__(self, eps: float = 0.1, random_state: int | None = None) -> None:
        self.n_components: int
        self.sparse_random_matrix: torch.Tensor
        self.eps = eps
        self.random_state = random_state

    def _sparse_random_matrix(self, n_features: int) -> torch.Tensor:
        """Generate a sparse random matrix of shape ``(n_components, n_features)``,
        stored in dense format for GPU compatibility."""
        # Density 'auto'. Factorize density
        density = 1 / np.sqrt(n_features)

        if density == 1:
            # skip index generation if totally dense
            binomial = torch.distributions.Binomial(total_count=1, probs=0.5)
            components = binomial.sample((self.n_components, n_features)) * 2 - 1
            components = 1 / np.sqrt(self.n_components) * components

        else:
            # Sparse matrix is not being generated here as it is stored as dense anyways
            components = torch.zeros(
                (self.n_components, n_features), dtype=torch.float32
            )
            for i in range(self.n_components):
                # find the indices of the non-zero components for row i
                nnz_idx = torch.distributions.Binomial(
                    total_count=n_features, probs=density
                ).sample()
                # get nnz_idx column indices
                c_idx = torch.tensor(
                    sample_without_replacement(
                        n_population=n_features,
                        n_samples=nnz_idx,
                        random_state=self.random_state,
                    ),
                    dtype=torch.int32,
                )
                data = (
                    torch.distributions.Binomial(total_count=1, probs=0.5).sample(
                        sample_shape=c_idx.size()
                    )
                    * 2
                    - 1
                )
                # assign data to only those columns
                components[i, c_idx] = data

            components *= np.sqrt(1 / density) / np.sqrt(self.n_components)

        return components

    @staticmethod
    def _johnson_lindenstrauss_min_dim(
        n_samples: int, eps: float = 0.1
    ) -> int | np.integer:
        """Find a 'safe' number of components for random projection."""
        denominator = (eps**2 / 2) - (eps**3 / 3)
        return (4 * np.log(n_samples) / denominator).astype(np.int64)

    def fit(self, embedding: torch.Tensor) -> "SparseRandomProjection":
        """Fit the random projection matrix to data of shape ``(n_samples, n_features)``."""
        n_samples, n_features = embedding.shape
        device = embedding.device

        self.n_components = self._johnson_lindenstrauss_min_dim(
            n_samples=n_samples, eps=self.eps
        )

        # Generate projection matrix
        # torch can't multiply directly on sparse matrix and moving sparse matrix to cuda throws error
        # (Could not run 'aten::empty_strided' with arguments from the 'SparseCsrCUDA' backend)
        # hence sparse matrix is stored as a dense matrix on the device
        self.sparse_random_matrix = self._sparse_random_matrix(
            n_features=n_features
        ).to(device)

        return self

    def transform(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project ``(n_samples, n_features)`` data to ``(n_samples, n_components)``."""
        if self.sparse_random_matrix is None:
            msg = "`fit()` has not been called on SparseRandomProjection yet."
            raise NotFittedError(msg)

        return embedding @ self.sparse_random_matrix.T.float()


# --- anomalib.models.components.sampling.k_center_greedy ---
# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Only the methods not overridden by KCenterGreedyExtended are kept.


class KCenterGreedy:
    """k-center-greedy method for coreset selection.

    Reference:
        - https://arxiv.org/abs/1708.00489
    """

    def reset_distances(self) -> None:
        """Reset minimum distances to None."""
        self.min_distances = None

    def get_new_idx(self) -> torch.Tensor:
        """Get index of the next sample based on maximum minimum distance.

        Returns:
            torch.Tensor: Index of the selected sample (tensor, not converted to int).

        Raises:
            TypeError: If `self.min_distances` is not a torch.Tensor.
        """
        if isinstance(self.min_distances, torch.Tensor):
            _, idx = torch.max(self.min_distances.squeeze(1), dim=0)
        else:
            msg = f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}"
            raise TypeError(msg)
        return idx


# --- sherlock.models.components.kcenter ---


class KCenterGreedyExtended(KCenterGreedy):
    # Makes the epsilon changeable. Default was 0.9
    def __init__(
        self,
        embedding: torch.Tensor,
        sampling_ratio: float | int,
        eps: float = 0.1,
    ) -> None:
        self.embedding = embedding
        self.coreset_size = (
            int(embedding.shape[0] * sampling_ratio)
            if isinstance(sampling_ratio, float)
            else sampling_ratio
        )

        self.features: torch.Tensor
        self.min_distances: torch.Tensor = None
        self.n_observations = self.embedding.shape[0]
        self.model = SparseRandomProjection(eps=eps)

    def update_distances(
        self, cluster_centers: list[int], chunk_size: int = 1024
    ) -> None:
        """Update min distances given cluster centers.

        Computes the distance from every feature to each cluster center in a
        single vectorized pass via ``torch.cdist`` and keeps the running minimum.
        This handles one *or many* centers at once, so initializing distances
        from a large set of already-selected indices no longer requires a Python
        loop with one full pass per center.

        Args:
            cluster_centers (list[int]): indices of cluster centers
            chunk_size (int): number of centers processed per ``cdist`` call.
                Bounds the intermediate ``(n_features, chunk_size)`` distance
                matrix so memory stays manageable when many centers are passed.
        """
        if not cluster_centers:
            return

        centers = self.features[cluster_centers]
        new_min: torch.Tensor | None = None
        for start in range(0, centers.shape[0], chunk_size):
            block = centers[start : start + chunk_size]
            # (n_features, block) -> (n_features, 1) row-wise minimum distance.
            block_min = torch.cdist(self.features, block, p=2).amin(dim=1, keepdim=True)
            new_min = (
                block_min if new_min is None else torch.minimum(new_min, block_min)
            )

        if self.min_distances is None:
            self.min_distances = new_min
        else:
            self.min_distances = torch.minimum(self.min_distances, new_min)

    def select_coreset_idxs(self, selected_idxs: list[int] | None = None) -> list[int]:
        """Greedily form a coreset to minimize the maximum distance of a cluster.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
          indices of samples selected to minimize distance to cluster centers
        """
        if selected_idxs is None:
            selected_idxs = []

        if self.embedding.ndim == 2:
            self.model.fit(self.embedding)
            if self.model.n_components > self.embedding.shape[1]:
                self.features = self.embedding
            else:
                self.features = self.model.transform(self.embedding)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.update_distances(cluster_centers=selected_idxs)

        selected_coreset_idxs: list[int] = []
        idx = int(
            torch.randint(
                low=len(selected_idxs), high=self.n_observations, size=(1,)
            ).item()
        )
        if selected_idxs:
            # Initialize min distances against all already-selected centers in a
            # single batched pass instead of one full pass per center.
            self.update_distances(cluster_centers=selected_idxs)
            self.min_distances[selected_idxs] = 0
            # Seed the greedy loop with an already-selected center so its first
            # ``update_distances`` call is a no-op rather than injecting a
            # non-selected point as a center.
            idx = selected_idxs[-1]

        bar = tqdm(
            range(self.coreset_size - len(selected_idxs)),
            desc="Selecting Coreset Indices.",
            leave=False,
        )
        for _ in bar:
            self.update_distances(cluster_centers=[idx])
            idx = self.get_new_idx().item()
            if idx in selected_idxs:
                msg = "New indices should not be in selected indices."
                raise ValueError(msg)
            self.min_distances[idx] = 0
            selected_coreset_idxs.append(idx)

        return selected_idxs + selected_coreset_idxs

    def sample_coreset(self, idxs) -> torch.Tensor:
        """Select coreset from the embedding.

        Returns:
            torch.Tensor: Selected coreset.
        """
        idxs = self.select_coreset_idxs(idxs)
        return self.embedding[idxs]


# --- sherlock.models.hacked_patchcore.HackedPatchcoreModel.get_embedding_subsample ---


def get_embedding_subsample(
    embedding: torch.Tensor,
    sampling_ratio: float,
    selected_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Subsample embedding based on coreset sampling and store to memory.

    Args:
        embedding (np.ndarray): Embedding tensor from the CNN
        sampling_ratio (float): Coreset sampling ratio
    Returns:
        torch.tensor: subsampled coreset of embeddings
    """

    sampler = KCenterGreedyExtended(
        embedding=embedding.float(), sampling_ratio=sampling_ratio, eps=0.9
    )
    coreset = sampler.sample_coreset(
        selected_indices if selected_indices is not None else []
    )
    return coreset
