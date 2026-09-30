"""MVTec AD as a ``LightningDataModule`` for few-shot and supervised training.

- ``train_dataloader``: ``shots`` defect-free train frames per category (all when None),
  plus — with ``anomalous_fraction`` — that fraction of every defect type's test frames,
  which are then removed from the test set. Batches carry ``label`` so a module can
  tell them apart.
- ``reference_dataloader``: the defect-free training frames only, unshuffled — what a
  trained module builds a memory bank from after its gradient epochs. With
  ``support_aug`` each few-shot frame also comes as seeded augmented views (see
  `AugmentedSupport`), in both this loader and ``train_dataloader``.
- ``test_dataloader``: the (remaining) test split.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.transforms import v2 as T

from adcore.mvtec import (
    MVTEC,
    DatasetSpec,
    MVTecDataset,
    default_transform,
    few_shot_subset,
    split_by_defect_type,
)


def _as_list(batch: list) -> list:
    return batch


class CachedDataset(Dataset):
    """Every item of an `MVTecDataset` decoded once and held in memory.

    Keeps ``records`` so it can stand in for the dataset when splitting. Load it with
    workers; read it with ``num_workers=0``, since workers would copy the cache.
    """

    def __init__(self, dataset: MVTecDataset, num_workers: int = 0, batch_size: int = 32):
        self.records = dataset.records
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_as_list
        )
        self.items = [item for batch in loader for item in batch]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def default_rotation() -> T.RandomRotation:
    """SubspaceAD's support augmentation: a rotation in [0, 345] degrees, nearest
    interpolation, black corners, no expansion."""
    return T.RandomRotation(
        degrees=(0, 345), interpolation=T.InterpolationMode.NEAREST, expand=False, fill=0
    )


@dataclass(frozen=True)
class AugmentSpec:
    """Settings for `AugmentedSupport`; ``augment=None`` means `default_rotation`."""

    n_views: int = 30
    augment: Callable | None = None
    exclude_categories: tuple[str, ...] = ("transistor",)


class AugmentedSupport(Dataset):
    """Each frame of a support set as its original (view 0) plus ``n_views`` augmented
    views (1..n_views), each item tagged with its ``view``.

    ``augment`` runs on the full-resolution uint8 image and mask, before the dataset's
    transform resizes them, as in SubspaceAD. View ``v`` of an image draws from the torch
    RNG seeded with ``(seed, image, v)``, where the image is identified by its category,
    defect type and filename — so views are reproducible, usable as a cache key, and the
    same image gets the same views at every shot count. The global RNG is left untouched.

    Frames of ``exclude_categories`` yield view 0 only.

    Unlike SubspaceAD, which redraws its rotations between its mean and covariance
    passes, the views are drawn once; runs are therefore not bit-identical to theirs.

    Args:
        dataset: An `MVTecDataset` or a `Subset` of one (e.g. from `few_shot_subset`).
        n_views: Augmented views per frame, on top of the original.
        augment: A torchvision v2 transform over ``(image, mask)``; defaults to
            `default_rotation`.
        seed: Picks the views.
        exclude_categories: Categories left unaugmented.
    """

    def __init__(
        self,
        dataset: MVTecDataset | Subset,
        n_views: int = 30,
        augment: Callable | None = None,
        seed: int = 0,
        exclude_categories: tuple[str, ...] | list[str] = ("transistor",),
    ) -> None:
        if isinstance(dataset, Subset):
            self.base, base_indices = dataset.dataset, list(dataset.indices)
        else:
            self.base, base_indices = dataset, list(range(len(dataset)))
        self.n_views = n_views
        self.augment = augment or default_rotation()
        self.seed = seed
        self.exclude_categories = tuple(exclude_categories)
        self.index = [
            (idx, view)
            for idx in base_indices
            for view in range(
                1
                if self.base.records[idx]["category"] in self.exclude_categories
                else n_views + 1
            )
        ]
        self.records = [self.base.records[idx] for idx, _ in self.index]

    def __len__(self) -> int:
        return len(self.index)

    def _view_seed(self, idx: int, view: int) -> int:
        record = self.base.records[idx]
        name = f"{record['category']}/{record['defect_type']}/{Path(record['image_path']).name}"
        key = zlib.crc32(name.encode())
        return int(np.random.SeedSequence([self.seed, key, view]).generate_state(1)[0])

    def __getitem__(self, i: int) -> dict:
        idx, view = self.index[i]
        image, mask = self.base.load(idx)
        if view > 0:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self._view_seed(idx, view))
                image, mask = self.augment(image, mask)
        return {**self.base.item(idx, image, mask), "view": view}


class MVTecDataModule(L.LightningDataModule):
    """One or more MVTec AD categories, few-shot.

    Args:
        root: MVTec AD directory, one subdirectory per category.
        category: One name, several, or None for all of them.
        shots: Defect-free training frames per category; None keeps them all.
        seed: Picks the shots and the anomalous training frames. Draws are nested across
            shots — see `adcore.mvtec.few_shot_subset`.
        anomalous_fraction: Fraction of every (category, defect type) test group moved
            into training, for supervised or prompt-guided training. 0 keeps the test
            split intact.
        image_size: Used to build the default transform.
        transform: A torchvision v2 transform over ``(image, mask)``, applied to every split.
        train_transform: Replaces ``transform`` for the defect-free training frames
            (the anomalous training frames come from the test split and keep ``transform``).
        spec: The dataset's layout, e.g. `adcore.mvtec.MVTEC` or `adcore.mvtec.VISA`.
        support_aug: Expand each few-shot frame into augmented views, see
            `AugmentedSupport`. Ignored when ``shots`` is None. None: no augmentation.
        test_dataset: A prebuilt test split (e.g. a `CachedDataset`) to use instead of
            reading one; experiments pass the same cached split to every run.
        batch_size, num_workers: DataLoader settings. A `CachedDataset` is read with
            ``num_workers=0``.
    """

    def __init__(
        self,
        root: str | Path,
        category: str | list[str] | None = None,
        shots: int | None = None,
        seed: int = 0,
        anomalous_fraction: float = 0.0,
        image_size: tuple[int, int] = (224, 224),
        transform=None,
        test_dataset: MVTecDataset | CachedDataset | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform=None,
        spec: DatasetSpec = MVTEC,
        support_aug: AugmentSpec | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.category = category
        self.shots = shots
        self.seed = seed
        self.anomalous_fraction = anomalous_fraction
        self.transform = transform or default_transform(tuple(image_size))
        self.train_transform = train_transform or self.transform
        self.spec = spec
        self.support_aug = support_aug
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._full_test = test_dataset
        self.reference: Dataset | None = None
        self.train: Dataset | None = None
        self.test: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.test is not None:
            return
        train = MVTecDataset(
            self.root, self.category, "train", self.train_transform, spec=self.spec
        )
        test = self._full_test or MVTecDataset(
            self.root, self.category, "test", self.transform, spec=self.spec
        )
        self.reference = few_shot_subset(train, self.shots, self.seed)
        if self.support_aug is not None and self.shots is not None:
            aug = self.support_aug
            self.reference = AugmentedSupport(
                self.reference,
                n_views=aug.n_views,
                augment=aug.augment,
                seed=self.seed,
                exclude_categories=aug.exclude_categories,
            )
        if self.anomalous_fraction > 0:
            anomalous, self.test = split_by_defect_type(
                test, self.anomalous_fraction, exclude="good", seed=self.seed
            )
            self.train = ConcatDataset([self.reference, anomalous])
        else:
            self.train, self.test = self.reference, test

    def _loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        base = dataset.dataset if isinstance(dataset, Subset) else dataset
        cached = isinstance(base, CachedDataset)
        # Worker start-up dominates for a handful of shots.
        workers = 0 if cached or len(dataset) <= self.batch_size else self.num_workers
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=workers
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train, shuffle=True)

    def reference_dataloader(self) -> DataLoader:
        return self._loader(self.reference)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test)
