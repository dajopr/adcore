"""MVTec AD as a ``LightningDataModule`` for few-shot and supervised training.

- ``train_dataloader``: ``shots`` defect-free train frames per category (all when None),
  plus — with ``anomalous_fraction`` — that fraction of every defect type's test frames,
  which are then removed from the test set. Batches carry ``label`` so a module can
  tell them apart.
- ``reference_dataloader``: the defect-free training frames only, unshuffled — what a
  trained module builds a memory bank from after its gradient epochs.
- ``test_dataloader``: the (remaining) test split.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from adcore.mvtec import (
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
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.category = category
        self.shots = shots
        self.seed = seed
        self.anomalous_fraction = anomalous_fraction
        self.transform = transform or default_transform(tuple(image_size))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._full_test = test_dataset
        self.reference: Dataset | None = None
        self.train: Dataset | None = None
        self.test: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.test is not None:
            return
        train = MVTecDataset(self.root, self.category, "train", self.transform)
        test = self._full_test or MVTecDataset(
            self.root, self.category, "test", self.transform
        )
        self.reference = few_shot_subset(train, self.shots, self.seed)
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
