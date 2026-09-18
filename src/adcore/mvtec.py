"""MVTec AD as a plain ``torch.utils.data.Dataset`` yielding dicts, so a bare ``DataLoader`` works."""

import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


def default_transform(image_size: tuple[int, int] = (224, 224)) -> T.Compose:
    """Resize + ImageNet normalisation.

    Applied to image and mask together: ``Resize`` uses nearest on a ``Mask``, and
    ``ConvertImageDtype``/``Normalize`` leave its 0/1 values untouched.
    """
    return T.Compose(
        [
            T.Resize(image_size),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_records(root: Path, categories: list[str], split: str) -> list[dict]:
    """Every frame of ``split``, sorted by (category, defect_type, filename)."""
    records = []
    for category in categories:
        for defect_dir in sorted((root / category / split).iterdir()):
            defect_type = defect_dir.name
            for image_path in sorted(defect_dir.glob("*.png")):
                mask_path = ""
                if defect_type != "good":
                    mask_path = str(
                        root
                        / category
                        / "ground_truth"
                        / defect_type
                        / f"{image_path.stem}_mask.png"
                    )
                records.append(
                    {
                        "image_path": str(image_path),
                        "mask_path": mask_path,
                        "label": int(defect_type != "good"),
                        "category": category,
                        "defect_type": defect_type,
                    }
                )
    return records


class MVTecDataset(Dataset):
    """MVTec AD frames as dicts with keys ``image`` (float32 CHW), ``mask`` (float32 1HW, 0/1),
    ``label`` (0 good, 1 anomalous), ``category``, ``defect_type``, ``image_path`` and
    ``mask_path`` (``""`` for good frames).

    Args:
        root: Directory holding one subdirectory per category.
        category: One name, several, or None for all of them.
        split: ``"train"`` or ``"test"``.
        transform: A torchvision v2 transform taking ``(image, mask)``.
        image_size: Only used to build the default transform.
    """

    def __init__(
        self,
        root: str | Path,
        category: str | list[str] | None = None,
        split: str = "test",
        transform=None,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        root = Path(root)
        if category is None:
            categories = sorted(p.name for p in root.iterdir() if p.is_dir())
        elif isinstance(category, str):
            categories = [category]
        else:
            categories = list(category)
        self.transform = transform or default_transform(image_size)
        self.records = find_records(root, categories, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        image = np.array(Image.open(record["image_path"]).convert("RGB"))
        if record["mask_path"]:
            mask = (np.array(Image.open(record["mask_path"])) > 0).astype(np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # One call for the pair, so a random transform draws the same parameters for both.
        image, mask = self.transform(
            tv_tensors.Image(image.transpose(2, 0, 1)), tv_tensors.Mask(mask)
        )

        return {
            **record,
            "image": image.as_subclass(torch.Tensor),
            "mask": mask.as_subclass(torch.Tensor).unsqueeze(0),
        }


def split_by_defect_type(
    dataset: MVTecDataset,
    train_fraction: float,
    exclude: str | list[str] | tuple[str] = (),
    seed: int = 0,
) -> tuple[Subset, Subset]:
    """Split into (train, test) by taking ``train_fraction`` of the frames of every
    (category, defect_type) group for train and the rest for test.

    Defect types in ``exclude`` (e.g. ``"good"``) go to test entirely. Reads only
    ``dataset.records``, so no image is decoded.
    """
    if isinstance(exclude, str):
        exclude = [exclude]

    groups: dict[tuple[str, str], list[int]] = {}
    for idx, record in enumerate(dataset.records):
        groups.setdefault((record["category"], record["defect_type"]), []).append(idx)

    rng = np.random.default_rng(seed)
    train_indices, test_indices = [], []
    for (_, defect_type), indices in groups.items():
        if defect_type in exclude:
            test_indices += indices
            continue
        indices = rng.permutation(indices).tolist()
        n_train = round(train_fraction * len(indices))
        train_indices += indices[:n_train]
        test_indices += indices[n_train:]

    return Subset(dataset, sorted(train_indices)), Subset(dataset, sorted(test_indices))


def few_shot_subset(dataset: MVTecDataset, shots: int | None, seed: int = 0) -> Subset:
    """``shots`` defect-free frames per category, drawn with ``seed``; None keeps them all.

    Each category draws from its own generator keyed on ``(seed, category)``, so a
    category's sample does not depend on which other categories the dataset holds. For a
    fixed seed the draws are nested — the 2-shot set contains the 1-shot set — so shot
    counts differ only in how many images they see, not in which.
    """
    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(dataset.records):
        if record["label"] == 0:
            groups.setdefault(record["category"], []).append(idx)

    indices = []
    for category, group in groups.items():
        if shots is None:
            indices += group
            continue
        if shots > len(group):
            raise ValueError(
                f"{shots}-shot needs {shots} good frames but {category} has {len(group)}"
            )
        rng = np.random.default_rng([seed, zlib.crc32(category.encode())])
        indices += rng.permutation(group)[:shots].tolist()

    return Subset(dataset, sorted(indices))
