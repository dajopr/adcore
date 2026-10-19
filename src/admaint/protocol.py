"""The data of one maintenance run: what the module is fitted on, the update steps, the
eval sets and how their labels are read.

Every builder takes datasets with ``records`` (an `adcore.MVTecDataset`, or a
`adcore.CachedDataset` of its test split) and returns a `Protocol` of subsets of them.

Two seeds draw the frames. The *fit seed* (``seed``) draws everything the fitted module
depends on, so runs that differ only in their *update seed* (``update_seed``) can share
one fit; the update seed draws the update frames, and with them which target frames are
left to evaluate.

- The good train frames are permuted with the fit seed as in `adcore.few_shot_subset`,
  so the initial n-shot set is adcore's. The front ``shots`` are the initial set and the
  last ``n_calibration`` frames calibrate the detection threshold and are never trained
  on. The frames in between are permuted again with the update seed and feed nominal
  update steps from the front.
- A target defect type's test frames are permuted with the fit seed; contaminants come
  from the back, out of a reserve sized by the grid's largest ``n_contam``. The rest is
  permuted with the update seed: updates take the front and the eval keeps what follows
  a reserve sized by the largest ``n_update``.

For a given pair of seeds the eval set does not depend on ``n_update`` or ``n_contam``,
the draws are nested across them, and updates, contaminants and eval never overlap.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from adcore.evaluation import Predictions

CONTAMINATION = "contamination"
EXTENSION = "extension"
STEPWISE = "stepwise"
DRIFT = "drift"
SCENARIOS = (CONTAMINATION, EXTENSION, STEPWISE, DRIFT)

View = Callable[[Predictions], Predictions]


@dataclass
class Step:
    """One update: ``op`` ("add" or "remove") over every frame of ``dataset``."""

    name: str
    op: str
    dataset: Dataset
    use_masks: bool = True

    def __post_init__(self) -> None:
        if self.op not in ("add", "remove"):
            raise ValueError(f"op must be 'add' or 'remove', got {self.op!r}")


@dataclass
class Protocol:
    """Everything a maintenance run reads.

    ``evals`` and ``calibration`` share their keys (``"test"``, or ``"source"`` /
    ``"target"`` under drift); eval sets may share a calibration set. ``views`` re-read the labels of an eval set's predictions:
    ``"original"`` always, ``"extended"`` when the target becomes nominal.
    """

    initial: Dataset
    steps: list[Step]
    evals: dict[str, Dataset]
    calibration: dict[str, Dataset]
    views: dict[str, View]
    target: str = ""
    sizes: dict[str, int] = field(default_factory=dict)


# --- datasets ------------------------------------------------------------------------


def _records(dataset: Dataset) -> list[dict]:
    if isinstance(dataset, Subset):
        records = _records(dataset.dataset)
        return [records[i] for i in dataset.indices]
    return dataset.records


class Relabeled(Dataset):
    """``dataset`` with every item's ``label`` replaced and, optionally, its mask emptied:
    contaminants enter training looking like good frames."""

    def __init__(self, dataset: Dataset, label: int = 0, zero_mask: bool = True) -> None:
        self.dataset = dataset
        self.label = label
        self.zero_mask = zero_mask
        self.records = [{**r, "label": label} for r in _records(dataset)]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = {**self.dataset[idx], "label": self.label}
        if self.zero_mask:
            item["mask"] = torch.zeros_like(item["mask"])
        return item


class Shifted(Dataset):
    """Every frame of an `adcore.MVTecDataset` (or a `Subset` of one) under a fixed domain
    shift.

    ``shift`` is a torchvision v2 transform over ``(image, mask)`` that runs on the
    full-resolution uint8 frame before the dataset's own transform, like
    `adcore.AugmentedSupport`'s augmentation. Geometric shifts move the mask with the
    image; photometric ones leave it as it is. A shift should be deterministic (e.g.
    ``ColorJitter(brightness=[0.5, 0.5])``): it is one domain for every run. A random one
    draws from a torch RNG seeded with ``seed`` and the frame's filename, so a frame always
    gets the same draw, and the global RNG is left untouched.
    """

    def __init__(self, dataset: Dataset, shift: Callable, seed: int = 0) -> None:
        if isinstance(dataset, Subset):
            self.base, self.indices = dataset.dataset, list(dataset.indices)
        else:
            self.base, self.indices = dataset, list(range(len(dataset)))
        self.shift = shift
        self.seed = seed
        self.records = [self.base.records[i] for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        image, mask = self.base.load(idx)
        with torch.random.fork_rng(devices=[]):
            name = Path(self.base.records[idx]["image_path"]).name
            torch.manual_seed(self.seed * 1_000_003 + zlib.crc32(name.encode()))
            image, mask = self.shift(image, mask)
        return self.base.item(idx, image, mask)


# --- draws ---------------------------------------------------------------------------


def _crc(text: str) -> int:
    return zlib.crc32(text.encode())


def nominal_order(dataset: Dataset, category: str, seed: int) -> list[int]:
    """The good frames of ``category`` in ``dataset``, permuted as by
    `adcore.few_shot_subset`: its n-shot set is the first n of this order."""
    group = [
        i
        for i, r in enumerate(_records(dataset))
        if r["category"] == category and r["label"] == 0
    ]
    return np.random.default_rng([seed, _crc(category)]).permutation(group).tolist()


def update_rng(seed: int, update_seed: int, *keys: str) -> np.random.Generator:
    """The generator of the update draws of one (fit seed, update seed, category, ...)."""
    return np.random.default_rng([seed, update_seed, *map(_crc, keys)])


def nominal_split(
    order: Sequence[int],
    shots: int | None,
    sizes: Sequence[int],
    n_calibration: int,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], list[list[int]], list[int]]:
    """(initial, one slice per entry of ``sizes``, calibration) from a `nominal_order`.

    ``sizes`` are cumulative frame counts (initial frames included) after each slice. The
    slices come from the frames between the initial and the calibration set, in the order
    ``rng`` permutes them to (as they are without one). With ``shots`` None the initial
    set is every frame not set aside for calibration (and ``sizes`` must be empty).
    """
    available = len(order) - n_calibration
    if n_calibration < 1 or available < 1:
        raise ValueError(f"{len(order)} good frames leave none for n_calibration={n_calibration}")
    calibration = list(order[available:])
    if shots is None:
        if sizes:
            raise ValueError("nominal update steps need a number of shots, not every frame")
        return sorted(order[:available]), [], calibration
    bounds = [shots, *sizes]
    if any(b <= a for a, b in zip(bounds, bounds[1:])):
        raise ValueError(f"step sizes {list(sizes)} must grow beyond shots={shots}")
    if bounds[-1] > available:
        raise ValueError(
            f"shots={shots} and steps {list(sizes)} need {bounds[-1]} good frames, but only "
            f"{available} are left after n_calibration={n_calibration}"
        )
    pool = list(order[shots:available])
    if rng is not None:
        pool = rng.permutation(pool).tolist()
    slices = [sorted(pool[a - shots : b - shots]) for a, b in zip(bounds, bounds[1:])]
    return sorted(order[:shots]), slices, calibration


def target_order(dataset: Dataset, category: str, target: str, seed: int) -> list[int]:
    """The test frames of one defect type, permuted with ``(seed, category, target)``."""
    group = [
        i
        for i, r in enumerate(_records(dataset))
        if r["category"] == category and r["defect_type"] == target
    ]
    rng = np.random.default_rng([seed, _crc(category), _crc(target)])
    return rng.permutation(group).tolist()


def target_split(
    order: Sequence[int],
    n_update: int,
    n_contam: int,
    max_update: int,
    max_contam: int,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """(update, contaminants, eval) of a `target_order`.

    Contaminants are the back ``n_contam`` frames. The frames before the grid's
    contamination reserve are permuted with ``rng`` (kept in order without one); updates
    are the front ``n_update`` of them and the eval what follows the update reserve.
    """
    if n_update > max_update or n_contam > max_contam:
        raise ValueError("n_update / n_contam exceed the grid maxima")
    n = len(order)
    if max_update + max_contam >= n:
        raise ValueError(
            f"{n} frames leave none to evaluate after {max_update} updates and "
            f"{max_contam} contaminants"
        )
    contaminants = sorted(order[n - n_contam :]) if n_contam else []
    pool = list(order[: n - max_contam])
    if rng is not None:
        pool = rng.permutation(pool).tolist()
    return sorted(pool[:n_update]), contaminants, sorted(pool[max_update:])


def n_target_eval(n_frames: int, max_update: int, max_contam: int) -> int:
    return max(n_frames - max_update - max_contam, 0)


# --- views ---------------------------------------------------------------------------


def original(predictions: Predictions) -> Predictions:
    return predictions


def as_nominal(target: str) -> View:
    """A view in which ``target`` frames are good: label 0, empty mask, defect type
    ``"good"``."""

    def view(predictions: Predictions) -> Predictions:
        hit = predictions.defect_types == target
        masks = predictions.masks.copy()
        masks[hit] = 0
        return replace(
            predictions,
            labels=np.where(hit, 0, predictions.labels),
            masks=masks,
            defect_types=np.where(hit, "good", predictions.defect_types),
        )

    return view


# --- scenarios -----------------------------------------------------------------------


def _without(dataset: Dataset, category: str, excluded: Sequence[int] = ()) -> list[int]:
    """Indices of ``category``'s frames, leaving out ``excluded``."""
    excluded = set(excluded)
    return [
        i
        for i, r in enumerate(_records(dataset))
        if r["category"] == category and i not in excluded
    ]


def _defect_protocol(
    train: Dataset,
    calibration_source: Dataset,
    test: Dataset,
    category: str,
    target: str,
    seed: int,
    *,
    update_seed: int,
    shots: int | None,
    n_update: int,
    n_contam: int,
    max_update: int,
    max_contam: int,
    n_calibration: int,
    op: str,
    views: dict[str, View],
) -> Protocol:
    if n_update < 1:
        raise ValueError(f"n_update must be at least 1, got {n_update}")
    initial, _, calibration = nominal_split(
        nominal_order(train, category, seed), shots, [], n_calibration
    )
    order = target_order(test, category, target, seed)
    update, contaminants, evaluated = target_split(
        order, n_update, n_contam, max_update, max_contam,
        update_rng(seed, update_seed, category, target),
    )
    # The target's other frames (updates, contaminants, the unused rest of the grid's
    # reserve) are never evaluated.
    held_out = set(order) - set(evaluated)
    initial_set: Dataset = Subset(train, initial)
    if contaminants:
        initial_set = ConcatDataset([initial_set, Relabeled(Subset(test, contaminants))])
    return Protocol(
        initial=initial_set,
        steps=[Step("update", op, Subset(test, update), use_masks=True)],
        evals={"test": Subset(test, _without(test, category, held_out))},
        calibration={"test": Subset(calibration_source, calibration)},
        views=views,
        target=target,
        sizes={
            "n_initial": len(initial_set),
            "n_eval_target": len(evaluated),
            "n_calibration": len(calibration),
        },
    )


def contamination(
    train: Dataset,
    calibration_source: Dataset,
    test: Dataset,
    category: str,
    target: str,
    seed: int,
    *,
    update_seed: int = 0,
    shots: int | None,
    n_contam: int,
    n_update: int,
    max_contam: int,
    max_update: int,
    n_calibration: int,
) -> Protocol:
    """Scenario 1: ``n_contam`` frames of ``target`` hide among the good training frames;
    ``n_update`` other frames of it, with masks, are removed from the nominal state.
    ``n_contam=0`` is the clean control."""
    return _defect_protocol(
        train,
        calibration_source,
        test,
        category,
        target,
        seed,
        update_seed=update_seed,
        shots=shots,
        n_update=n_update,
        n_contam=n_contam,
        max_update=max_update,
        max_contam=max_contam,
        n_calibration=n_calibration,
        op="remove",
        views={"original": original},
    )


def extension(
    train: Dataset,
    calibration_source: Dataset,
    test: Dataset,
    category: str,
    target: str,
    seed: int,
    *,
    update_seed: int = 0,
    shots: int | None,
    n_update: int,
    max_update: int,
    n_calibration: int,
) -> Protocol:
    """Scenario 2: clean training; ``n_update`` frames of ``target``, with masks, are added
    to the nominal state. The ``extended`` view scores ``target`` as good."""
    return _defect_protocol(
        train,
        calibration_source,
        test,
        category,
        target,
        seed,
        update_seed=update_seed,
        shots=shots,
        n_update=n_update,
        n_contam=0,
        max_update=max_update,
        max_contam=0,
        n_calibration=n_calibration,
        op="add",
        views={"original": original, "extended": as_nominal(target)},
    )


def stepwise(
    train: Dataset,
    calibration_source: Dataset,
    test: Dataset,
    category: str,
    seed: int,
    *,
    update_seed: int = 0,
    shots: int,
    steps: Sequence[int],
    n_calibration: int,
) -> Protocol:
    """Scenario 3: ``shots`` good frames, then more good frames added whole until there
    are ``steps[i]`` after step i."""
    if not steps:
        raise ValueError("stepwise needs at least one step")
    initial, slices, calibration = nominal_split(
        nominal_order(train, category, seed), shots, steps, n_calibration,
        update_rng(seed, update_seed, category),
    )
    return Protocol(
        initial=Subset(train, initial),
        steps=[
            Step(f"nominal{size}", "add", Subset(train, indices), use_masks=False)
            for size, indices in zip(steps, slices)
        ],
        evals={"test": Subset(test, _without(test, category))},
        calibration={"test": Subset(calibration_source, calibration)},
        views={"original": original},
        sizes={"n_initial": len(initial), "n_calibration": len(calibration)},
    )


def drift(
    train: Dataset,
    calibration_source: Dataset,
    test: Dataset,
    shifted_test: Dataset,
    category: str,
    seed: int,
    *,
    update_seed: int = 0,
    shift: Callable,
    shots: int,
    n_update: int,
    n_calibration: int,
    retire: bool = False,
) -> Protocol:
    """Scenario 4: fitted on ``shots`` good frames; ``n_update`` further good frames under
    ``shift`` are added whole. With ``retire`` a second step removes the initial frames.
    Evaluated on the test split as it is (``source``) and shifted (``target``), both
    against a threshold calibrated on unshifted frames."""
    if n_update < 1:
        raise ValueError(f"n_update must be at least 1, got {n_update}")
    initial, (update,), calibration = nominal_split(
        nominal_order(train, category, seed), shots, [shots + n_update], n_calibration,
        update_rng(seed, update_seed, category),
    )
    steps = [Step("update", "add", Shifted(Subset(train, update), shift), use_masks=False)]
    if retire:
        steps.append(Step("retire", "remove", Subset(train, initial), use_masks=False))
    return Protocol(
        initial=Subset(train, initial),
        steps=steps,
        evals={
            "source": Subset(test, _without(test, category)),
            "target": Subset(shifted_test, _without(shifted_test, category)),
        },
        # The deployed threshold comes from the nominal frames the operator validated on,
        # so the target set's nominal_fpr shows the false alarms the shift causes.
        calibration=dict.fromkeys(("source", "target"), Subset(calibration_source, calibration)),
        views={"original": original},
        sizes={"n_initial": len(initial), "n_calibration": len(calibration)},
    )


def threshold_metrics(
    predictions: Predictions, threshold: float, target: str = ""
) -> dict[str, Any]:
    """Rates of frames scored above ``threshold``, by their original labels:
    ``nominal_fpr`` (good frames), ``defect_detect_rate`` (anomalous frames other than
    ``target``) and, with a ``target``, ``target_detect_rate``."""
    above = predictions.image_scores > threshold
    good = predictions.labels == 0
    is_target = predictions.defect_types == target if target else np.zeros_like(good)
    other = ~good & ~is_target

    def rate(selected: np.ndarray) -> float:
        return float(above[selected].mean()) if selected.any() else float("nan")

    metrics = {
        "threshold": float(threshold),
        "nominal_fpr": rate(good),
        "defect_detect_rate": rate(other),
    }
    if target:
        metrics["target_detect_rate"] = rate(is_target)
    return metrics
