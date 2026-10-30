"""Convert the raw Real-IAD download into the MVTec layout that `adcore.mvtec.REALIAD` reads.

Real-IAD ships its images and masks under ``<category>/{OK,NG/<defect_type>}/<sample>/`` and
its splits as one JSON per category. The single-view split (``realiad_jsons_sv/``) becomes

    <dst>/<category>/train/good/<name>.jpg
    <dst>/<category>/test/{good,<defect_type>}/<name>.jpg
    <dst>/<category>/ground_truth/<defect_type>/<stem>.png     (0/255 mask)

with the per-sample directories flattened away. Images are copied byte for byte (or
symlinked with ``--symlink``); Real-IAD's anti-aliased masks are rewritten as 0/255, any
nonzero pixel counting as anomalous, as `adcore.mvtec` reads masks anyway.

    adcore-convert-realiad /data/.../raw/realiad_1024 /data/.../raw/realiad_1024/realiad_sv
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def convert_realiad(
    src: str | Path,
    dst: str | Path,
    json_dir: str | Path | None = None,
    categories: list[str] | None = None,
    symlink: bool = False,
) -> int:
    """Write the split in ``json_dir`` (default ``<src>/realiad_jsons_sv``) under ``dst``.

    Args:
        src: The raw Real-IAD root, holding one directory per category and the JSON dirs.
        dst: Output root; its category directories must not exist yet.
        json_dir: A directory of Real-IAD split JSONs, one ``<category>.json`` each, with
            ``meta.prefix``, ``meta.normal_class`` and ``train``/``test`` lists of
            ``{anomaly_class, image_path, mask_path}``.
        categories: Only convert these; None converts every category in ``json_dir``.
        symlink: Symlink images instead of copying them. Masks are always written.

    Returns:
        The number of images written.
    """
    src, dst = Path(src), Path(dst)
    json_dir = Path(json_dir) if json_dir is not None else src / "realiad_jsons_sv"
    available = sorted(path.stem for path in json_dir.glob("*.json"))
    if categories and (missing := set(categories) - set(available)):
        raise ValueError(f"not in {json_dir}: {sorted(missing)}")
    categories = sorted(categories) if categories else available

    for category in categories:
        if (dst / category).exists():
            raise FileExistsError(f"{dst / category} already exists")

    rows = []
    for category in categories:
        with open(json_dir / f"{category}.json") as f:
            split = json.load(f)
        prefix = src / split["meta"]["prefix"]
        normal_class = split["meta"]["normal_class"]
        for split_name in ("train", "test"):
            for row in split[split_name]:
                defect_type = (
                    "good" if row["anomaly_class"] == normal_class else row["anomaly_class"]
                )
                mask = prefix / row["mask_path"] if row["mask_path"] else None
                rows.append(
                    (category, split_name, defect_type, prefix / row["image_path"], mask)
                )

    targets: set[Path] = set()
    for category, split_name, defect_type, image, mask in tqdm(rows, desc="Real-IAD"):
        target = dst / category / split_name / defect_type / image.name
        if target in targets:
            raise ValueError(f"two images map to {target}")
        targets.add(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if symlink:
            os.symlink(image.resolve(), target)
        else:
            shutil.copyfile(image, target)

        if defect_type != "good":
            if mask is None:
                raise ValueError(f"{image} is anomalous but has no mask")
            pixels = np.array(Image.open(mask).convert("L"))
            mask_target = (
                dst / category / "ground_truth" / defect_type / f"{image.stem}.png"
            )
            mask_target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(((pixels > 0) * 255).astype(np.uint8)).save(mask_target)
    return len(targets)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])  # type: ignore
    parser.add_argument("src", type=Path, help="raw Real-IAD root")
    parser.add_argument("dst", type=Path, help="output root, e.g. <src>/realiad_sv")
    parser.add_argument(
        "--json-dir", type=Path, help="default: <src>/realiad_jsons_sv"
    )
    parser.add_argument("--categories", nargs="+", help="default: all")
    parser.add_argument(
        "--symlink", action="store_true", help="symlink images, don't copy"
    )
    args = parser.parse_args(argv)
    n = convert_realiad(
        args.src, args.dst, args.json_dir, args.categories, args.symlink
    )
    print(f"wrote {n} images to {args.dst}")


if __name__ == "__main__":
    main()
