"""Convert the raw VisA download into the MVTec layout that `adcore.mvtec.VISA` reads.

VisA ships its images under ``<category>/Data/{Images,Masks}/{Normal,Anomaly}/`` and its
splits as CSVs. The 1cls split (``split_csv/1cls.csv``) becomes

    <dst>/<category>/train/good/<name>.JPG
    <dst>/<category>/test/{good,bad}/<name>.JPG
    <dst>/<category>/ground_truth/bad/<stem>.png     (0/255 mask)

which is what VisA's own ``prepare_data.py`` produces. Images are copied byte for byte
(or symlinked with ``--symlink``); VisA's 0/1 masks are rewritten as 0/255.

    adcore-convert-visa /data/.../raw/visa /data/.../raw/visa/visa_pytorch
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def convert_visa(
    src: str | Path,
    dst: str | Path,
    split_csv: str | Path | None = None,
    categories: list[str] | None = None,
    symlink: bool = False,
) -> int:
    """Write the split in ``split_csv`` (default ``<src>/split_csv/1cls.csv``) under ``dst``.

    Args:
        src: The raw VisA root, holding one directory per category and ``split_csv/``.
        dst: Output root; its category directories must not exist yet.
        split_csv: A VisA split CSV with columns ``object,split,label,image,mask``.
        categories: Only convert these; None converts every category in the CSV.
        symlink: Symlink images instead of copying them. Masks are always written.

    Returns:
        The number of images written.
    """
    src, dst = Path(src), Path(dst)
    split_csv = (
        Path(split_csv) if split_csv is not None else src / "split_csv" / "1cls.csv"
    )
    with open(split_csv, newline="") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if not categories or row["object"] in categories
        ]
    if categories and (missing := set(categories) - {row["object"] for row in rows}):
        raise ValueError(f"not in {split_csv}: {sorted(missing)}")

    for category in sorted({row["object"] for row in rows}):
        if (dst / category).exists():
            raise FileExistsError(f"{dst / category} already exists")

    targets: set[Path] = set()
    for row in tqdm(rows, desc="VisA"):
        image = src / row["image"]
        defect_type = "good" if row["label"] == "normal" else "bad"
        target = dst / row["object"] / row["split"] / defect_type / image.name
        if target in targets:
            raise ValueError(f"two images map to {target}")
        targets.add(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if symlink:
            os.symlink(image.resolve(), target)
        else:
            shutil.copyfile(image, target)

        if defect_type == "bad":
            mask = np.array(Image.open(src / row["mask"]).convert("L"))
            mask_target = (
                dst / row["object"] / "ground_truth" / "bad" / f"{image.stem}.png"
            )
            mask_target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(((mask > 0) * 255).astype(np.uint8)).save(mask_target)
    return len(targets)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("src", type=Path, help="raw VisA root")
    parser.add_argument("dst", type=Path, help="output root, e.g. <src>/visa_pytorch")
    parser.add_argument(
        "--split-csv", type=Path, help="default: <src>/split_csv/1cls.csv"
    )
    parser.add_argument("--categories", nargs="+", help="default: all")
    parser.add_argument(
        "--symlink", action="store_true", help="symlink images, don't copy"
    )
    args = parser.parse_args(argv)
    n = convert_visa(args.src, args.dst, args.split_csv, args.categories, args.symlink)
    print(f"wrote {n} images to {args.dst}")


if __name__ == "__main__":
    main()
