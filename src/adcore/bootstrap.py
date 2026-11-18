"""Paired bootstrap confidence intervals of image AUROC over the test images.

    table = paired_bootstrap(
        {"ours": "runs/mvtec/ours", "baseline": "runs/mvtec/baseline"}, reference="baseline"
    )

Reads the per-image ``scores/<run>.csv`` files `FewShotExperiment` writes into each output
directory. Every replicate resamples the test images of each category with replacement,
positives and negatives separately so their counts stay fixed, and takes the image AUROC
per category, then the mean over categories and then over seeds, as `summarize` does. All
variants, seeds and shot counts share each replicate's resample, so a difference to the
reference is a paired bootstrap. The interval covers the test-set sampling only; the
spread over seeds is separate.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

_RUN = re.compile(r"(?P<category>.+)_(?P<shots>\d+|full)shot_seed(?P<seed>\d+)\.csv")


def _load(dirs: dict[str, Path]) -> tuple[dict, dict]:
    """Scores by (variant, category, shots, seed) in a common image order, and the labels
    of every category."""
    scores, labels = {}, {}
    for name, directory in dirs.items():
        for path in sorted((Path(directory) / "scores").glob("*.csv")):
            match = _RUN.fullmatch(path.name)
            if match is None:
                continue
            category = match["category"]
            df = pd.read_csv(path)
            # Images from the category directory down, so runs made under another
            # dataset root pair with the rest
            relative = df["image_path"].str.extract(rf"((?:^|/){re.escape(category)}/test/.*)$")[0]
            df["image_path"] = relative.fillna(df["image_path"]).str.lstrip("/")
            df = df.sort_values("image_path")
            if category not in labels:
                labels[category] = (df["image_path"].to_numpy(), df["label"].to_numpy())
            elif not np.array_equal(labels[category][0], df["image_path"].to_numpy()):
                raise ValueError(f"{path} tests other images than the other {category} runs")
            shots = None if match["shots"] == "full" else int(match["shots"])
            scores[(name, category, shots, int(match["seed"]))] = df["image_score"].to_numpy(
                np.float64
            )
    return scores, {c: y for c, (_, y) in labels.items()}


def _auroc_rows(s: np.ndarray, n_pos: int) -> np.ndarray:
    """AUROC of every row of ``s`` whose first ``n_pos`` columns are the positives, ties
    at half weight (Mann-Whitney U)."""
    ranks = rankdata(s, axis=1)
    n_neg = s.shape[1] - n_pos
    return (ranks[:, :n_pos].sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_runs(
    dirs: dict[str, str | Path], n_boot: int = 2000, seed: int = 0
) -> dict[tuple[str, int | None, int], tuple[float, np.ndarray]]:
    """Per (variant, shots, seed) run that has every category: the image AUROC averaged
    over categories, and its ``n_boot`` bootstrap replicates from one stratified resample
    of each category's test images shared by every run."""
    scores, labels = _load({name: Path(d) for name, d in dirs.items()})
    categories = sorted(labels)
    rng = np.random.default_rng(seed)
    # One stratified resample per category, positives first, shared by every run
    resamples = {}
    for category, y in labels.items():
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        resamples[category] = (
            np.concatenate(
                [rng.choice(pos, (n_boot, len(pos))), rng.choice(neg, (n_boot, len(neg)))],
                axis=1,
            ),
            len(pos),
        )
    runs = {}
    for variant, shots, run_seed in sorted(
        {(k[0], k[2], k[3]) for k in scores}, key=lambda k: (k[0], k[1] is None, k[1] or 0, k[2])
    ):
        if not all((variant, c, shots, run_seed) in scores for c in categories):
            continue
        point, boot = [], []
        for category in categories:
            s = scores[(variant, category, shots, run_seed)]
            idx, n_pos = resamples[category]
            y = labels[category]
            order = np.concatenate([np.flatnonzero(y == 1), np.flatnonzero(y == 0)])
            point.append(_auroc_rows(s[order][None], n_pos)[0])
            boot.append(_auroc_rows(s[idx], n_pos))
        runs[(variant, shots, run_seed)] = (float(np.mean(point)), np.mean(boot, axis=0))
    return runs


def paired_bootstrap(
    dirs: dict[str, str | Path] | list[str | Path],
    reference: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Image AUROC (in %) with a 95% bootstrap interval per variant and shot count, and
    over the shot counts ("mean"); with ``reference``, also the paired difference to it
    (``diff``, ``diff_low``, ``diff_high``) and its two-sided bootstrap p-value.

    ``dirs`` maps variant names to output directories; a list names each by its
    directory. A (variant, shots) uses the seeds that have every category, a difference
    the seeds both variants have, and "mean" the seeds every shot count has.
    """
    if not isinstance(dirs, dict):
        dirs = {Path(d).name: d for d in dirs}
    if reference is not None and reference not in dirs:
        raise ValueError(f"reference {reference!r} is not one of {list(dirs)}")
    runs = bootstrap_runs(dirs, n_boot, seed)
    shot_counts = sorted({k[1] for k in runs}, key=lambda s: (s is None, s or 0))

    def seeds_of(variant: str, shots: list) -> set[int]:
        found = [{sd for v, s, sd in runs if v == variant and s == sh} for sh in shots]
        return set.intersection(*found) if found else set()

    def estimate(variant: str, shots: list, seeds: set[int]) -> tuple[float, np.ndarray]:
        keys = [(variant, sh, sd) for sh in shots for sd in sorted(seeds)]
        return (
            float(np.mean([runs[k][0] for k in keys])),
            np.mean([runs[k][1] for k in keys], axis=0),
        )

    rows = []
    for variant in dirs:
        groups = [("full" if s is None else str(s), [s]) for s in shot_counts]
        for label, shots in [*groups, ("mean", shot_counts)]:
            seeds = seeds_of(variant, shots)
            if not seeds:
                continue
            point, boot = estimate(variant, shots, seeds)
            row = {
                "variant": variant,
                "shots": label,
                "seeds": ",".join(map(str, sorted(seeds))),
                "image_auroc": 100 * point,
                "ci_low": 100 * np.percentile(boot, 2.5),
                "ci_high": 100 * np.percentile(boot, 97.5),
            }
            if reference is not None and variant != reference:
                common = seeds & seeds_of(reference, shots)
                if common:
                    p, b = estimate(variant, shots, common)
                    rp, rb = estimate(reference, shots, common)
                    diff = b - rb
                    row |= {
                        "diff_seeds": ",".join(map(str, sorted(common))),
                        "diff": 100 * (p - rp),
                        "diff_low": 100 * np.percentile(diff, 2.5),
                        "diff_high": 100 * np.percentile(diff, 97.5),
                        # Share of replicates on the other side of zero, two-sided
                        "p_boot": min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())),
                    }
            rows.append(row)
    return pd.DataFrame(rows)
