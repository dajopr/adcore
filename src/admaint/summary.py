"""A scorecard of adaptability: one row per model, a few numbers per scenario.

    results = load_results("runs")          # every admaint sweep under runs/
    card = scorecard(results)               # (dataset, model) x (scenario, metric, mean/std)
    print(format_scorecard(card))           # "mean ± std"

or ``admaint-summary runs``. Per scenario, the first metric is the headline and the rest
are side effects:

contamination
    ``recovery``: of the target AUROC lost to contamination (against the clean run of the
    same frames, ``n_contam=0`` at phase 0), the fraction the update wins back, as
    Σ(after − before) / Σ(clean − before) over runs that lost more than ``min_damage``.
    1 means fully recovered. NaN when contamination never cost that much.
    ``damage``: clean − contaminated target AUROC. ``control_other_auroc_delta`` and
    ``control_fpr_delta``: how the update on a clean model (``n_contam=0``) moves the
    AUROC of the other defect types and the nominal false-alarm rate (0 is best).
extension
    ``acceptance``: fraction of target frames below the threshold after the update (1 is
    best). ``extended_auroc_delta``: change of the AUROC that scores the target as good.
    ``other_auroc_delta``: change of the AUROC of the other defect types (0 is best).
stepwise
    ``auc``: image AUROC averaged over the frame budget (area under AUROC vs. good frames
    seen, divided by its width). ``final_auroc``: after the last step.
    ``update_ms_per_frame``: update time per added frame.
drift
    ``target_auroc_delta``: change of the shifted-domain AUROC from the update.
    ``target_fpr_after``: shifted good frames above the threshold after it (the threshold
    comes from unshifted frames). ``source_auroc_delta``: change on the unshifted domain.
    Drift's ``retire`` step is not part of the scorecard.

Every number is a mean over targets / shifts and grid settings within a category, then
over categories, per (fit seed, update seed); ``recovery`` pools those runs instead.
Update seeds are then averaged within each fit seed. ``mean`` and ``std`` are over fit
seeds; ``update_std`` is the spread over update seeds on the same fit, averaged over fit
seeds (NaN with one update seed). Within a dataset and scenario, models are compared on the runs they all
have; the others are dropped with a warning.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from adcore.experiment import results_frame
from admaint import protocol as P
from admaint.experiment import RUN_FIELDS

log = logging.getLogger(__name__)

GROUP = ["dataset", "model"]
KEY = [*GROUP, *RUN_FIELDS]


def load_results(root: str | Path = "runs", experiments: Sequence[str] | None = None) -> pd.DataFrame:
    """The rows of every admaint sweep under ``root`` (an ``output_root``), with
    ``experiment`` (e.g. ``mvtec-contamination``), ``model`` (``patchcore-wrn50``, or
    ``patchcore-wrn50/<variant>``) and ``dataset`` (the spec, e.g. ``mvtec``) columns.

    Few-shot sweeps of adcore in the same tree are skipped.
    """
    root = Path(root)
    frames = []
    for path in sorted(root.rglob("results.jsonl")):
        directory = path.parent
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        if "scenario" not in config:
            continue
        parts = directory.relative_to(root).parts
        if len(parts) >= 2:
            experiment, model = parts[0], "/".join(parts[1:])
        elif parts:
            experiment, model = root.name, parts[0]
        else:
            experiment, model = directory.parent.name, directory.name
        if experiments is not None and experiment not in experiments:
            continue
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if rows and "update_seed" not in rows[0]:
            log.warning("skipping %s: made before update seeds, with other draws", directory)
            continue
        frames.append(
            results_frame(rows).assign(
                experiment=experiment, model=model, dataset=config.get("spec", "")
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --- per-run values --------------------------------------------------------------------


def _rows(results: pd.DataFrame, scenario: str, eval_set: str, view: str = "original") -> pd.DataFrame:
    return results[
        (results["scenario"] == scenario)
        & (results["eval_set"] == eval_set)
        & (results["view"] == view)
    ]


def _phases(rows: pd.DataFrame, column: str, after: int | None = None) -> pd.DataFrame:
    """Per run, ``column`` at phase 0 and at phase ``after`` (default: the run's last),
    averaged over the run's rows of that phase."""
    if after is None:
        last = rows.groupby(KEY, dropna=False)["phase"].transform("max")
    else:
        last = after
    before = rows[rows["phase"] == 0].groupby(KEY, dropna=False)[column].mean()
    after_values = rows[rows["phase"] == last].groupby(KEY, dropna=False)[column].mean()
    return pd.DataFrame({"before": before, "after": after_values})


def _target(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["defect_type"] == rows["target"]]


def _others(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[(rows["defect_type"] != "all") & (rows["defect_type"] != rows["target"])]


def _overall(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["defect_type"] == "all"]


def contamination_runs(results: pd.DataFrame) -> pd.DataFrame:
    """Per contamination run: target AUROC ``before`` / ``after`` the update, ``clean``
    (the ``n_contam=0`` run of the same frames at phase 0), ``damage`` and ``gain``; and for
    the clean control runs the side effects of the update."""
    rows = _rows(results, P.CONTAMINATION, "test")
    target = _phases(_target(rows), "image_auroc").reset_index()
    # The clean run with the same seeds evaluates the same frames.
    same_frames = [*GROUP, "category", "target", "shift", "shots", "seed", "update_seed"]
    clean = (
        target[target["n_contam"] == 0]
        .groupby(same_frames, dropna=False)["before"]
        .first()
        .rename("clean")
        .reset_index()
    )
    runs = target.merge(clean, on=same_frames, how="left").set_index(KEY)
    runs["damage"] = runs["clean"] - runs["before"]
    runs["gain"] = runs["after"] - runs["before"]
    contaminated = runs.index.get_level_values("n_contam") > 0
    runs.loc[~contaminated, ["damage", "gain"]] = np.nan

    for name, phases in (
        ("control_other_auroc_delta", _phases(_others(rows), "image_auroc")),
        ("control_fpr_delta", _phases(_overall(rows), "nominal_fpr")),
    ):
        delta = (phases["after"] - phases["before"]).reindex(runs.index)
        runs[name] = delta.where(~contaminated)
    return runs


def extension_runs(results: pd.DataFrame) -> pd.DataFrame:
    rows = _rows(results, P.EXTENSION, "test")
    detected = _phases(_overall(rows), "target_detect_rate")
    extended = _phases(_overall(_rows(results, P.EXTENSION, "test", "extended")), "image_auroc")
    other = _phases(_others(rows), "image_auroc")
    return pd.DataFrame(
        {
            "acceptance": 1 - detected["after"],
            "extended_auroc_delta": extended["after"] - extended["before"],
            "other_auroc_delta": other["after"] - other["before"],
        }
    )


def stepwise_runs(results: pd.DataFrame) -> pd.DataFrame:
    rows = _overall(_rows(results, P.STEPWISE, "test")).sort_values("phase")

    def summarize(run: pd.DataFrame) -> pd.Series:
        frames = run["n_initial"].to_numpy() + run["n_step"].cumsum().to_numpy()
        auroc = run["image_auroc"].to_numpy()
        width = frames[-1] - frames[0]
        added = run["n_step"].sum()
        return pd.Series(
            {
                "auc": np.trapezoid(auroc, frames) / width if width else auroc[0],
                "final_auroc": auroc[-1],
                "update_ms_per_frame": 1000 * run["update_seconds"].sum() / added
                if added
                else np.nan,
            }
        )

    if rows.empty:
        return pd.DataFrame(columns=["auc", "final_auroc", "update_ms_per_frame"])
    return rows.groupby(KEY, dropna=False).apply(summarize)


def drift_runs(results: pd.DataFrame) -> pd.DataFrame:
    rows = results[(results["scenario"] == P.DRIFT) & (results["view"] == "original")]
    rows = _overall(rows)
    # Phase 1 is the update; a later `retire` step is left out.
    target = rows[rows["eval_set"] == "target"]
    source = rows[rows["eval_set"] == "source"]
    target_auroc = _phases(target, "image_auroc", after=1)
    target_fpr = _phases(target, "nominal_fpr", after=1)
    source_auroc = _phases(source, "image_auroc", after=1)
    return pd.DataFrame(
        {
            "target_auroc_delta": target_auroc["after"] - target_auroc["before"],
            "target_fpr_after": target_fpr["after"],
            "source_auroc_delta": source_auroc["after"] - source_auroc["before"],
        }
    )


RUN_METRICS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    P.CONTAMINATION: contamination_runs,
    P.EXTENSION: extension_runs,
    P.STEPWISE: stepwise_runs,
    P.DRIFT: drift_runs,
}
METRICS = {
    P.CONTAMINATION: ["recovery", "damage", "control_other_auroc_delta", "control_fpr_delta"],
    P.EXTENSION: ["acceptance", "extended_auroc_delta", "other_auroc_delta"],
    P.STEPWISE: ["auc", "final_auroc", "update_ms_per_frame"],
    P.DRIFT: ["target_auroc_delta", "target_fpr_after", "source_auroc_delta"],
}


# --- aggregation -----------------------------------------------------------------------


def common_runs(results: pd.DataFrame) -> pd.DataFrame:
    """Only the runs every model of a (dataset, scenario) has, so models are compared on
    the same categories, targets, grid settings and seeds."""
    if results.empty:
        return results
    duplicated = results.drop_duplicates([*KEY, "scenario", "experiment"]).duplicated(
        [*KEY, "scenario"]
    )
    if duplicated.any():
        raise ValueError(
            "the same runs of one model are in several experiments; pick some with "
            "load_results(..., experiments=[...])"
        )
    kept = []
    for (dataset, scenario), group in results.groupby(["dataset", "scenario"]):
        keys = group[[*RUN_FIELDS, "model"]].drop_duplicates()
        per_model = keys.groupby("model").apply(
            lambda k: set(map(tuple, k[list(RUN_FIELDS)].astype(object).to_numpy())),
            include_groups=False,
        )
        shared = set.intersection(*per_model)
        run_keys = pd.Series(
            list(map(tuple, group[list(RUN_FIELDS)].astype(object).to_numpy())), index=group.index
        )
        keep = run_keys.isin(shared)
        for model, own in per_model.items():
            if len(own) > len(shared):
                log.warning(
                    "%s/%s: leaving out %d of %s's %d runs that not every model has",
                    dataset, scenario, len(own) - len(shared), model, len(own),
                )
        kept.append(group[keep])
    return pd.concat(kept)


DRAW = [*GROUP, "seed", "update_seed"]


def _over_seeds(per_draw: pd.DataFrame) -> pd.DataFrame:
    """``mean`` / ``std`` over fit seeds of the update-seed means, and ``update_std``."""
    per_seed = per_draw.groupby([*GROUP, "seed"], dropna=False).mean()
    stats = per_seed.groupby(GROUP, dropna=False).agg(["mean", "std"])
    within = per_draw.groupby([*GROUP, "seed"], dropna=False).std().groupby(GROUP, dropna=False).mean()
    within.columns = pd.MultiIndex.from_product([within.columns, ["update_std"]])
    columns = [(m, stat) for m in per_draw.columns for stat in ("mean", "std", "update_std")]
    return stats.join(within)[columns]


def _aggregate(runs: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Mean within each category, then over categories, per (fit seed, update seed)."""
    values = runs[list(metrics)].reset_index()
    per_category = values.groupby([*DRAW, "category"], dropna=False)[list(metrics)].mean()
    return _over_seeds(per_category.groupby(DRAW, dropna=False).mean())


def _recovery(runs: pd.DataFrame, min_damage: float) -> pd.DataFrame:
    damaged = runs[runs["damage"] > min_damage].reset_index()
    pooled = damaged.groupby(DRAW, dropna=False)[["gain", "damage"]].sum()
    per_draw = (pooled["gain"] / pooled["damage"]).rename("recovery").to_frame()
    draws = runs.reset_index()[DRAW].drop_duplicates().set_index(DRAW)
    return _over_seeds(draws.join(per_draw))


def scorecard(
    results: pd.DataFrame, *, min_damage: float = 0.01, common: bool = True
) -> pd.DataFrame:
    """One row per (dataset, model), columns (scenario, metric, ``mean`` / ``std`` /
    ``update_std``) for every scenario in ``results``; see the module docstring.

    Args:
        results: Rows from `load_results`, or a `MaintenanceExperiment`'s results with
            ``dataset`` and ``model`` columns added.
        min_damage: Contamination runs that lost less target AUROC are left out of
            ``recovery``.
        common: Compare models on the runs they all have (`common_runs`).
    """
    if common:
        results = common_runs(results)
    parts = {}
    for scenario, run_metrics in RUN_METRICS.items():
        if not (results["scenario"] == scenario).any():
            continue
        runs = run_metrics(results)
        metrics = [m for m in METRICS[scenario] if m != "recovery"]
        table = _aggregate(runs, metrics)
        if scenario == P.CONTAMINATION:
            table = _recovery(runs, min_damage).join(table)
        parts[scenario] = table
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1, names=["scenario", "metric", "stat"])


def format_scorecard(card: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """``"mean ± std (upd ± update_std)"`` per (scenario, metric), leaving out spreads
    that need more than one fit / update seed."""

    def part(stat: str, fmt: str) -> pd.DataFrame:
        values = card.xs(stat, axis=1, level="stat")
        return values.map(lambda v: "" if pd.isna(v) else fmt.format(v))

    text = part("mean", f"{{:.{decimals}f}}")
    spread = part("std", f" ± {{:.{decimals}f}}") + part("update_std", f" (upd ± {{:.{decimals}f}})")
    return text + spread.where(text != "", "")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scorecard of admaint sweeps, one row per model.")
    parser.add_argument("root", nargs="?", default="runs", help="output_root of the sweeps")
    parser.add_argument("--experiments", nargs="*", help="experiment directories to include")
    parser.add_argument("--min-damage", type=float, default=0.01)
    parser.add_argument("--all-runs", action="store_true", help="don't restrict models to shared runs")
    parser.add_argument("--csv", type=Path, help="also write the numeric scorecard here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    results = load_results(args.root, args.experiments)
    if results.empty:
        raise SystemExit(f"no admaint results under {args.root}")
    card = scorecard(results, min_damage=args.min_damage, common=not args.all_runs)
    if args.csv is not None:
        card.to_csv(args.csv)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_colwidth", 30):
        for dataset, table in format_scorecard(card).groupby(level="dataset"):
            print(f"\n== {dataset} ==")
            print(table.droplevel("dataset").T.to_string())
