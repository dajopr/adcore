"""``adcore-progress``: how far `FewShotExperiment` output directories are, with an ETA.

    adcore-progress runs/mvtec/my_model runs/visa/my_model
    adcore-progress runs/mvtec/*/ --window 30        # rate over the last 30 minutes
    watch -n 300 adcore-progress runs/mvtec/my_model

A directory's expected runs are its ``config.json``'s categories x shots x seeds; a run is
finished once ``results.jsonl`` has its ``defect_type == "all"`` row. The rate counts the
``scores/*.csv`` (one per finished run) written within the window, so the ETA treats every
run alike: read it as rough when categories differ in size.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Progress:
    directory: Path
    expected: int
    finished: int
    recent: int  # runs finished within the window
    window_minutes: float
    missing_categories: list[str]

    @property
    def rate_per_hour(self) -> float:
        return 60 * self.recent / self.window_minutes

    @property
    def eta_hours(self) -> float | None:
        left = self.expected - self.finished
        if left <= 0:
            return 0.0
        return left / self.rate_per_hour if self.recent else None


def progress(directory: str | Path, window_minutes: float = 60.0, now: float | None = None) -> Progress:
    """`Progress` of one output directory."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    expected = {
        (c, s, sd) for c in config["categories"] for s in config["shots"] for sd in config["seeds"]
    }
    finished = set()
    results = directory / "results.jsonl"
    if results.exists():
        for line in results.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("defect_type") == "all":
                    finished.add((row["category"], row.get("shots"), row["seed"]))
    finished &= expected
    cutoff = (now or time.time()) - 60 * window_minutes
    recent = sum(p.stat().st_mtime >= cutoff for p in (directory / "scores").glob("*.csv"))
    done_by_category = {c for c, *_ in finished}
    missing = [
        c
        for c in config["categories"]
        if c not in done_by_category or any(k[0] == c and k not in finished for k in expected)
    ]
    return Progress(directory, len(expected), len(finished), recent, window_minutes, missing)


def _hours(h: float | None) -> str:
    if h is None:
        return "?"
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dirs", nargs="+", type=Path, help="FewShotExperiment output directories")
    parser.add_argument("--window", type=float, default=60.0, help="minutes the rate is taken over")
    args = parser.parse_args(argv)
    for directory in args.dirs:
        if not (directory / "config.json").exists():
            print(f"{directory}: no config.json, not an output directory")
            continue
        p = progress(directory, args.window)
        print(
            f"{directory}: {p.finished}/{p.expected} runs ({100 * p.finished / max(p.expected, 1):.0f}%), "
            f"{p.rate_per_hour:.1f}/h over {args.window:g} min, ETA {_hours(p.eta_hours)}"
            + (f"; unfinished: {', '.join(p.missing_categories)}" if p.missing_categories else "")
        )


if __name__ == "__main__":
    main()
