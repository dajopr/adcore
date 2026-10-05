"""Few-shot results in a local MLflow store, so sweeps of different architectures can be
compared side by side (``adcore-dashboard``, or ``mlflow ui``).

    tracking = MLflowTracking(experiment="mvtec-fewshot", arch="patchcore-wrn50")
    FewShotExperiment(..., output_dir="runs/patchcore-wrn50", tracking=tracking).run()
    upload_sweep("runs/older-sweep", MLflowTracking(experiment="mvtec-fewshot", arch="..."))
    results = load_runs("sqlite:///mlflow.db", experiments=["mvtec-fewshot"])

Layout: one MLflow experiment per benchmark, one parent run per sweep (tag ``sweep``) and
a nested child run per (category, shots, seed). A child's metrics are its whole-test-set
row (``image_auroc``, ...) plus one ``defect/<type>/<metric>`` per defect type, so
`load_runs` can rebuild the rows of ``results.jsonl``. ``results.jsonl`` stays the source
of truth for resuming; MLflow is a copy for comparison.

Needs the ``track`` extra (``mlflow``).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from adcore.evaluation import METRICS

if TYPE_CHECKING:
    from mlflow import MlflowClient

    from adcore.experiment import Run

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

# Columns that identify a run, and those shared by all of a run's rows.
_KEY = ("category", "shots", "seed")
_RUN_FIELDS = ("n_train", "n_anomalous_train", "pro_connectivity", "fit_seconds", "test_seconds")
# Per (category, defect_type) columns of `adcore.evaluation.score`; anything else on a
# run (e.g. a module's training losses via Lightning) is not part of its results rows.
_ROW_FIELDS = (*METRICS, "n_images", "n_anomalous")
# The sweep grid grows between sessions; children carry it, the parent does not.
_GRID = ("categories", "shots", "seeds")
_KIND = "adcore_kind"
_PARENT = "mlflow.parentRunId"
_MAX_PARAM = 6000


def _shots_param(shots: int | None) -> str:
    return "full" if shots is None else str(shots)


def _parse_shots(value: str) -> int | None:
    return None if value == "full" else int(value)


def _metric_name(defect_type: str, column: str) -> str:
    if defect_type == "all":
        return column
    safe = re.sub(r"[^\w\-. ]", "_", defect_type)
    return f"defect/{safe}/{column}"


@dataclass
class MLflowTracking:
    """Where and under which labels a sweep's runs are recorded.

    Args:
        experiment: MLflow experiment, e.g. one per benchmark (``"mvtec-fewshot"``).
        arch: The label sweeps are compared by in the dashboard.
        tracking_uri: Defaults to ``mlflow.db`` in the working directory.
        params: Extra params on every run (backbone, layers, ...).
        tags: Extra tags on every run.
        sweep: Name of the parent run; `FewShotExperiment` defaults it to the name of
            its ``output_dir``.
    """

    experiment: str
    arch: str
    tracking_uri: str = DEFAULT_TRACKING_URI
    params: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    sweep: str | None = None

    def __post_init__(self) -> None:
        self._client: MlflowClient | None = None
        self._experiment_id: str | None = None

    @property
    def client(self) -> MlflowClient:
        if self._client is None:
            from mlflow import MlflowClient

            self._client = MlflowClient(self.tracking_uri)
        return self._client

    @property
    def experiment_id(self) -> str:
        if self._experiment_id is None:
            found = self.client.get_experiment_by_name(self.experiment)
            self._experiment_id = (
                found.experiment_id
                if found is not None
                else self.client.create_experiment(self.experiment)
            )
        return self._experiment_id

    def _tags(self, sweep: str, kind: str) -> dict[str, str]:
        return {**self.tags, "sweep": sweep, "arch": self.arch, _KIND: kind}

    def start_sweep(self, sweep: str, config: dict[str, Any]) -> str:
        """The sweep's parent run id, created on first use and reused on resume."""
        found = self.client.search_runs(
            [self.experiment_id],
            f"tags.{_KIND} = 'sweep' and tags.sweep = '{sweep}'",
            max_results=1,
        )
        params = {
            key: value
            for key, value in {**config, **self.params}.items()
            if key not in _GRID
        }
        if found:
            run = found[0]
            # Params are immutable in MLflow; keep the first session's and add new keys.
            params = {k: v for k, v in params.items() if k not in run.data.params}
            self.client.update_run(run.info.run_id, status="RUNNING")
            run_id = run.info.run_id
        else:
            run_id = self.client.create_run(
                self.experiment_id, run_name=sweep, tags=self._tags(sweep, "sweep")
            ).info.run_id
        self._log(run_id, params=params)
        return run_id

    def end_sweep(self, parent_id: str) -> None:
        self.client.set_terminated(parent_id)

    def start_run(self, parent_id: str, sweep: str, run: Run, output_dir: Path | None = None) -> str:
        """A child run for one (category, shots, seed)."""
        tags = {**self._tags(sweep, "run"), _PARENT: parent_id}
        if output_dir is not None:
            tags["output_dir"] = str(output_dir)
        run_id = self.client.create_run(
            self.experiment_id, run_name=run.name, tags=tags
        ).info.run_id
        self._log(
            run_id,
            params={
                **self.params,
                "arch": self.arch,
                "category": run.category,
                "shots": _shots_param(run.shots),
                "seed": run.seed,
            },
        )
        return run_id

    def log_rows(self, run_id: str, rows: Sequence[dict[str, Any]]) -> None:
        """A run's ``results.jsonl`` rows as metrics, and the run marked finished."""
        metrics: dict[str, float] = {}
        for row in rows:
            for column, value in row.items():
                if column in (*_KEY, "defect_type") or not isinstance(value, (int, float)):
                    continue
                name = column if column in _RUN_FIELDS else _metric_name(row["defect_type"], column)
                metrics[name] = float(value)
        self._log(run_id, metrics=metrics)
        self.client.set_terminated(run_id)

    def fail_run(self, run_id: str) -> None:
        self.client.set_terminated(run_id, status="FAILED")

    def logged_keys(self, parent_id: str) -> set[tuple[str, int | None, int]]:
        """(category, shots, seed) of the sweep's finished children."""
        return {
            (p["category"], _parse_shots(p["shots"]), int(p["seed"]))
            for p in (
                run.data.params
                for run in _search_all(
                    self.client,
                    [self.experiment_id],
                    f"tags.{_KIND} = 'run' and tags.`{_PARENT}` = '{parent_id}' "
                    "and attributes.status = 'FINISHED'",
                )
            )
        }

    def _log(self, run_id: str, params: dict[str, Any] | None = None, metrics: dict[str, float] | None = None) -> None:
        from mlflow.entities import Metric, Param

        now = int(time.time() * 1000)
        param_list = [
            Param(key, _param_value(value)) for key, value in (params or {}).items()
        ]
        metric_list = [Metric(key, value, now, 0) for key, value in (metrics or {}).items()]
        # log_batch takes at most 100 params and 1000 metrics per call.
        for i in range(0, len(param_list), 100):
            self.client.log_batch(run_id, params=param_list[i : i + 100])
        for i in range(0, len(metric_list), 1000):
            self.client.log_batch(run_id, metrics=metric_list[i : i + 1000])


def _param_value(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=repr)
    return text[:_MAX_PARAM]


def _search_all(client: MlflowClient, experiment_ids: list[str], filter_string: str) -> Iterable[Any]:
    token = None
    while True:
        page = client.search_runs(
            experiment_ids, filter_string, max_results=1000, page_token=token
        )
        yield from page
        token = page.token
        if not token:
            return


def upload_sweep(output_dir: str | Path, tracking: MLflowTracking) -> int:
    """Copy an existing sweep directory's ``results.jsonl`` into MLflow.

    Runs already recorded under the sweep are skipped, so it is safe to repeat; the sweep
    name defaults to the directory name. Returns how many runs were added.
    """
    output_dir = Path(output_dir)
    sweep = tracking.sweep or output_dir.name
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    with (output_dir / "results.jsonl").open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    parent_id = tracking.start_sweep(sweep, config)
    added = backfill(tracking, parent_id, sweep, rows, output_dir)
    tracking.end_sweep(parent_id)
    return added


def backfill(
    tracking: MLflowTracking,
    parent_id: str,
    sweep: str,
    rows: Sequence[dict[str, Any]],
    output_dir: Path | None = None,
) -> int:
    """Record the runs in ``rows`` (``results.jsonl`` rows) that the sweep lacks."""
    from adcore.experiment import Run

    by_run: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(tuple(row[k] for k in _KEY), []).append(row)
    done = tracking.logged_keys(parent_id)
    todo = [key for key in by_run if key not in done]
    for key in todo:
        run_id = tracking.start_run(parent_id, sweep, Run(*key), output_dir)
        tracking.log_rows(run_id, by_run[key])
    return len(todo)


def list_experiments(tracking_uri: str = DEFAULT_TRACKING_URI) -> list[str]:
    from mlflow import MlflowClient

    return sorted(e.name for e in MlflowClient(tracking_uri).search_experiments())


def load_runs(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiments: Sequence[str] | None = None,
    sweeps: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Every finished run's rows, shaped like `adcore.experiment.results_frame` output,
    plus ``experiment``, ``sweep`` and ``arch`` columns.

    The rows are those of ``results.jsonl``, so `adcore.experiment.summarize` and
    `adcore.experiment.table` work on any subset (e.g. one ``arch``).
    """
    from mlflow import MlflowClient

    from adcore.experiment import results_frame

    client = MlflowClient(tracking_uri)
    found = client.search_experiments()
    names = {e.experiment_id: e.name for e in found if experiments is None or e.name in experiments}
    if not names:
        return results_frame([])

    rows = []
    runs = _search_all(
        client,
        list(names),
        f"tags.{_KIND} = 'run' and attributes.status = 'FINISHED'",
    )
    for run in runs:
        tags, params, metrics = run.data.tags, run.data.params, run.data.metrics
        if sweeps is not None and tags.get("sweep") not in sweeps:
            continue
        shared = {
            "experiment": names[run.info.experiment_id],
            "sweep": tags.get("sweep"),
            "arch": tags.get("arch"),
            "category": params["category"],
            "shots": _parse_shots(params["shots"]),
            "seed": int(params["seed"]),
            "start_time": run.info.start_time,
            **{k: metrics[k] for k in _RUN_FIELDS if k in metrics},
        }
        by_defect: dict[str, dict[str, float]] = {}
        for name, value in metrics.items():
            if name in _RUN_FIELDS:
                continue
            if name.startswith("defect/"):
                _, defect_type, column = name.split("/", 2)
            else:
                defect_type, column = "all", name
            if column not in _ROW_FIELDS:
                continue
            by_defect.setdefault(defect_type, {})[column] = value
        for defect_type in sorted(by_defect, key=lambda d: (d != "all", d)):
            rows.append({**shared, "defect_type": defect_type, **by_defect[defect_type]})

    frame = results_frame(rows)
    if frame.empty:
        return frame
    # A run recorded twice (e.g. interrupted before results.jsonl got it): keep the latest.
    return (
        frame.sort_values("start_time", kind="stable")
        .drop_duplicates(["experiment", "sweep", *_KEY, "defect_type"], keep="last")
        .sort_values(["experiment", "sweep", *_KEY], kind="stable", na_position="last")
        .reset_index(drop=True)
    )
