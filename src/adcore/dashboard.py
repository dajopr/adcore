"""Streamlit dashboard comparing few-shot sweeps recorded with `adcore.tracking`.

    adcore-dashboard                                  # reads ./mlflow.db
    adcore-dashboard --tracking-uri sqlite:////abs/path/mlflow.db --port 8501

All statistics come from `adcore.experiment.summarize` / `table`, applied per group
(architecture or sweep), so the numbers match what those print locally.

Needs the ``track`` extra (``mlflow``, ``streamlit``, ``plotly``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from adcore.evaluation import METRICS
from adcore.experiment import _shot_labels, summarize, table
from adcore.tracking import DEFAULT_TRACKING_URI, list_experiments, load_runs

# Categorical slots in fixed order (light, dark); a group keeps its slot when others
# are filtered out. Past eight, groups fall back to gray.
_SERIES = [
    ("#2a78d6", "#3987e5"),
    ("#eb6834", "#d95926"),
    ("#1baf7a", "#199e70"),
    ("#eda100", "#c98500"),
    ("#e87ba4", "#d55181"),
    ("#008300", "#008300"),
    ("#4a3aa7", "#9085e9"),
    ("#e34948", "#e66767"),
]
_OVERFLOW = ("#8a8984", "#8a8984")
_SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_FACET_COLUMNS = 4


def main() -> None:
    """``adcore-dashboard``: ``streamlit run`` this file, passing the options through."""
    from streamlit.web import cli

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--",
        "--tracking-uri",
        args.tracking_uri,
    ]
    sys.exit(cli.main())


def _app_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    return parser.parse_known_args()[0]


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _shot_order(results: pd.DataFrame) -> list[str]:
    return list(_shot_labels(results["shots"]).categories)


def _per_group_summary(
    results: pd.DataFrame, group_by: str, metric: str
) -> pd.DataFrame:
    """Long frame: group, shots (label), category (incl. "mean"), mean, std — in %."""
    parts = []
    for group, frame in results.groupby(group_by, sort=False):
        stats = summarize(frame, [metric])[metric].reset_index()
        stats["shots"] = stats["shots"].astype(str)
        stats["category"] = stats["category"].astype(str)
        parts.append(stats.assign(group=group))
    if not parts:
        return pd.DataFrame(columns=["group", "shots", "category", "mean", "std"])
    out = pd.concat(parts, ignore_index=True)
    out[["mean", "std"]] *= 100
    return out


def _add_line(fig, stats, group, color, order, *, row=None, col=None, legend=True):
    import plotly.graph_objects as go

    stats = stats.set_index("shots").reindex(order).dropna(subset=["mean"])
    x, mean, std = list(stats.index), stats["mean"], stats["std"].fillna(0)
    where = {"row": row, "col": col} if row is not None else {}
    if len(x) > 1 and std.any():
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=list(mean + std) + list((mean - std)[::-1]),
                fill="toself",
                fillcolor=_rgba(color, 0.15),
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=group,
            ),
            **where,
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            customdata=std,
            name=group,
            legendgroup=group,
            showlegend=legend,
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 8, "color": color},
            hovertemplate=f"<b>{group}</b><br>%{{x}} shot: %{{y:.1f}} ± %{{customdata:.1f}}<extra></extra>",
        ),
        **where,
    )


def _style(fig, title_y: str, height: int = 420):
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="closest",
    )
    fig.update_xaxes(type="category", title_text="shots", showgrid=False)
    fig.update_yaxes(title_text=title_y, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def app() -> None:
    import plotly.graph_objects as go
    import streamlit as st
    from plotly.subplots import make_subplots

    st.set_page_config(page_title="adcore sweeps", layout="wide")
    args = _app_args()

    @st.cache_data(ttl=60, show_spinner="Loading runs from MLflow…")
    def load(uri: str, experiment: str) -> pd.DataFrame:
        return load_runs(uri, experiments=[experiment])

    with st.sidebar:
        st.header("Data")
        uri = st.text_input("Tracking URI", args.tracking_uri)
        experiments = list_experiments(uri)
        experiments = [e for e in experiments if e != "Default"] or experiments
        if not experiments:
            st.info(f"No MLflow experiments in {uri}.")
            st.stop()
        experiment = st.selectbox("Experiment", experiments)
        if st.button("Reload", width="stretch"):
            load.clear()
        results = load(uri, experiment)
        if results.empty:
            st.info("No finished runs in this experiment yet.")
            st.stop()

        group_by = st.radio("Compare by", ["arch", "sweep"], horizontal=True)
        all_groups = sorted(results[group_by].dropna().unique())
        sweeps = st.multiselect(
            "Sweeps", sorted(results["sweep"].unique()), default=sorted(results["sweep"].unique())
        )
        metric = st.selectbox("Metric", METRICS)
        all_categories = list(dict.fromkeys(results["category"]))
        categories = st.multiselect("Categories", all_categories, default=all_categories)

    dark = getattr(getattr(st.context, "theme", None), "type", None) == "dark"
    colors = {
        group: (_SERIES[i] if i < len(_SERIES) else _OVERFLOW)[int(dark)]
        for i, group in enumerate(all_groups)
    }
    if len(all_groups) > len(_SERIES):
        st.warning(
            f"{len(all_groups)} groups; past {len(_SERIES)} they share gray. "
            "Narrow the sweeps to compare them by color."
        )

    results = results[results["sweep"].isin(sweeps) & results["category"].isin(categories)]
    if results.empty:
        st.info("Nothing selected.")
        st.stop()
    groups = [g for g in all_groups if g in set(results[group_by])]
    order = _shot_order(results)
    stats = _per_group_summary(results, group_by, metric)
    label = f"{metric} (%)"

    st.title(experiment)
    overall = results[results["defect_type"] == "all"]
    cols = st.columns(4)
    cols[0].metric("Groups", len(groups))
    cols[1].metric("Sweeps", overall["sweep"].nunique())
    cols[2].metric("Runs", len(overall))
    cols[3].metric("Categories", overall["category"].nunique())

    overview, per_category, heatmap, tables, defects, cost = st.tabs(
        ["Overview", "Per category", "Heatmap", "Tables", "Defect types", "Cost"]
    )

    with overview:
        st.caption(
            f"Mean over the selected categories; band = std over seeds of that mean "
            f"(as in `summarize`). Compared by {group_by}."
        )
        fig = go.Figure()
        for group in groups:
            mine = stats[(stats["group"] == group) & (stats["category"] == "mean")]
            _add_line(fig, mine, group, colors[group], order)
        st.plotly_chart(_style(fig, label), width="stretch")
        best = (
            stats[stats["category"] == "mean"]
            .pivot(index="group", columns="shots", values="mean")
            .reindex(columns=[s for s in order if s in set(stats["shots"])])
            .round(1)
        )
        st.dataframe(best, width="stretch")

    with per_category:
        cats = [c for c in categories if c in set(stats["category"])]
        n_rows = -(-len(cats) // _FACET_COLUMNS)
        fig = make_subplots(
            rows=n_rows,
            cols=_FACET_COLUMNS,
            subplot_titles=cats,
            shared_xaxes=True,
            vertical_spacing=0.25 / max(n_rows, 1),
            horizontal_spacing=0.04,
        )
        for i, category in enumerate(cats):
            for group in groups:
                mine = stats[(stats["group"] == group) & (stats["category"] == category)]
                _add_line(
                    fig,
                    mine,
                    group,
                    colors[group],
                    order,
                    row=i // _FACET_COLUMNS + 1,
                    col=i % _FACET_COLUMNS + 1,
                    legend=i == 0,
                )
        _style(fig, "", height=240 * n_rows + 60)
        fig.update_xaxes(title_text="")
        st.plotly_chart(fig, width="stretch")

    with heatmap:
        shots = st.select_slider(
            "Shots", options=[s for s in order if s in set(stats["shots"])], key="heat_shots"
        )
        grid = (
            stats[(stats["shots"] == shots)]
            .pivot(index="category", columns="group", values="mean")
            .reindex(index=[*categories, "mean"], columns=groups)
            .dropna(how="all")
        )
        fig = go.Figure(
            go.Heatmap(
                z=grid.values,
                x=list(grid.columns),
                y=list(grid.index),
                colorscale=[[i / (len(_SEQUENTIAL) - 1), c] for i, c in enumerate(_SEQUENTIAL)],
                text=grid.round(1).values,
                texttemplate="%{text}",
                xgap=2,
                ygap=2,
                colorbar={"title": label},
                hovertemplate="%{y} · %{x}: %{z:.1f}<extra></extra>",
            )
        )
        fig.update_layout(
            height=28 * len(grid) + 120,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(fig, width="stretch")

    with tables:
        st.caption(f"`table(results, {metric!r})` per {group_by}: mean ± std over seeds, in %.")
        for group in groups:
            st.subheader(group)
            st.dataframe(
                table(results[results[group_by] == group], metric), width="stretch"
            )

    with defects:
        left, right = st.columns(2)
        category = left.selectbox("Category", categories, key="defect_category")
        shots = right.select_slider("Shots", options=order, key="defect_shots")
        per_defect = results[
            (results["category"] == category)
            & (results["defect_type"] != "all")
            & (_shot_labels(results["shots"]).astype(str) == shots)
        ]
        if per_defect.empty:
            st.info("No per-defect rows for this selection.")
        else:
            agg = (
                per_defect.groupby([group_by, "defect_type"])[metric]
                .agg(["mean", "std"])
                .mul(100)
                .reset_index()
            )
            fig = go.Figure()
            for group in groups:
                mine = agg[agg[group_by] == group]
                fig.add_trace(
                    go.Bar(
                        x=mine["defect_type"],
                        y=mine["mean"],
                        error_y={"type": "data", "array": mine["std"].fillna(0), "thickness": 1},
                        name=group,
                        marker={"color": colors[group], "cornerradius": 4},
                        hovertemplate=f"<b>{group}</b><br>%{{x}}: %{{y:.1f}}<extra></extra>",
                    )
                )
            fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
            _style(fig, label)
            fig.update_xaxes(title_text="defect type")
            st.plotly_chart(fig, width="stretch")

    with cost:
        st.caption("Fit + test seconds per run, mean over categories and seeds.")
        timing = overall.assign(
            seconds=overall["fit_seconds"] + overall["test_seconds"],
            shots=_shot_labels(overall["shots"]).astype(str),
        )
        fig = go.Figure()
        for group in groups:
            mine = (
                timing[timing[group_by] == group]
                .groupby("shots")["seconds"]
                .agg(["mean", "std"])
                .reset_index()
            )
            _add_line(fig, mine, group, colors[group], order)
        st.plotly_chart(_style(fig, "seconds per run"), width="stretch")


if __name__ == "__main__":
    app()
