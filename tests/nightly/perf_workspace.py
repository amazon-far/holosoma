#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "wandb",
#   "wandb-workspaces",
# ]
# ///
"""Creates a saved wandb workspace view plotting nightly performance over time.

Each panel shows one point per nightly run against when that run started, so
throughput and timing drift are visible across nights rather than within a run.
The values come from the run summary keys written by backfill_perf_summary.py --
run that first, or the panels will be empty.
"""

from __future__ import annotations

import argparse
from os import getenv

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

WANDB_ENTITY = getenv("WANDB_ENTITY", "amazon-far")
WANDB_PROJECT = "nightly-holosoma-runs"

DEFAULT_VIEW_NAME = "Nightly performance trends"

# Summary keys written by backfill_perf_summary.py. Keep in sync with that script.
SUMMARY_PREFIX = "perf/"
RUN_DATE_KEY = f"{SUMMARY_PREFIX}run_date"
RUN_DURATION_KEY = f"{SUMMARY_PREFIX}run_duration_s"

# (section title, summary key stem) for the three per-iteration metrics.
METRIC_SECTIONS = (
    ("Training throughput (fps)", "total_fps"),
    ("Collection time per iteration (s)", "collection_time"),
    ("Learning time per iteration (s)", "learning_time"),
)


def _stat_key(stem: str, stat: str) -> str:
    """Summary key for one metric stem/stat pair, e.g. `perf/total_fps_mean`."""
    return f"{SUMMARY_PREFIX}{stem}_{stat}"


def _scatter(key: str, title: str, trend: bool) -> wr.ScatterPlot:
    """One point per run against run start time.

    `trend` adds a regression line and a running min/max envelope across runs -- these
    describe the night-to-night series, not the within-run spread.
    """
    return wr.ScatterPlot(
        title=title,
        x=wr.SummaryMetric(RUN_DATE_KEY),
        y=wr.SummaryMetric(key),
        regression=trend,
        running_ymin=trend,
        running_ymax=trend,
    )


def _runset_settings() -> ws.RunsetSettings:
    """Runset ordered by run date, with every run in the project selected.

    Deliberately not grouped. Grouping by tags would be the natural way to separate the
    experiment / simulator / GPU-count matrix cells, but `Tags()` serializes into the
    saved view as a metric named "Tags", which matches nothing: the runset then resolves
    to zero runs and every panel fails. The same underlying bug is noted against
    `FilterExpr` in generate_report.py. Grouping is a one-click control in the UI, so
    it is left to the reader.
    """
    return ws.RunsetSettings(order=[ws.Ordering(ws.Summary(RUN_DATE_KEY))])


def _bar(keys: list[str], title: str) -> wr.BarPlot:
    """All requested summary keys side by side, one bar group per run."""
    return wr.BarPlot(
        title=title,
        metrics=[wr.SummaryMetric(key) for key in keys],
        orientation="v",
    )


def _metric_section(title: str, stem: str, trend: bool) -> ws.Section:
    """Min/mean/max scatters plus a combined bar panel for one metric."""
    return ws.Section(
        name=title,
        is_open=True,
        panels=[
            _scatter(_stat_key(stem, "mean"), f"{stem} mean per run", trend=trend),
            _scatter(_stat_key(stem, "min"), f"{stem} min per run", trend=False),
            _scatter(_stat_key(stem, "max"), f"{stem} max per run", trend=False),
            _bar([_stat_key(stem, stat) for stat in ("min", "mean", "max")], f"{stem} min/mean/max by run"),
        ],
    )


def _duration_section(trend: bool) -> ws.Section:
    """Total wall-clock duration. One value per run, so no min/mean/max."""
    return ws.Section(
        name="Total run duration (s)",
        is_open=True,
        panels=[
            _scatter(RUN_DURATION_KEY, "Run duration per run", trend=trend),
            _bar([RUN_DURATION_KEY], "Run duration by run"),
        ],
    )


def build_workspace(view_name: str, project: str, trend: bool = True) -> ws.Workspace:
    """Assembles the view: one section per metric, plus run duration."""
    sections = [_metric_section(title, stem, trend) for title, stem in METRIC_SECTIONS]
    sections.append(_duration_section(trend))

    return ws.Workspace(
        entity=WANDB_ENTITY,
        project=project,
        name=view_name,
        sections=sections,
        runset_settings=_runset_settings(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        "perf_workspace",
        description="Creates a saved wandb workspace view of nightly performance trends",
    )
    parser.add_argument("--name", default=DEFAULT_VIEW_NAME, help="Name for the saved view")
    parser.add_argument("--project", default=WANDB_PROJECT, help="wandb project to build the view against")
    parser.add_argument("--no-trend", action="store_true", help="Omit the regression line and running min/max overlays")
    args = parser.parse_args()

    workspace = build_workspace(args.name, args.project, trend=not args.no_trend)
    saved = workspace.save_as_new_view()
    print(f"Saved workspace view: {saved.url}")


if __name__ == "__main__":
    main()
