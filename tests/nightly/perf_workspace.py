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
import unicodedata
from os import getenv
from urllib.parse import urlsplit

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

WANDB_ENTITY = getenv("WANDB_ENTITY", "amazon-far")
WANDB_PROJECT = "nightly-holosoma-runs"

DEFAULT_VIEW_NAME = "Nightly performance trends"

# Summary keys written by backfill_perf_summary.py. Keep in sync with that script.
SUMMARY_PREFIX = "perf/"
RUN_DATE_KEY = f"{SUMMARY_PREFIX}run_date"
RUN_DURATION_KEY = f"{SUMMARY_PREFIX}run_duration_s"
CELL_KEY = f"{SUMMARY_PREFIX}cell"

# (section title, summary key stem) for the three per-iteration metrics.
METRIC_SECTIONS = (
    ("Training throughput (fps)", "total_fps"),
    ("Collection time per iteration (s)", "collection_time"),
    ("Learning time per iteration (s)", "learning_time"),
)


def _stat_key(stem: str, stat: str) -> str:
    """Summary key for one metric stem/stat pair, e.g. `perf/total_fps_mean`."""
    return f"{SUMMARY_PREFIX}{stem}_{stat}"


def _safe_view_name(name: str) -> str:
    """Replaces characters wandb refuses in a view name.

    wandb rejects symbol characters in view names, reporting "Emojis are not allowed"
    whatever the character actually was. The categories it refuses include Sm, which
    covers the "|" separating the parts of a cell label. Only the name is affected; the
    label itself is still matched verbatim by the runset filter.
    """
    disallowed = ("So", "Sk", "Sm", "Sc", "Cs")
    cleaned = "".join("-" if unicodedata.category(c).startswith(disallowed) else c for c in name)
    return " ".join(cleaned.split())


def _scatter(key: str, title: str, trend: bool) -> wr.ScatterPlot:
    """One point per run against when that run started.

    The x-axis is wandb's own run-creation field rather than the numeric `perf/run_date`
    summary key, because the backend knows that field is a timestamp and labels the axis
    with dates. A summary key is just a number, so it renders as raw epoch seconds, and
    ScatterPlot has no axis-format option to correct it.

    `trend` adds a regression line and a running min/max envelope across runs -- these
    describe the night-to-night series, not the within-run spread.
    """
    return wr.ScatterPlot(
        title=title,
        x=wr.Metric("CreatedTimestamp"),
        y=wr.SummaryMetric(key),
        regression=trend,
        running_ymin=trend,
        running_ymax=trend,
    )


def _runset_settings(cell: str | None) -> ws.RunsetSettings:
    """Runset ordered by run date, optionally narrowed to a single matrix cell.

    Deliberately ungrouped. Grouping collapses each group to one aggregated point rather
    than one point per run, and the aggregation cannot combine run timestamps: the
    x-axis degenerates to meaningless dates (the Unix epoch among them) and the
    regression reports NaN. Cells are separated by filtering a view to one of them
    instead, which keeps a point per run.
    """
    return ws.RunsetSettings(
        filters=[ws.Summary(CELL_KEY) == cell] if cell else [],
        order=[ws.Ordering(ws.Summary(RUN_DATE_KEY))],
    )


def _metric_section(title: str, stem: str, trend: bool) -> ws.Section:
    """Min, mean and max per run for one metric."""
    return ws.Section(
        name=title,
        is_open=True,
        panels=[
            _scatter(_stat_key(stem, "mean"), f"{stem} mean per run", trend=trend),
            _scatter(_stat_key(stem, "min"), f"{stem} min per run", trend=False),
            _scatter(_stat_key(stem, "max"), f"{stem} max per run", trend=False),
        ],
    )


def _duration_section(trend: bool) -> ws.Section:
    """Total wall-clock duration. One value per run, so no min/mean/max."""
    return ws.Section(
        name="Total run duration (s)",
        is_open=True,
        panels=[_scatter(RUN_DURATION_KEY, "Run duration per run", trend=trend)],
    )


def build_workspace(view_name: str, project: str, trend: bool = True, cell: str | None = None) -> ws.Workspace:
    """Assembles the view: one section per metric, plus run duration."""
    sections = [_metric_section(title, stem, trend) for title, stem in METRIC_SECTIONS]
    sections.append(_duration_section(trend))

    return ws.Workspace(
        entity=WANDB_ENTITY,
        project=project,
        name=view_name,
        sections=sections,
        runset_settings=_runset_settings(cell),
    )


def _browser_url(saved_url: str, project: str) -> str:
    """The saved view's URL as a browser needs it.

    View URLs are built from the host alone, dropping any path prefix a deployment
    serves the app under, which gives a 404 where one is used. A run's own URL carries
    the prefix, so rebuild the view URL on top of that. Falls back to the original if no
    run is readable.
    """
    try:
        api = wandb.Api(timeout=60)
        run = next(iter(api.runs(path=f"{WANDB_ENTITY}/{project}", per_page=1)), None)
        run_url = getattr(run, "url", None) if run else None
        if not run_url or "/runs/" not in run_url:
            return saved_url
        return f"{run_url.rsplit('/runs/', 1)[0]}?{urlsplit(saved_url).query}"
    except Exception as e:
        print(f"Could not resolve the browser URL, reporting the raw one: {e}")
        return saved_url


def main() -> None:
    parser = argparse.ArgumentParser(
        "perf_workspace",
        description="Creates a saved wandb workspace view of nightly performance trends",
    )
    parser.add_argument("--name", default=DEFAULT_VIEW_NAME, help="Name for the saved view")
    parser.add_argument("--project", default=WANDB_PROJECT, help="wandb project to build the view against")
    parser.add_argument("--no-trend", action="store_true", help="Omit the regression line and running min/max overlays")
    parser.add_argument(
        "--cell",
        default=None,
        help='Restrict the view to one matrix cell, e.g. "t1_29dof_manager | isaacsim | gpus-4"',
    )
    args = parser.parse_args()

    name = _safe_view_name(args.name if args.cell is None else f"{args.name} - {args.cell}")
    workspace = build_workspace(name, args.project, trend=not args.no_trend, cell=args.cell)
    saved = workspace.save_as_new_view()
    print(f"Saved workspace view: {_browser_url(saved.url, args.project)}")


if __name__ == "__main__":
    main()
