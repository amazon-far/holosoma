#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "wandb",
#   "slack-sdk",
# ]
# ///

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from os import getenv
from typing import NamedTuple

import wandb

WANDB_ENTITY = getenv("WANDB_ENTITY", "amazon-far")
SLACK_CHANNEL = getenv("SLACK_CHANNEL", "")
SLACK_TOKEN = getenv("SLACK_BOT_TOKEN", "")

# Github assigned variables
GITHUB_SERVER_URL = getenv("GITHUB_SERVER_URL")
GITHUB_REPOSITORY = getenv("GITHUB_REPOSITORY")
GITHUB_RUN_ID = getenv("GITHUB_RUN_ID")

# Throughput regression warning applied by nightly.py. Keep in sync with that script.
PERF_WARNING_TAG = "nightly_perf_warning"

# Performance trend included in the Slack summary, one block per cell. The cell label and the
# metric key are written by backfill_perf_summary.py; keep in sync with that script.
PERF_PROJECT = "nightly-holosoma-runs"
PERF_CELL_KEY = "perf/cell"
TREND_CELLS = (
    "g1_29dof_manager | isaacsim | gpus-1",
    "g1_29dof_manager | mjwarp | gpus-1",
)
TREND_METRIC_KEY = "perf/collection_time_mean"
TREND_METRIC_LABEL = "collection_time mean"
# Unit of the tracked metric, appended to every figure. Change alongside the metric key.
TREND_UNIT = "s"
# Link offered under each cell's figures, pointing at a saved workspace view of that cell.
# A view URL carries nothing but the view's identifier -- the cell filter lives in the
# saved view rather than in the URL, so one view per cell is needed and no single base URL
# can be narrowed to a cell. That identifier also only exists on the instance the view was
# created on, so each one is named from the environment rather than here, in
# `PERF_TREND_VIEW_URL_<cell>` (see `_trend_view_url`). Left unset, the link falls back to
# the project the runs were read from, derived from a run's own URL and so correct on any
# instance.
TREND_VIEW_URL_ENV_PREFIX = "PERF_TREND_VIEW_URL"
TREND_VIEW_LABEL = "Performance trends"
# Enough history to cover the longest comparison window and its predecessor.
TREND_LOOKBACK_DAYS = 60
# How many standard deviations the two figures must be apart before the change counts as
# real rather than noise. The spread used is that of the *difference*, so a comparison
# accounts for the uncertainty on both sides rather than only the earlier one.
TREND_SIGMA = 2.0
# A single night has no spread of its own, so it is judged against how much this cell
# normally varies from night to night, measured over this many recent runs.
TREND_DAILY_BASELINE_RUNS = 14
# Whether a falling value is an improvement. True for a timing metric, false if the
# tracked metric is changed to something like throughput, where higher is better.
TREND_LOWER_IS_BETTER = True
# Verdict markers. Colour rather than direction: the figures either side of the arrow
# already show which way the metric moved, so what the marker adds is whether that is
# good or bad. Deliberately circles, not the squares the run-status legend uses, so the
# two sets don't read as the same thing in one message.
TREND_BETTER = "🟢"
TREND_WORSE = "🔴"
TREND_FLAT = "⚪"

# Simulator tag values set by nightly.simulator_tag, which is not always a config name --
# both MuJoCo backends report `mujoco`, and the GPU one is tagged `mjwarp`. "isaacgym" is retained
# for ARCHIVED runs only -- the backend is retired; dropping it would make historical runs resolve
# their exp_name to the literal tag instead of falling back to "unknown-sim".
KNOWN_SIMULATORS = {"isaacgym", "isaacsim", "mujoco", "mjwarp"}
# GPU count for a multi-GPU nightly run; used only as a fallback for legacy runs
# tagged with "multigpu" but no explicit "gpus-N" tag. Keep in sync with nightly.py.
MULTIGPU_NUM_GPUS = 4


class RunStatus(Enum):
    """Enum representing the possible states of a nightly test run."""

    SUCCEEDED = auto()
    CRASHED = auto()
    METRICS_REGRESSION = auto()
    PERF_WARNING = auto()
    FAILED = auto()
    UNKNOWN = auto()


# Mapping to help visually determine what happened via a slack message.
RunStatus2Emoji = {
    RunStatus.SUCCEEDED: "🟩",
    RunStatus.CRASHED: "🟥",
    RunStatus.METRICS_REGRESSION: "⚠️",
    RunStatus.PERF_WARNING: "🐌",
    RunStatus.FAILED: "🚫",
    RunStatus.UNKNOWN: "❓",
}

# Statuses that shouldn't be called out as failures. A throughput warning means the
# run passed its metrics but was slower than recent runs of the same matrix cell.
HEALTHY_STATUSES = frozenset({RunStatus.SUCCEEDED, RunStatus.PERF_WARNING})


class NightlyRun(NamedTuple):
    """A single nightly wandb run, enriched with the details we surface in Slack."""

    url: str
    status: RunStatus
    exp_name: str
    simulator: str  # e.g. "isaacsim"/"mujoco", or "unknown-sim" if untagged
    gpus: str  # e.g. "1gpu"/"4gpu", or "unknown-gpu" if untagged

    def failed_run_label(self) -> str:
        """Human-readable label for the "Failed runs" list.

        Includes simulator + GPU count so runs of the SAME experiment across
        different matrix dimensions (e.g. single- vs multi-GPU, isaacgym vs
        isaacsim) are distinguishable rather than collapsing to a duplicate.
        """
        return f"{self.exp_name} ({self.simulator}, {self.gpus})"


def _gpus_from_tags(run_tags: list[str]) -> str:
    """Derives a GPU-count label from a run's wandb tags.

    Prefers the explicit "gpus-N" tag; falls back to the legacy
    "singlegpu"/"multigpu" tags for older runs; else "unknown-gpu".
    """
    for tag in run_tags:
        if tag.startswith("gpus-"):
            return f"{tag[len('gpus-') :]}gpu"
    if "multigpu" in run_tags:
        return f"{MULTIGPU_NUM_GPUS}gpu"
    if "singlegpu" in run_tags:
        return "1gpu"
    return "unknown-gpu"


def _simulator_from_tags(run_tags: list[str]) -> str:
    """Derives the simulator name from a run's wandb tags, else "unknown-sim"."""
    for tag in run_tags:
        if tag in KNOWN_SIMULATORS:
            return tag
    return "unknown-sim"


def _get_run_status(run: wandb.Run) -> RunStatus:
    """Gets `RunStatus` for a given wandb `run`."""

    # Work around mypy issues while still maintaining annotations
    run_state = getattr(run, "state", None)
    run_tags = getattr(run, "tags", []) or []

    if run_state == "finished" and "nightly_test_passed" in run_tags:
        # A metrics failure is the more serious signal, so the throughput warning
        # only surfaces on runs that otherwise passed.
        status = RunStatus.PERF_WARNING if PERF_WARNING_TAG in run_tags else RunStatus.SUCCEEDED
    elif run_state == "finished" and "nightly_test_failed" in run_tags:
        status = RunStatus.METRICS_REGRESSION
    elif run_state == "crashed":
        status = RunStatus.CRASHED
    elif run_state == "failed":
        status = RunStatus.FAILED
    else:
        # We don't cover some states (like {running, pending, killed}). These will fall back to UNKNOWN.
        status = RunStatus.UNKNOWN
    return status


def _run_exp_name(run: wandb.Run) -> str:
    """Extracts the experiment name for a run.

    Prefers the experiment tag (set by nightly.py as the sanitized exp name);
    falls back to parsing it out of the run URL for older/untagged runs.
    """
    run_tags = getattr(run, "tags", []) or []
    reserved_prefixes = ("nightly-", "gha-run-id-", "gpus-")
    reserved_exact = KNOWN_SIMULATORS | {
        "singlegpu",
        "multigpu",
        "nightly_test_passed",
        "nightly_test_failed",
        PERF_WARNING_TAG,
    }
    for tag in run_tags:
        if tag.startswith(reserved_prefixes) or tag in reserved_exact:
            continue
        return tag
    # Fallback for untagged runs: URL ends in nightly-<exp>-<simulator>[-multigpu]-<timestamp>.
    try:
        if not run.url or run.url is None:
            return "unknwon"
        return run.url.split("/")[6].split("-")[1]
    except (IndexError, AttributeError):
        return "unknown"


def _fetch_project_runs(
    api: wandb.Api, project_name: str, since_iso: str, filter_tags: list[str] | None = None
) -> list[NightlyRun]:
    """Helper function to fetch runs for a single project.

    Returns a list of NightlyRun records (url, status, exp name, simulator, gpus).
    """
    run_data: list[NightlyRun] = []
    filters: dict[str, dict[str, str | list[str]]] = {
        "created_at": {"$gte": since_iso},
    }

    if filter_tags:
        filters["tags"] = {"$in": filter_tags}

    try:
        runs = api.runs(
            path=f"{WANDB_ENTITY}/{project_name}",
            filters=filters,
            order="-created_at",
        )
        # Determine run status based on run state and test results
        for run in runs:
            run_tags = getattr(run, "tags", []) or []
            run_data.append(
                NightlyRun(
                    url=run.url,
                    status=_get_run_status(run),
                    exp_name=_run_exp_name(run),
                    simulator=_simulator_from_tags(run_tags),
                    gpus=_gpus_from_tags(run_tags),
                )
            )
    except Exception as e:
        print(f"Error fetching runs for project {project_name}: {e}")
    return run_data


def get_latest_report_url() -> str | None:
    """
    Fetches the URL of the most recent nightly report.

    If GITHUB_RUN_ID is set, looks for a report with that run ID in the title.
    Otherwise, returns the most recent report from the nightly-holosoma-runs project.

    Returns:
        Report URL if found, None otherwise
    """
    try:
        api = wandb.Api(timeout=60)
        project_path = f"{WANDB_ENTITY}/nightly-holosoma-runs"

        # Try to get reports from the project
        reports = api.reports(project_path)

        if not reports:
            print(f"Found no reports in {project_path}")
            return None

        print(f"Found {len(reports)} reports in {project_path}")
        print(f"Report names: {[r.display_name for r in reports]}")

        # Search for a report matching the current GitHub run ID
        if GITHUB_RUN_ID:
            for report in reports:
                if GITHUB_RUN_ID in report.display_name:
                    return report.url

        # Fall back to the most recent report
        # Reports are ordered by creation time (newest first)
        for report in reports:
            if "Nightly Training Report" in report.display_name:
                return report.url

    except Exception as e:
        print(f"Error fetching report URL: {e}")

    return None


def get_last_nightly_runs() -> list[NightlyRun]:
    """Fetches all runs in wandb with the tag that have completed runs
    within the last 24 hours.
    """

    api = wandb.Api(timeout=60)

    nightly_runs: list[NightlyRun] = []
    filter_tags = []

    # Fetch all projects for the FAR entity
    all_projects = list(api.projects(WANDB_ENTITY))
    nightly_projects = [project for project in all_projects if project.name.startswith("nightly")]

    # Get runs from the last 24 hours
    since_time = datetime.now(timezone.utc) - timedelta(hours=24)
    since_iso = since_time.isoformat()

    # GHA run ids filter
    if GITHUB_RUN_ID:
        filter_tags.append(f"gha-run-id-{GITHUB_RUN_ID}")

    # Use parallel processing to speed up API calls
    # Default to a reasonable number of workers based on CPU count
    max_workers = min(32, (os.cpu_count() or 1) + 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future2project_name = {
            executor.submit(_fetch_project_runs, api, p.name, since_iso, filter_tags): p.name for p in nightly_projects
        }

        # Process completed tasks as they finish
        for future in as_completed(future2project_name):
            project_name = future2project_name[future]
            try:
                nightly_runs.extend(future.result())
            except Exception as e:
                print(f"Error processing project {project_name}: {e}")

    return nightly_runs


class PeriodStat(NamedTuple):
    """A metric averaged over one period, with the standard error of that average."""

    mean: float
    stderr: float | None  # undefined, and so None, for a single run


def _period_stat(values: list[float]) -> PeriodStat | None:
    """Averages a period's values, or None when the period holds no runs."""
    if not values:
        return None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return PeriodStat(mean=mean, stderr=None)
    return PeriodStat(mean=mean, stderr=statistics.stdev(values) / math.sqrt(len(values)))


def _format_stat(stat: PeriodStat | None) -> str:
    """Renders a period as "3.358s ±0.006s", dropping the error where it is undefined."""
    if stat is None:
        return "n/a"
    if stat.stderr is None:
        return f"{stat.mean:.3f}{TREND_UNIT}"
    return f"{stat.mean:.3f}{TREND_UNIT} ±{stat.stderr:.3f}{TREND_UNIT}"


def _difference_error(earlier: PeriodStat | None, later: PeriodStat | None) -> float | None:
    """Standard error of the difference between two period means.

    Combining both sides' errors in quadrature is the denominator of Welch's t: dividing
    the gap between the means by it asks how many standard deviations apart the two
    distributions are, given how well each mean is known. None when either side is a
    single run and so carries no error of its own.
    """
    if earlier is None or later is None or earlier.stderr is None or later.stderr is None:
        return None
    return math.hypot(earlier.stderr, later.stderr)


def _trend_marker(later: PeriodStat | None, earlier: PeriodStat | None, sigma: float | None) -> str:
    """Colour for the verdict, diagonal arrow for the direction.

    A gap smaller than `TREND_SIGMA` times `sigma` is reported as flat, so the colour
    only calls out a change large enough to be distinguishable from noise.
    """
    if later is None or earlier is None:
        return "❔"

    delta = later.mean - earlier.mean
    if sigma is None or sigma <= 0 or abs(delta) <= TREND_SIGMA * sigma:
        return f"{TREND_FLAT}➡️"

    improved = delta < 0 if TREND_LOWER_IS_BETTER else delta > 0
    arrow = "↘️" if delta < 0 else "↗️"
    return f"{TREND_BETTER if improved else TREND_WORSE}{arrow}"


def _trend_view_url(cell: str) -> str:
    """URL of the saved workspace view scoped to one cell, or "" if nothing names it.

    The variable name is the prefix followed by the cell label upper-cased, with every run
    of other characters replaced by an underscore -- derived from the label so that
    tracking another cell needs no second table to be kept in step with it. The view it
    should point at is the one `perf_workspace.py --cell <cell>` creates.
    """
    variable = f"{TREND_VIEW_URL_ENV_PREFIX}_{re.sub('[^A-Z0-9]+', '_', cell.upper()).strip('_')}"
    url = getenv(variable, "")
    if not url:
        print(f"{variable} is unset; linking the whole project rather than a view of {cell!r}")
    return url


def _fetch_trend_series(cell: str) -> tuple[list[tuple[datetime, float]], str | None]:
    """The tracked metric for one cell newest first, with a link to its project.

    The link is taken from a run's own URL so it addresses whichever wandb instance the
    job is pointed at, rather than naming one.
    """
    api = wandb.Api(timeout=60)
    since = (datetime.now(timezone.utc) - timedelta(days=TREND_LOOKBACK_DAYS)).isoformat()

    runs = api.runs(
        path=f"{WANDB_ENTITY}/{PERF_PROJECT}",
        filters={
            f"summary_metrics.{PERF_CELL_KEY}": {"$eq": cell},
            "created_at": {"$gte": since},
        },
        order="-created_at",
    )

    series: list[tuple[datetime, float]] = []
    project_url: str | None = None
    for run in runs:
        value = (run.summary or {}).get(TREND_METRIC_KEY)
        created = getattr(run, "created_at", None)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not created:
            continue
        series.append((datetime.fromisoformat(str(created).replace("Z", "+00:00")), float(value)))
        run_url = getattr(run, "url", None)
        if project_url is None and run_url and "/runs/" in run_url:
            project_url = run_url.rsplit("/runs/", 1)[0]

    series.sort(key=lambda entry: entry[0], reverse=True)
    return series, project_url


def _format_trend_block(cell: str) -> str | None:
    """One cell's figures across three timescales, or None if it has too little history.

    Returns None rather than raising: this is a nicety on top of the run summary and
    must never be the reason the summary fails to post.
    """
    try:
        series, project_url = _fetch_trend_series(cell)
        if len(series) < 2:
            print(f"Not enough history for {cell!r} to build a trend block")
            return None

        now = datetime.now(timezone.utc)

        def window(from_days_ago: int, to_days_ago: int) -> PeriodStat | None:
            start = now - timedelta(days=from_days_ago)
            end = now - timedelta(days=to_days_ago)
            return _period_stat([value for stamp, value in series if start <= stamp < end])

        # A single night carries no error of its own, so the scale for the 1d comparison
        # comes from how much this cell normally varies from night to night. Two single
        # observations are being differenced, so their spreads combine in quadrature the
        # same way the segment errors do -- giving sd * sqrt(2), not sd.
        recent = [value for _, value in series[:TREND_DAILY_BASELINE_RUNS]]
        nightly_sd = statistics.stdev(recent) if len(recent) > 1 else None
        nightly_spread = math.hypot(nightly_sd, nightly_sd) if nightly_sd else None

        # Each comparison reads earlier to later, so the arrow between the two figures
        # is the direction time runs and the trailing marker is the verdict.
        # The 1d pair is the two newest runs rather than calendar days, so a skipped
        # night compares against the previous run that exists instead of reporting a gap.
        segments = [(f"{span}d", window(span * 2, span), window(span, 0)) for span in (7, 30)]
        comparisons = [
            ("1d", _period_stat([series[1][1]]), _period_stat([series[0][1]]), nightly_spread),
            *((label, before, after, _difference_error(before, after)) for label, before, after in segments),
        ]

        # Fenced and column-aligned: Slack renders ordinary text proportionally, so the
        # padding that lines the arrows up only holds inside a code block.
        rendered = [
            (_trend_marker(after, before, sigma), label, _format_stat(before), _format_stat(after))
            for label, before, after, sigma in comparisons
        ]
        label_width = max(len(label) for _, label, _, _ in rendered)
        before_width = max(len(before) for _, _, before, _ in rendered)

        lines = [
            f"⏱️ {TREND_METRIC_LABEL} · {cell}",
            "```",
            *(
                f"{marker} {label:>{label_width}}: {before:<{before_width}} → {after}"
                for marker, label, before, after in rendered
            ),
            "```",
        ]
        link = _trend_view_url(cell) or project_url
        if link:
            lines.append(f"📈 [{TREND_VIEW_LABEL}]({link})")
        return "\n".join(lines)
    except Exception as e:
        print(f"Could not build the trend block for {cell!r}: {e}")
        return None


def format_trend_report() -> str | None:
    """A trend block per tracked cell, or None when none could be built.

    Cells are reported separately because the metric is not comparable across them.
    """
    blocks = [block for cell in TREND_CELLS if (block := _format_trend_block(cell))]
    return "\n\n".join(blocks) if blocks else None


def format_run_line(run: NightlyRun) -> str:
    """Formats a single run as an emoji-prefixed WandB link line for Slack/CLI."""
    return f"{RunStatus2Emoji.get(run.status)} {run.url} ({run.simulator}, {run.gpus})"


def post_summary_to_slack():
    """Posts summary of runs to slack channel."""
    # Local: slack-sdk is only in this script's own dependencies, so importing it at module
    # scope would make the module unimportable anywhere else.
    from slack_sdk import WebClient

    if not SLACK_TOKEN or not SLACK_CHANNEL:
        raise ValueError("SLACK_BOT_TOKEN or SLACK_CHANNEL env var not set, can't post message to slack")
    slack_client = WebClient(token=SLACK_TOKEN)

    summary_message = "*Nightly Build Completed!*\n"

    # Get the report URL
    report_url = get_latest_report_url()

    # Add report link at the top if available
    if report_url:
        summary_message += f"📊 [Wandb Report]({report_url})\n"

    run_url = f"{GITHUB_SERVER_URL}/{GITHUB_REPOSITORY}/actions/runs/{GITHUB_RUN_ID}"
    summary_message += f"Build Link: [Githun Run {GITHUB_RUN_ID}]({run_url})"

    trend_report = format_trend_report()
    if trend_report:
        summary_message += f"\n{trend_report}"

    nightly_runs = get_last_nightly_runs()

    # Each failed run is labelled with its simulator + GPU count so the same
    # experiment failing on different matrix dimensions (single- vs multi-GPU,
    # isaacgym vs isaacsim) shows up as distinct entries instead of duplicates.
    problem_runs = [run.failed_run_label() for run in nightly_runs if run.status not in HEALTHY_STATUSES]

    if problem_runs:
        summary_message += "\nFailed runs: " + ", ".join(problem_runs)

    wandb_summaries = [format_run_line(run) for run in nightly_runs]
    summary_message += "\nWandB Links:\n```\n" + "\n".join(wandb_summaries) + "\n```"
    summary_message += "\n" + " ".join(
        f"{emoji} = {status.name.replace('_', ' ').title()}" for status, emoji in RunStatus2Emoji.items()
    )

    slack_client.chat_postMessage(channel=SLACK_CHANNEL, markdown_text=summary_message, unfurl_links=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_summary", description="Summarizes latest nightly runs")

    parser.add_argument("--slack", action="store_true")
    parser.add_argument("--trend-only", action="store_true", help="Print just the performance trend line")

    args = parser.parse_args()

    if args.slack:
        post_summary_to_slack()
    elif args.trend_only:
        print(format_trend_report() or "no trend report available")
    else:
        print("\n")  # To make the message layout cleaner
        print(format_trend_report() or "no trend report available")
        print("\n".join(format_run_line(run) for run in get_last_nightly_runs()))
