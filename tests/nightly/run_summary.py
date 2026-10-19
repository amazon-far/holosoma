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
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from os import getenv
from typing import NamedTuple

import wandb
from slack_sdk import WebClient

WANDB_ENTITY = getenv("WANDB_ENTITY", "amazon-far")
SLACK_CHANNEL = getenv("SLACK_CHANNEL", "")
SLACK_TOKEN = getenv("SLACK_BOT_TOKEN", "")

# Github assigned variables
GITHUB_SERVER_URL = getenv("GITHUB_SERVER_URL")
GITHUB_REPOSITORY = getenv("GITHUB_REPOSITORY")
GITHUB_RUN_ID = getenv("GITHUB_RUN_ID")

# Throughput regression warning applied by nightly.py. Keep in sync with that script.
PERF_WARNING_TAG = "nightly_perf_warning"

# One-line performance trend included in the Slack summary. The cell label and the metric
# key are written by backfill_perf_summary.py; keep in sync with that script.
PERF_PROJECT = "nightly-holosoma-runs"
PERF_CELL_KEY = "perf/cell"
TREND_CELL = "g1_29dof_manager | isaacsim | gpus-1"
TREND_METRIC_KEY = "perf/collection_time_mean"
TREND_METRIC_LABEL = "collection_time mean"
# Enough history to cover the longest comparison window and its predecessor.
TREND_LOOKBACK_DAYS = 60
# Relative change below which a comparison reads as flat. Without it, ordinary
# run-to-run noise shows as a direction every night.
TREND_DEADBAND = 0.01

# Simulator tag values set by nightly.py (config.simulator.config.name). "isaacgym" is retained
# for ARCHIVED runs only -- the backend is retired; dropping it would make historical runs resolve
# their exp_name to the literal tag instead of falling back to "unknown-sim".
KNOWN_SIMULATORS = {"isaacgym", "isaacsim", "mujoco"}
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
    # Fallback: URL looks like .../runs/nightly-<exp>-[multigpu-]<timestamp>
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
    """Renders a period as "3.358 ±0.006s", dropping the error where it is undefined."""
    if stat is None:
        return "n/a"
    if stat.stderr is None:
        return f"{stat.mean:.3f}s"
    return f"{stat.mean:.3f} ±{stat.stderr:.3f}s"


def _trend_emoji(current: PeriodStat | None, previous: PeriodStat | None) -> str:
    """Direction of travel between two periods, or flat inside the deadband.

    Up means the metric grew. For a timing metric that is slower, not better.
    """
    if current is None or previous is None or not previous.mean:
        return "❔"
    change = (current.mean - previous.mean) / abs(previous.mean)
    if abs(change) < TREND_DEADBAND:
        return "➡️"
    return "⬆️" if change > 0 else "⬇️"


def _fetch_trend_series() -> list[tuple[datetime, float]]:
    """The tracked metric for the tracked cell, newest first."""
    api = wandb.Api(timeout=60)
    since = (datetime.now(timezone.utc) - timedelta(days=TREND_LOOKBACK_DAYS)).isoformat()

    runs = api.runs(
        path=f"{WANDB_ENTITY}/{PERF_PROJECT}",
        filters={
            f"summary_metrics.{PERF_CELL_KEY}": {"$eq": TREND_CELL},
            "created_at": {"$gte": since},
        },
        order="-created_at",
    )

    series: list[tuple[datetime, float]] = []
    for run in runs:
        value = (run.summary or {}).get(TREND_METRIC_KEY)
        created = getattr(run, "created_at", None)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not created:
            continue
        series.append((datetime.fromisoformat(str(created).replace("Z", "+00:00")), float(value)))

    series.sort(key=lambda entry: entry[0], reverse=True)
    return series


def format_trend_line() -> str | None:
    """One line comparing the tracked metric across three timescales.

    Returns None rather than raising: this is a nicety on top of the run summary and
    must never be the reason the summary fails to post.
    """
    try:
        series = _fetch_trend_series()
        if len(series) < 2:
            print(f"Not enough history for {TREND_CELL!r} to build a trend line")
            return None

        now = datetime.now(timezone.utc)

        def window(from_days_ago: int, to_days_ago: int) -> PeriodStat | None:
            start = now - timedelta(days=from_days_ago)
            end = now - timedelta(days=to_days_ago)
            return _period_stat([value for stamp, value in series if start <= stamp < end])

        # The two newest runs rather than calendar days, so a skipped night compares
        # against the previous run that exists instead of reporting a gap.
        comparisons = [("today", _period_stat([series[0][1]]), _period_stat([series[1][1]]))]
        for label, span in (("7d", 7), ("30d", 30)):
            comparisons.append((label, window(span, 0), window(span * 2, span)))

        parts = [
            f"{label} {_format_stat(current)} vs {_format_stat(previous)} {_trend_emoji(current, previous)}"
            for label, current, previous in comparisons
        ]
        return f"⏱️ {TREND_METRIC_LABEL} · {TREND_CELL} — " + " · ".join(parts)
    except Exception as e:
        print(f"Could not build the performance trend line: {e}")
        return None


def format_run_line(run: NightlyRun) -> str:
    """Formats a single run as an emoji-prefixed WandB link line for Slack/CLI."""
    return f"{RunStatus2Emoji.get(run.status)} {run.url} ({run.simulator}, {run.gpus})"


def post_summary_to_slack():
    """Posts summary of runs to slack channel."""
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

    trend_line = format_trend_line()
    if trend_line:
        summary_message += f"\n{trend_line}"

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
        print(format_trend_line() or "no trend line available")
    else:
        print("\n")  # To make the message layout cleaner
        print(format_trend_line() or "no trend line available")
        print("\n".join(format_run_line(run) for run in get_last_nightly_runs()))
