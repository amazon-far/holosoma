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
import os
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

# Simulator tag values set by nightly.py (config.simulator.config.name).
KNOWN_SIMULATORS = {"isaacgym", "isaacsim", "mujoco"}
# GPU count for a multi-GPU nightly run; used only as a fallback for legacy runs
# tagged with "multigpu" but no explicit "gpus-N" tag. Keep in sync with nightly.py.
MULTIGPU_NUM_GPUS = 4


class RunStatus(Enum):
    """Enum representing the possible states of a nightly test run."""

    SUCCEEDED = auto()
    CRASHED = auto()
    METRICS_REGRESSION = auto()
    FAILED = auto()
    UNKNOWN = auto()


# Mapping to help visually determine what happened via a slack message.
RunStatus2Emoji = {
    RunStatus.SUCCEEDED: "🟩",
    RunStatus.CRASHED: "🟥",
    RunStatus.METRICS_REGRESSION: "⚠️",
    RunStatus.FAILED: "🚫",
    RunStatus.UNKNOWN: "❓",
}


class NightlyRun(NamedTuple):
    """A single nightly wandb run, enriched with the details we surface in Slack."""

    url: str
    status: RunStatus
    exp_name: str
    simulator: str  # e.g. "isaacgym"/"isaacsim", or "unknown-sim" if untagged
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
        status = RunStatus.SUCCEEDED
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
    reserved_exact = KNOWN_SIMULATORS | {"singlegpu", "multigpu", "nightly_test_passed", "nightly_test_failed"}
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

    nightly_runs = get_last_nightly_runs()

    # Each failed run is labelled with its simulator + GPU count so the same
    # experiment failing on different matrix dimensions (single- vs multi-GPU,
    # isaacgym vs isaacsim) shows up as distinct entries instead of duplicates.
    problem_runs = [run.failed_run_label() for run in nightly_runs if run.status != RunStatus.SUCCEEDED]

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

    args = parser.parse_args()

    if args.slack:
        post_summary_to_slack()
    else:
        print("\n")  # To make the message layout cleaner
        print("\n".join(format_run_line(run) for run in get_last_nightly_runs()))
