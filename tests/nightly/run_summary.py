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

import wandb
from slack_sdk import WebClient

WANDB_ENTITY = getenv("WANDB_ENTITY", "far-wandb")
SLACK_CHANNEL = getenv("SLACK_CHANNEL", "")
SLACK_TOKEN = getenv("SLACK_BOT_TOKEN", "")


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


def _fetch_project_runs(
    api: wandb.Api, project_name: str, since_iso: str, filter_tags: list[str] | None = None
) -> list[tuple[str, RunStatus]]:
    """Helper function to fetch runs for a single project.

    Returns a list of tuples containing (url, run_status)
    """
    run_data: list[tuple[str, RunStatus]] = []  # (url, run_status)
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
        run_data.extend((run.url, _get_run_status(run)) for run in runs)
    except Exception as e:
        print(f"Error fetching runs for project {project_name}: {e}")
    return run_data


def get_last_nightly_urls() -> list[str]:
    """Fetches the url of all runs in wandb with the tag that have completed runs
    within the last 12 hours.
    """

    api = wandb.Api(timeout=60)

    nightly_urls = []
    filter_tags = []

    # Fetch all projects for the FAR entity
    all_projects = list(api.projects(WANDB_ENTITY))
    nightly_projects = [project for project in all_projects if project.name.startswith("nightly")]

    # Get runs from the last 12 hours
    since_time = datetime.now(timezone.utc) - timedelta(hours=16)
    since_iso = since_time.isoformat()

    # GHA run ids filter
    if os.getenv("GITHUB_RUN_ID"):
        filter_tags.append(f"gha-run-id-{os.getenv('GITHUB_RUN_ID')}")

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
                run_data = future.result()
                for url, status in run_data:
                    nightly_urls.append(f"{RunStatus2Emoji.get(status)} {url}")
            except Exception as e:
                nightly_urls.append(f"Error processing project {project_name}: {e}")

    return nightly_urls


def post_summary_to_slack():
    """Posts summary of runs to slack channel."""
    if not SLACK_TOKEN or not SLACK_CHANNEL:
        raise ValueError("SLACK_BOT_TOKEN or SLACK_CHANNEL env var not set, can't post message to slack")
    slack_client = WebClient(token=SLACK_TOKEN)

    run_url = f"{getenv('GITHUB_SERVER_URL')}/{getenv('GITHUB_REPOSITORY')}/actions/runs/{getenv('GITHUB_RUN_ID')}"
    summary_message = f"*Nightly Build Completed!*\nBuild URL: {run_url}"
    wandb_summaries = get_last_nightly_urls()
    problem_runs = []
    for wandb_result in wandb_summaries:
        if wandb_result[0] != "🟩":
            exp_name = wandb_result.split("/")[4]
            problem_runs.append(exp_name)

    if problem_runs:
        summary_message += "\nFailed runs: " + ", ".join(problem_runs)

    summary_message += "\nWandB URLs:\n```\n" + "\n".join(wandb_summaries) + "\n```"

    slack_client.chat_postMessage(channel=SLACK_CHANNEL, text=summary_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_summary", description="Summarizes latest nightly runs")

    parser.add_argument("--slack", action="store_true")

    args = parser.parse_args()

    if args.slack:
        post_summary_to_slack()
    else:
        print("\n")  # To make the message layout cleaner
        print("\n".join(get_last_nightly_urls()))
