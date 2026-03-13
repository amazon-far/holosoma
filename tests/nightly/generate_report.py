#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "wandb",
#   "wandb-workspaces",
# ]
# ///

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from os import getenv

import wandb
import wandb_workspaces.reports.v2 as wr

WANDB_ENTITY = getenv("WANDB_ENTITY", "amazon-far")
WANDB_PROJECT = "nightly-holosoma-runs"

# Github assigned variables
GITHUB_SERVER_URL = getenv("GITHUB_SERVER_URL")
GITHUB_REPOSITORY = getenv("GITHUB_REPOSITORY")
GITHUB_RUN_ID = getenv("GITHUB_RUN_ID")


def get_filters(
    gh_run_id: str | None = None, extra_tags: list[str] | None = None, expr: bool = False
) -> dict[str, dict[str, list[str]]] | str:
    """Filters for runs.

    Args:
        gh_run_id: Optional GitHub run ID to filter by
        expr: whether to return in wandb api format or workspace format (FilterExpr)
        extra_tags: list of extra tags to filter by
    Returns
        Filters specified
    """
    # FilterExpr Tags are bugged
    # filters = [ws.Tags().isin([f"gha-run-id-{gh_run_id}"])]

    tags = extra_tags if extra_tags else []
    if gh_run_id:
        tags.append(f"gha-run-id-{gh_run_id}")

    return (
        f"Tags() in {tags}"
        if expr
        else {
            "tags": {"$in": tags},
        }
    )


def get_nightly_run_ids(gh_run_id: str | None = None) -> dict[str, list[str]]:
    """
    Fetches run IDs from nightly runs grouped by experiment type.

    Args:
        hours: Number of hours to look back for runs
        gh_run_id: Optional GitHub run ID to filter by

    Returns:
        Dictionary mapping experiment names to lists of run IDs
    """
    api = wandb.Api(timeout=60)

    filters = get_filters(gh_run_id)
    assert isinstance(filters, dict)

    # Fetch all runs from the nightly-runs project
    runs = api.runs(
        path=f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters=filters,
        order="-created_at",
    )

    # Group runs by experiment type
    grouped_runs: dict[str, list[str]] = {}

    for run in runs:
        # Extract experiment name from tags
        # Expected tags include experiment name (e.g., "g1-29dof"), simulator, gpu config
        exp_name = None
        for tag in run.tags:
            if tag.startswith("nightly-"):
                continue
            if tag in ["isaacgym", "isaacsim", "singlegpu", "multigpu", "nightly_test_passed", "nightly_test_failed"]:
                continue
            if tag.startswith("gha-run-id-"):
                continue
            exp_name = tag
            break

        if exp_name:
            if exp_name not in grouped_runs:
                grouped_runs[exp_name] = []
            grouped_runs[exp_name].append(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run.id}")

    return grouped_runs


def create_metric_panel(metric_name: str, title: str | None = None) -> wr.LinePlot:
    """Create a line plot panel for a specific metric across multiple runs."""
    return wr.LinePlot(
        x="global_step",
        y=[metric_name],
        title=title or metric_name,
        range_x=(0, None),
    )


def create_experiment_section(exp_name: str, run_ids: list[str], gh_run_id: str) -> list:
    """
    Create a report section for a specific experiment with organized metric groups.

    Args:
        exp_name: Name of the experiment (e.g., "g1-29dof")
        run_ids: List of wandb run IDs for this experiment

    Returns:
        List of report blocks for this experiment
    """
    blocks = [
        wr.H1(f"Experiment: {exp_name}"),
        wr.P(f"Analysis of {len(run_ids)} nightly runs for {exp_name}"),
    ]

    # Define metric groups based on experiment type
    if "wbt" in exp_name:
        # Whole Body Tracking metrics
        metric_groups = {
            "Position Tracking": [
                "Episode/rew_motion_global_ref_position_error_exp",
                "Episode/rew_motion_relative_body_position_error_exp",
            ],
            "Orientation Tracking": [
                "Episode/rew_motion_global_ref_orientation_error_exp",
                "Episode/rew_motion_relative_body_orientation_error_exp",
            ],
            "Velocity Tracking": [
                "Episode/rew_motion_global_body_lin_vel",
                "Episode/rew_motion_global_body_ang_vel",
            ],
            "Total Reward": [
                "Episode/rew_total",
            ],
        }
    else:
        # Locomotion metrics
        metric_groups = {
            "Velocity Tracking": [
                "Episode/rew_tracking_lin_vel",
                "Episode/rew_tracking_ang_vel",
            ],
            "Total Reward": [
                "Episode/rew_alive",
            ],
            "Additional Rewards": [
                "Episode/rew_penalty_orientation",
                "Episode/rew_pose",
                "Episode/rew_feet_phase",
                "Episode/rew_penalty_action_rate",
            ],
        }

    # Add common training metrics to all experiments
    metric_groups["Training Metrics"] = [
        # "Loss/Value",
        # "Loss/Entropy",
        # "Loss/Surrogate",
        "Train/mean_reward",
        # "Loss/critic_learning_rate",
        # "Loss/actor_learning_rate",
        "Loss/actor_loss",
        # "Loss/critic_loss",
    ]

    metric_groups["Episode Statistics"] = [
        "Env/average_episode_length",
        # "Episode/terrain_level",
    ]

    filters = get_filters(gh_run_id, extra_tags=[exp_name], expr=True)
    assert isinstance(filters, str)
    runset = wr.Runset(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        filters=filters,
    )

    # Create panels for each metric group
    for group_name, metrics in metric_groups.items():
        blocks.append(wr.H2(group_name))

        # Add a paragraph listing the specific runs being analyzed
        run_names = [run_id.split("/")[-1] for run_id in run_ids]
        blocks.append(wr.P(f"Analyzing {len(run_names)} run(s): {', '.join(run_names)}"))

        panels = [create_metric_panel(metric, title=metric.split("/")[-1]) for metric in metrics]

        blocks.append(wr.PanelGrid(panels=panels, runsets=[runset]))  # type: ignore[arg-type]

    return blocks


def create_comparison_section(all_run_ids: dict[str, list[str]], gh_run_id: str) -> list:
    """
    Create a comparison section showing performance across all experiments.

    Args:
        all_run_ids: Dictionary mapping experiment names to run ID lists

    Returns:
        List of report blocks for the comparison section
    """
    blocks = [
        wr.H1("Cross-Experiment Comparison"),
        wr.P("Comparing total rewards and key metrics across all experiments"),
    ]

    # Create comparison charts
    comparison_metrics = [
        ("Episode/rew_alive", "Alive Reward Comparison"),
        # ("Env/average_episode_length", "Episode Length Comparison"),
        ("Train/mean_episode_length", "Episode Length Comparison"),
        ("Loss/actor_learning_rate", "Learning Rate Comparison"),
    ]

    panels = []
    for metric, title in comparison_metrics:
        panels.append(
            wr.LinePlot(
                x="global_step",
                y=[metric],
                title=title,
            )
        )

    # Collect all run names for display
    all_run_names = []
    for run_list in all_run_ids.values():
        all_run_names.extend([run_id.split("/")[-1] for run_id in run_list])

    # Add a paragraph listing the specific runs being compared
    blocks.append(wr.P(f"Comparing {len(all_run_names)} run(s) across experiments: {', '.join(all_run_names)}"))

    filters = get_filters(gh_run_id, expr=True)
    assert isinstance(filters, str)
    runset = wr.Runset(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        filters=filters,
    )

    blocks.append(wr.PanelGrid(panels=panels, runsets=[runset]))  # type: ignore[arg-type]

    return blocks


def create_nightly_report(gh_run_id: str, hours: int = 24, publish: bool = False) -> str:
    """
    Create a comprehensive wandb report for nightly runs.

    Args:
        hours: Number of hours to look back for runs
        gh_run_id: Optional GitHub run ID to filter by
        publish: Whether to make the report public

    Returns:
        URL of the created report
    """
    print(f"Fetching nightly runs from the github action run {gh_run_id}...")
    grouped_runs = get_nightly_run_ids(gh_run_id=gh_run_id)

    if not grouped_runs:
        print("No nightly runs found!")
        return ""

    print(f"Found runs for {len(grouped_runs)} experiments:")
    for exp_name, run_ids in grouped_runs.items():
        print(f"  - {exp_name}: {len(run_ids)} runs")

    num_runs = sum(len(runs) for runs in grouped_runs.values())
    num_exps = len(grouped_runs)
    # Create report blocks
    report_blocks = [
        wr.H1("Nightly Training Report"),
        wr.P(f"Automated report generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"),
        wr.P(f"This report contains analysis of {num_runs} nightly training runs across {num_exps} experiments."),
    ]

    # Add comparison section first
    if len(grouped_runs) > 1:
        report_blocks.extend(create_comparison_section(grouped_runs, gh_run_id))

    # Add individual experiment sections
    for exp_name in sorted(grouped_runs.keys()):
        run_ids = grouped_runs[exp_name]
        report_blocks.extend(create_experiment_section(exp_name, run_ids, gh_run_id))

    # Add footer
    report_blocks.append(wr.P("---"))
    report_blocks.append(
        wr.P(
            "This report was automatically generated by the nightly training pipeline. "
            "The visualizations show all runs in the project - use the run names listed in each section "
            "to manually filter to the specific nightly runs in the wandb UI if needed."
        )
    )

    # Create the report
    print("\nCreating report...")

    title_suffix = f" (GHA Run {gh_run_id})" if gh_run_id else ""
    report_title = f"Nightly Training Report - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}{title_suffix}"

    report = wr.Report(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        title=report_title,
        description=f"Automated analysis of nightly training runs from the past {hours} hours",
        blocks=report_blocks,  # type: ignore[arg-type]
    )

    report.save()
    report_url = report.url

    if publish:
        print("Making report public...")
        # Note: wandb-workspaces doesn't have a direct publish method in v2
        # The report is created with default visibility settings
        # To make it public, you may need to use the wandb API directly or do it manually
        print("Note: Please set the report visibility to 'public' manually in the wandb UI")

    print("\nReport created successfully!")
    print(f"URL: {report_url}")

    return report_url


def main():
    parser = argparse.ArgumentParser(description="Generate a wandb report for nightly training runs")
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Number of hours to look back for runs (default: 24)",
    )
    parser.add_argument(
        "--github-run-id",
        type=str,
        default=None,
        help="Filter runs by GitHub Actions run ID",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Make the report public (note: may require manual action)",
    )

    args = parser.parse_args()

    # Use environment variable if available
    gh_run_id = args.github_run_id or GITHUB_RUN_ID

    create_nightly_report(
        gh_run_id=gh_run_id,
        hours=args.hours,
        publish=args.publish,
    )


if __name__ == "__main__":
    main()
