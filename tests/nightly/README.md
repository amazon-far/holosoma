# Nightly Training Reports

This directory contains scripts for running and analyzing nightly training runs.

## Scripts

### `nightly.py`
Main script that runs nightly training experiments. It:
- Runs training for a specified experiment configuration
- Validates that metrics fall within expected ranges
- Tags runs as passed/failed based on metric validation

Usage:
```bash
python tests/nightly/nightly.py exp:g1-29dof logger:wandb simulator:isaacgym
```

### `run_summary.py`
Generates a summary of recent nightly runs and posts to Slack. Automatically includes a link to the most recent wandb report at the top of the message.

Usage:
```bash
# Print summary to console
uv run --script ./tests/nightly/run_summary.py

# Post to Slack (includes report link if available)
uv run --script ./tests/nightly/run_summary.py --slack
```

### `generate_report.py`
Generates a comprehensive wandb report with visualizations of nightly training runs.

The report includes:
- **Cross-Experiment Comparison**: Compare total rewards, episode lengths, and learning rates across all experiments
- **Per-Experiment Sections**: Detailed analysis for each experiment type with organized metric groups:
  - **Locomotion experiments** (g1-29dof, t1-29dof):
    - Velocity Tracking: linear and angular velocity tracking rewards
    - Total Reward: overall performance
    - Additional Rewards: action rate, stand still, feet air time
    - Training Metrics: policy loss, value loss, learning rate
    - Episode Statistics: episode length, terrain level

  - **Whole Body Tracking experiments** (g1-29dof-wbt):
    - Position Tracking: global and relative body position errors
    - Orientation Tracking: global and relative body orientation errors
    - Velocity Tracking: linear and angular velocity rewards
    - Total Reward: overall performance
    - Training Metrics: policy loss, value loss, learning rate
    - Episode Statistics: episode length, terrain level

Usage:
```bash
# Generate report for last 24 hours of runs
uv run --script ./tests/nightly/generate_report.py

# Generate report for last 48 hours
uv run --script ./tests/nightly/generate_report.py --hours 48

# Filter by GitHub Actions run ID
uv run --script ./tests/nightly/generate_report.py --github-run-id=123456789

# Make report public (requires manual visibility setting in wandb UI)
uv run --script ./tests/nightly/generate_report.py --publish
```

Options:
- `--hours`: Number of hours to look back for runs (default: 24)
- `--github-run-id`: Filter runs by specific GitHub Actions run ID
- `--publish`: Flag to indicate the report should be made public (note: visibility must be set manually in wandb UI)

The script automatically:
- Groups runs by experiment type (g1-29dof, t1-29dof, g1-29dof-wbt, etc.)
- Creates organized metric visualizations based on experiment type
- Generates line plots for all tracked metrics
- Provides cross-experiment comparisons when multiple experiments are present

## GitHub Actions Integration

The nightly workflow automatically:
1. Runs training for multiple experiments across different configurations (simulator, GPU setup)
2. Generates a wandb report with all results
3. Posts a summary to Slack with a link to the report at the top

## Environment Variables

All scripts support the following environment variables:
- `WANDB_ENTITY`: wandb entity/org name (default: "amazon-far")
- `WANDB_API_KEY`: wandb API key for authentication
- `GITHUB_RUN_ID`: GitHub Actions run ID (automatically set in CI)
- `SLACK_BOT_TOKEN`: Slack bot token for posting messages
- `SLACK_CHANNEL`: Slack channel to post to

Reports are created in the `nightly-holosoma-runs` project on wandb.

## Report Structure

Generated reports include:
1. **Overview**: Summary of all runs analyzed, with specific run names listed
2. **Cross-Experiment Comparison** (if multiple experiments): Side-by-side comparison of key metrics
3. **Individual Experiment Sections**: Detailed analysis with metric-specific charts grouped by category:
   - Tracking performance (velocity, position, orientation)
   - Training dynamics (losses, learning rate)
   - Episode statistics

All charts are organized in grid layouts for easy comparison within each metric category.

**Note**: Due to limitations in the wandb-workspaces API, visualizations currently show all runs in the project. Each section lists the specific run names being analyzed - you can use these names to manually filter the charts in the wandb UI to focus on specific nightly runs.
