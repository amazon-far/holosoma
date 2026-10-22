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
python tests/nightly/nightly.py exp:g1-29dof logger:wandb simulator:isaacsim
```

### `run_summary.py`
Generates a summary of recent nightly runs and posts to Slack. Automatically includes a link to the most recent wandb report at the top of the message.

Usage:
The message also carries a performance trend for a single tracked configuration
(`TREND_CELL`): a heading naming the metric and configuration, then one bullet per
timescale comparing the mean of `TREND_METRIC_KEY`, each rendered as
`<window>: <earlier> → <later>`:

- `1d` — the two most recent runs
- `7d` — the last 7 days against the 7 before
- `30d` — the last 30 days against the 30 before

The 7d and 30d figures include the standard error of their mean; the `1d` pair cannot,
having one run each side. The earlier figure is always on the left, so the `→` is the
direction time runs.

The trailing marker pairs a verdict colour with a diagonal arrow for the direction: 🟢
better, 🔴 worse, ⚪ flat. Which direction counts as better comes from
`TREND_LOWER_IS_BETTER`, which must be flipped if `TREND_METRIC_KEY` is pointed at a
metric like throughput where higher is better. The colours are circles so they don't read
as the squares in the run-status legend lower down the same message.

A change is only coloured when the two figures are more than `TREND_SIGMA` standard
deviations apart, so ordinary noise reads as flat:

- For `7d` and `30d`, the spread used is that of the *difference* between the two means,
  combining both sides' standard errors in quadrature. That is the denominator of
  Welch's t, so the test asks how many standard deviations apart the two distributions
  are given how well each mean is known — rather than comparing against one side alone.
- For `1d` there is one run each side and so no error to use. The two runs supply the
  change and the scale comes from elsewhere: the standard deviation of the last
  `TREND_DAILY_BASELINE_RUNS` runs, which is how much the cell normally wanders from one
  night to the next. Since two single observations are being differenced, that deviation
  is combined in quadrature with itself — `sd * sqrt(2)` — so the row is testing the same
  quantity as the segment rows rather than a figure 1.4x stricter.

The `1d` pair is the two most recent runs rather than calendar days, so a skipped night
compares against the previous run that exists instead of reporting a gap.

Under the figures is a link to a saved workspace view of the same configuration, taken
from `PERF_TREND_VIEW_URL_<CELL>` (see [Environment Variables](#environment-variables)).
With that variable unset the link falls back to the whole project, which shows every
configuration at once.

```bash
# Print summary to console
uv run --script ./tests/nightly/run_summary.py

# Print just the trend line
uv run --script ./tests/nightly/run_summary.py --trend-only

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

### `backfill_perf_summary.py`
Writes per-run performance aggregates into each run's wandb summary.

The `Perf/*` metrics are logged once per training iteration, so they only exist as per-run
time series. wandb's cross-run panels read run summary and config rather than history, so
plotting throughput across nights needs one value per run at the run level. This script
derives those values from a run's full, unsampled history and writes the following.
Iteration zero is excluded from every statistic: it pays for JIT compilation and CUDA
graph capture, and is slow enough to otherwise become the reported fps minimum and
timing maximum for the entire run.

- `perf/total_fps_{min,mean,max}`
- `perf/collection_time_{min,mean,max}`
- `perf/learning_time_{min,mean,max}`
- `perf/run_date` — run start as a Unix timestamp, used as the x-axis of the trend panels
- `perf/run_duration_s` — total wall-clock duration
- `perf/backfill_version` — marks the run as processed

Runs already marked at the current version are skipped, so it is safe to re-run. It runs
after each nightly build and can be used once to backfill history. Runs that logged no
`Perf/*` history (a crash, say) are still marked, so they aren't rescanned on every pass;
bump `BACKFILL_VERSION` in the script to reprocess everything after changing what it writes.

By default it processes `nightly-holosoma-runs`, the project the nightly build writes to.
The older per-experiment `nightly-<exp>` projects no longer receive runs and are only
included with `--all-projects`.

```bash
# Preview without writing
uv run --script ./tests/nightly/backfill_perf_summary.py --dry-run --limit 3

# Backfill every run in the nightly project
uv run --script ./tests/nightly/backfill_perf_summary.py

# Recompute runs that are already marked processed
uv run --script ./tests/nightly/backfill_perf_summary.py --force
```

The first full pass reads every run's history and takes a while. It is resumable, so it
can be interrupted and restarted without redoing completed runs.

### `perf_workspace.py`
Creates a saved wandb workspace view plotting the above over time — one point per run
against when that run started, so drift is visible across nights rather than within a run.

Sections cover training throughput, collection time and learning time (min, mean and max
per run), and total run duration. Run `backfill_perf_summary.py` first or the panels will
be empty.

```bash
uv run --script ./tests/nightly/perf_workspace.py
uv run --script ./tests/nightly/perf_workspace.py --name "My trends view"

# A view of a single matrix cell, which is what the Slack trend block links to
uv run --script ./tests/nightly/perf_workspace.py --cell "g1_29dof_manager | isaacsim | gpus-1"
```

Throughput is not comparable across cells — multi-GPU figures are roughly 4x single-GPU by
construction — and the panels share a y-axis, so a view covering every cell is only useful
for spotting shape. `--cell` filters a view to one cell on the `perf/cell` summary key,
which keeps a point per run; grouping would not, because it collapses each group to one
aggregated point and the aggregation cannot combine run timestamps.

The command prints the saved view's URL. Nothing derives it: a view identifier is assigned
when the view is saved and only exists on the instance it was created on, and it is the
only thing a view URL carries — the cell filter is part of the stored view, not a query
parameter — so a per-cell link has to be created once and then configured.

## Throughput warnings

`nightly.py` also compares each run's mean fps against the last `PERF_BASELINE_RUNS` runs
of the same matrix cell (matching experiment, simulator and GPU count), and tags runs more
than `PERF_WARN_SIGMA` standard deviations below that baseline's mean with
`nightly_perf_warning`. Scaling to the measured spread rather than a fixed percentage
means the threshold tightens automatically on a cell that normally runs consistently.

These appear in the Slack summary as 🐌, and only ever on runs that otherwise passed: a
crash or a metrics regression takes precedence. They are **not** counted as failures —
throughput moves with the runner, so the check never fails a build. It is skipped until at
least `PERF_BASELINE_MIN_RUNS` prior runs carry the metric, and also when those runs are
all identical and so give no spread to measure against.

## GitHub Actions Integration

The nightly workflow automatically:
1. Runs training for multiple experiments across different configurations (simulator, GPU setup)
2. Generates a wandb report with all results
3. Posts a summary to Slack with a link to the report at the top

## Environment Variables

All scripts support the following environment variables:
- `WANDB_ENTITY`: wandb entity/org name (default: "amazon-far"). CI sets this from a
  repository variable; set it yourself when running these scripts locally, or they will
  query an entity you have no runs in.
- `WANDB_API_KEY`: wandb API key for authentication
- `WANDB_BASE_URL`: wandb server URL
- `GITHUB_RUN_ID`: GitHub Actions run ID (automatically set in CI)
- `SLACK_BOT_TOKEN`: Slack bot token for posting messages
- `SLACK_CHANNEL`: Slack channel to post to
- `PERF_TREND_VIEW_URL_<CELL>`: saved workspace view the Slack trend block links to, one
  variable per tracked configuration. `<CELL>` is the cell label upper-cased with every
  run of other characters replaced by an underscore, so the tracked
  `g1_29dof_manager | isaacsim | gpus-1` reads
  `PERF_TREND_VIEW_URL_G1_29DOF_MANAGER_ISAACSIM_GPUS_1`. Set it to the URL
  `perf_workspace.py --cell` printed for that cell; the summary logs the name it looked
  for whenever it is unset. CI supplies these from repository variables, because a view
  identifier is specific to the wandb instance the view was created on.

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
