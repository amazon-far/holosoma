#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import datetime
import itertools
import math
import os
import statistics
import subprocess
import sys
import time
from os import getenv
from pathlib import Path
from typing import Any, cast

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import get_annotated_experiment_config
from holosoma.policy_task import PolicyTask
from holosoma.train_agent import train_on_task
from holosoma.utils.config_registry import parse_config

REPO_ROOT = Path(__file__).parent.parent.parent.absolute()

# Github assigned variables
GITHUB_SERVER_URL = getenv("GITHUB_SERVER_URL")
GITHUB_REPOSITORY = getenv("GITHUB_REPOSITORY")
GITHUB_RUN_ID = getenv("GITHUB_RUN_ID")

# Number of GPUs used for a multi-GPU nightly run (matches the distributed
# launcher below and the x4 GPU runner in .github/workflows/nightly-training.yaml).
MULTIGPU_NUM_GPUS = 4

NIGHTLY_STATUS_TAGS = frozenset({"nightly_test_passed", "nightly_test_failed"})
WANDB_RUN_FINISH_ATTEMPTS = 10
WANDB_STATUS_UPDATE_ATTEMPTS = 3
WANDB_STATUS_UPDATE_DELAY_S = 3.0

# Throughput regression warning. Deliberately kept out of NIGHTLY_STATUS_TAGS: a slow
# night is reported but never fails the build, so it must not displace the pass/fail tag.
PERF_WARNING_TAG = "nightly_perf_warning"
# Per-run mean fps, written by backfill_perf_summary.py. Keep in sync with that script,
# including the number of leading iterations skipped, or this run's mean won't be
# comparable to the stored baselines.
PERF_FPS_MEAN_KEY = "perf/total_fps_mean"
PERF_WARMUP_ROWS = 1
# One nightly run per cell, so the last 14 runs are the last 14 nights. Counted in runs
# rather than days so a skipped night shortens the window instead of thinning the sample.
PERF_BASELINE_RUNS = 14
PERF_BASELINE_MIN_RUNS = 5
# How far below the baseline mean a run has to fall, in standard deviations of that
# baseline, before it is called out. Scaling to the measured spread rather than a fixed
# percentage means the threshold tightens on a cell that normally runs consistently.
PERF_WARN_SIGMA = 2.0


def now_timestamp() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def wait_for_wandb_run(api: Any, run_path: str) -> Any:
    for attempt in range(1, WANDB_RUN_FINISH_ATTEMPTS + 1):
        run = api.run(run_path)
        if run.state == "finished":
            return run

        print(f"W&B run is not finished yet (state={run.state!r}); retrying ({attempt}/{WANDB_RUN_FINISH_ATTEMPTS})")
        time.sleep(WANDB_STATUS_UPDATE_DELAY_S)

    raise RuntimeError(f"W&B run did not finish before validation: {run_path}")


def update_wandb_status(api: Any, run_path: str, status_tag: str) -> None:
    if status_tag not in NIGHTLY_STATUS_TAGS:
        raise ValueError(f"Unexpected nightly status tag: {status_tag}")

    for attempt in range(1, WANDB_STATUS_UPDATE_ATTEMPTS + 1):
        run = api.run(run_path)
        run.tags = [tag for tag in (run.tags or []) if tag not in NIGHTLY_STATUS_TAGS] + [status_tag]
        run.update()
        time.sleep(WANDB_STATUS_UPDATE_DELAY_S)

        updated_run = api.run(run_path)
        if status_tag in (updated_run.tags or []):
            return

        print(f"W&B status tag did not persist; retrying ({attempt}/{WANDB_STATUS_UPDATE_ATTEMPTS})")

    raise RuntimeError(f"Failed to persist W&B status tag {status_tag!r} for {run_path}")


def append_wandb_tag(api: Any, run_path: str, tag: str) -> None:
    """Adds a tag to a finished run, leaving its existing tags in place."""
    run = api.run(run_path)
    existing = list(run.tags or [])
    if tag in existing:
        return
    run.tags = existing + [tag]
    run.update()


def mean_run_fps(run: Any) -> float | None:
    """Mean of a run's per-iteration fps, or None when it logged none.

    Reads the full history rather than `run.history()`, which samples, and skips the
    same leading rows as backfill_perf_summary.py so the result is comparable to the
    baseline values that script stores.
    """
    total = 0.0
    count = 0
    for row in itertools.islice(run.scan_history(keys=["Perf/total_fps"]), PERF_WARMUP_ROWS, None):
        value = row.get("Perf/total_fps")
        if isinstance(value, (int, float)) and math.isfinite(value):
            total += float(value)
            count += 1
    return total / count if count else None


def baseline_fps(api: Any, run_path: str, cell_tags: list[str], current_run_id: str) -> tuple[float, float] | None:
    """Mean and standard deviation of recent prior runs sharing this run's matrix cell.

    Returns None when too few prior runs carry the metric, or when they are all
    identical and so give no spread to measure against.
    """
    entity, project, _ = run_path.split("/")
    previous: list[float] = []
    required = set(cell_tags)

    for other in api.runs(path=f"{entity}/{project}", filters={"tags": {"$all": cell_tags}}, order="-created_at"):
        if other.id == current_run_id:
            continue
        # The filter above is only an optimisation; re-check locally so an unsupported
        # or ignored server-side filter can't silently pool runs from other cells.
        if not required.issubset(set(other.tags or [])):
            continue
        value = (other.summary or {}).get(PERF_FPS_MEAN_KEY)
        if isinstance(value, (int, float)) and math.isfinite(value):
            previous.append(float(value))
        if len(previous) >= PERF_BASELINE_RUNS:
            break

    if len(previous) < PERF_BASELINE_MIN_RUNS:
        print(
            f"Throughput baseline needs {PERF_BASELINE_MIN_RUNS} prior runs with {PERF_FPS_MEAN_KEY}, "
            f"found {len(previous)}; skipping throughput check"
        )
        return None

    spread = statistics.stdev(previous)
    if spread <= 0:
        print("Throughput baseline has no spread; skipping throughput check")
        return None
    return statistics.fmean(previous), spread


def check_throughput_regression(cell_tags: list[str]) -> None:
    """Tags the run when its mean fps falls well below recent runs of the same cell.

    Warn-only: throughput moves with the runner, so this never raises and never fails
    the build. Any error here is reported and ignored for the same reason.
    """
    # lazy import to avoid conflicts with Isaac
    import wandb

    try:
        if wandb.run is None:
            return
        api = wandb.Api()
        run_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"
        run = wait_for_wandb_run(api, run_path)

        current = mean_run_fps(run)
        if current is None:
            print("Run logged no fps history; skipping throughput check")
            return

        stats = baseline_fps(api, run_path, cell_tags, wandb.run.id)
        if stats is None:
            return

        baseline, spread = stats
        threshold = baseline - PERF_WARN_SIGMA * spread
        summary = (
            f"mean fps {current:0.0f} vs baseline {baseline:0.0f} ±{spread:0.0f} "
            f"(warn below {threshold:0.0f}, {PERF_WARN_SIGMA:g} sigma)"
        )
        if current < threshold:
            print(f"Throughput regression warning: {summary}")
            append_wandb_tag(api, run_path, PERF_WARNING_TAG)
        else:
            print(f"Throughput within range: {summary}")
    except Exception as e:
        print(f"Throughput check failed, ignoring: {e}")


def validate_wandb_metrics(config: ExperimentConfig):
    # lazy import to avoid conflicts with Isaac
    import wandb

    assert wandb.run is not None, "wandb run failed! wandb.run is `None`"
    api = wandb.Api()
    run_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"
    run = wait_for_wandb_run(api, run_path)
    df_hist = run.history()

    failures: list[str] = []
    assert config.nightly is not None  # for type checking
    assert config.nightly.metrics is not None

    for k, v in config.nightly.metrics.items():
        v_min = float(v[0])
        v_max = float(v[1])
        v_last_100 = df_hist[k][-100:].mean()

        is_in_range = v_min <= v_last_100 <= v_max
        if not is_in_range:
            msg = f"Metric {k}={v_last_100:0.2f} is not in range ({v_min}, {v_max})"
            print(msg)
            failures.append(msg)

    if failures:
        print(f"Some tests failed! Metrics outside of expected ranges: {failures}")
        update_wandb_status(api, run_path, "nightly_test_failed")
        raise RuntimeError("Nightly metrics are outside their expected ranges")
    update_wandb_status(api, run_path, "nightly_test_passed")


def main():
    original_args = sys.argv[1:]
    config = parse_config(get_annotated_experiment_config)

    # Check if multigpu is requested and we're not already in a distributed process
    if config.training.multigpu and "RANK" not in os.environ:
        # The Isaac Sim environment does not install the torchrun console script,
        # but its bundled PyTorch still provides the equivalent module entry point.
        env = os.environ.copy()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={MULTIGPU_NUM_GPUS}",
                __file__,
                *original_args,  # Pass all original arguments
            ],
            env=env,
            check=False,
        )

        sys.exit(result.returncode)

    config = dataclasses.replace(config, training=dataclasses.replace(config.training, seed=42))
    # Get experiment name from config instead of hydra runtime choices
    exp = config.training.name or config.logger.name
    # Sanitize experiment name for wandb project name (cannot contain /,\,#,?,%,:)
    sanitized_exp = (
        exp.replace("/", "-").replace("\\", "-").replace("#", "-").replace("?", "-").replace("%", "-").replace(":", "-")
    )

    # Add multigpu suffix if enabled
    multigpu_suffix = "-multigpu" if config.training.multigpu else ""

    config = config.get_nightly_config()

    run_tags = [
        sanitized_exp,
        config.simulator.config.name,
    ]

    if GITHUB_RUN_ID:
        run_tags.append(f"gha-run-id-{GITHUB_RUN_ID}")

    if config.training.multigpu:
        run_tags.append("multigpu")
        run_tags.append(f"gpus-{MULTIGPU_NUM_GPUS}")
    else:
        run_tags.append("singlegpu")
        run_tags.append("gpus-1")

    # Tags identifying this matrix cell (experiment, simulator, GPU count). Throughput
    # is only comparable within a cell, so the regression baseline is limited to these.
    cell_tags = [
        sanitized_exp,
        config.simulator.config.name,
        f"gpus-{MULTIGPU_NUM_GPUS}" if config.training.multigpu else "gpus-1",
    ]

    nightly_name = f"nightly-{sanitized_exp}{multigpu_suffix}-{now_timestamp()}"

    config = dataclasses.replace(
        config,
        logger=dataclasses.replace(
            config.logger,
            project="nightly-holosoma-runs",
            name=nightly_name,
            id=nightly_name,  # set id to name so url is readable
            tags=tuple(run_tags),
        ),
    )

    with config.create_task() as env:
        # 1. Train
        train_on_task(config, cast("PolicyTask", env))

        # 2. Validate metrics (explicit, linear flow) - only on rank 0
        if os.environ.get("RANK", "0") == "0":
            # Warn-only, and ahead of validation so it still reports on a failing night.
            check_throughput_regression(cell_tags)
            validate_wandb_metrics(config)

    # 4. simulation_app automatically closed when exiting context


if __name__ == "__main__":
    main()
