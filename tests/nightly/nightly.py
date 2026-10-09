#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import datetime
import os
import subprocess
import sys
import time
from datetime import timezone
from os import getenv
from pathlib import Path
from typing import Any

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import get_annotated_experiment_config
from holosoma.train_agent import training_context
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


def now_timestamp() -> str:
    return datetime.datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


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

    with training_context(config) as ctx:
        # 1. Train
        ctx.train()

        # 2. Validate metrics (explicit, linear flow) - only on rank 0
        if os.environ.get("RANK", "0") == "0":
            validate_wandb_metrics(config)

    # 4. simulation_app automatically closed when exiting context


if __name__ == "__main__":
    main()
