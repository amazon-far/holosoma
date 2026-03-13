from __future__ import annotations

import dataclasses
import datetime
import os
import subprocess
import sys
from datetime import timezone
from os import getenv
from pathlib import Path

import tyro

from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.train_agent import training_context
from holosoma.utils.tyro_utils import TYRO_CONIFG

REPO_ROOT = Path(__file__).parent.parent.parent.absolute()

# Github assigned variables
GITHUB_SERVER_URL = getenv("GITHUB_SERVER_URL")
GITHUB_REPOSITORY = getenv("GITHUB_REPOSITORY")
GITHUB_RUN_ID = getenv("GITHUB_RUN_ID")


def now_timestamp() -> str:
    return datetime.datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def validate_wandb_metrics(config: AnnotatedExperimentConfig):
    # lazy import to avoid conflicts with Isaac
    import wandb

    assert wandb.run is not None, "wandb run failed! wandb.run is `None`"
    api = wandb.Api()
    run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
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

    # 3. Any other post-training work can go here
    if len(failures) > 0:
        print(f"Some tests failed! Metrics outside of expected ranges: {failures}")
        run.tags += ("nightly_test_failed",)
        run.update()
    else:
        run.tags += ("nightly_test_passed",)
        run.update()


def main():
    config = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)

    # Check if multigpu is requested and we're not already in a torchrun process
    if config.training.multigpu and "RANK" not in os.environ:
        # Re-launch with torchrun
        env = os.environ.copy()

        result = subprocess.run(
            [
                "torchrun",
                "--nproc_per_node=4",
                __file__,
                *sys.argv[1:],  # Pass all original arguments
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
    else:
        run_tags.append("singlegpu")

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
