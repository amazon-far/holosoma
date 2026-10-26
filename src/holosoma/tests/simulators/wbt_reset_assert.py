"""WBT reset regression: reset_all() leaves valid motion targets, so step 1 ends no episode.

Builds the G1 whole-body-tracking task, resets it, and takes one zero-action step. Each env
starts on its sampled motion frame, so tracking error is small and no termination should fire.
If reset_all() left the tracked-body targets at their zero-initialized values, the bad-tracking
terminations would end almost every episode on that first step.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import cast

# Running this file directly puts tests/simulators first, where the isaacsim test package would
# shadow NVIDIA's isaacsim package.
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from holosoma.config_values import experiment
from holosoma.config_values.simulator import SIMULATOR_REGISTRY
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.safe_torch_import import torch
from tests.simulators._sim_harness import run_and_hard_exit

NUM_ENVS = 16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simulator", required=True, choices=sorted(SIMULATOR_REGISTRY))
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()

    preset = experiment.g1_29dof_wbt
    tyro_config = dataclasses.replace(
        preset,
        simulator=SIMULATOR_REGISTRY[args.simulator],
        training=dataclasses.replace(preset.training, num_envs=NUM_ENVS, headless=True),
    )

    failures: list[str] = []
    with tyro_config.create_task() as task:
        env = cast("WholeBodyTrackingManager", task)
        motion = cast("MotionCommand", env.command_manager.get_state("motion_command"))
        env.reset_all()
        if not bool(motion.body_pos_relative_w.abs().amax() > 0.0):
            failures.append("reset_all() left the tracked-body targets at zero")

        actions = torch.zeros(NUM_ENVS, tyro_config.robot.actions_dim, device=env.device)
        _, _, dones, _ = env.step({"actions": actions})
        if int(dones.sum()) != 0:
            failures.append(f"{int(dones.sum())}/{NUM_ENVS} episodes ended on the first zero-action step")

        # Report before leaving the session: closing the Isaac Sim app ends the process.
        if failures:
            print(f"FAIL: {args.simulator} WBT reset: " + "; ".join(failures), flush=True)
            return 1
        if args.result_file is not None:
            args.result_file.write_text("OK\n")
        print(f"PASS: {args.simulator} WBT reset targets and first step", flush=True)
        return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
