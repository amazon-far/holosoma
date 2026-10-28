"""Environment-contract regression: batch shapes and episode bookkeeping through ``env.step``.

Builds the G1 locomotion training environment through ``training_context`` (which owns the
simulator app, so it starts before any environment module imports the backend) and steps zero
actions. It checks the public environment behavior every backend must keep:

1. Actor and critic observations, rewards and dones all have ``num_envs`` rows.
2. Completed-episode metrics have one entry per done; all-environment metrics keep the full batch.
3. The logged average episode length is never NaN, some episodes finish, and ``reset_all()``
   returns the average episode length to zero.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# Running this file directly puts tests/simulators first, where the isaacsim test package would
# shadow NVIDIA's isaacsim package.
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from holosoma.config_values import experiment
from holosoma.config_values.simulator import SIMULATOR_REGISTRY
from holosoma.train_agent import get_tyro_env_config, training_context
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import torch
from tests.simulators._sim_harness import run_and_hard_exit

NUM_ENVS = 16
NUM_STEPS = 100
DEVICE = "cuda:0"
EPISODE_METRIC = "rew_tracking_lin_vel"


def _check_step(failures: list[str], step: int, obs_dict, rewards, dones, infos) -> None:
    sizes = {
        "actor_obs": len(obs_dict["actor_obs"]),
        "critic_obs": len(obs_dict["critic_obs"]),
        "rewards": len(rewards),
        "dones": len(dones),
        f"episode_all[{EPISODE_METRIC}]": len(infos["episode_all"][EPISODE_METRIC]),
    }
    failures.extend(
        f"step {step}: {name} has {size} rows, expected {NUM_ENVS}" for name, size in sizes.items() if size != NUM_ENVS
    )
    completed = len(infos["episode"][EPISODE_METRIC])
    if completed != int(dones.sum()):
        failures.append(f"step {step}: episode[{EPISODE_METRIC}] has {completed} entries for {int(dones.sum())} dones")
    if torch.isnan(infos["to_log"]["average_episode_length"]):
        failures.append(f"step {step}: average_episode_length is NaN")


def main() -> int:
    parser = argparse.ArgumentParser(description="Environment batch-shape and episode-bookkeeping regression.")
    parser.add_argument("--simulator", required=True, choices=sorted(SIMULATOR_REGISTRY))
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()

    seeding(0)
    preset = experiment.g1_29dof
    tyro_config = dataclasses.replace(
        preset,
        simulator=SIMULATOR_REGISTRY[args.simulator],
        training=dataclasses.replace(preset.training, num_envs=NUM_ENVS, headless=True),
    )

    failures: list[str] = []
    tyro_env_config = get_tyro_env_config(tyro_config)
    with (
        training_context(tyro_config) as context,
        context.simulation_session(get_class(tyro_config.env_class)(tyro_env_config, device=DEVICE)) as env,
    ):
        obs_dict = env.reset_all()
        failures.extend(
            f"reset_all: {name} has {len(obs_dict[name])} rows, expected {NUM_ENVS}"
            for name in ("actor_obs", "critic_obs")
            if len(obs_dict[name]) != NUM_ENVS
        )

        actions = torch.zeros(NUM_ENVS, tyro_config.robot.actions_dim, device=DEVICE)
        n_dones = 0
        for step in range(NUM_STEPS):
            obs_dict, rewards, dones, infos = env.step({"actions": actions})
            _check_step(failures, step, obs_dict, rewards, dones, infos)
            n_dones += int(dones.sum())
        if n_dones == 0:
            failures.append(f"no episode finished in {NUM_STEPS} zero-action steps")

        env.reset_all()
        if env.average_episode_length != 0.0:
            failures.append(f"average_episode_length is {env.average_episode_length} after reset_all, expected 0.0")

        # Report before leaving the session: closing the Isaac Sim app ends the process.
        return _report(args, failures, n_dones)


def _report(args: argparse.Namespace, failures: list[str], n_dones: int) -> int:
    print(f"ENV_STEP backend={args.simulator} num_envs={NUM_ENVS} steps={NUM_STEPS} dones={n_dones}", flush=True)
    if failures:
        print(f"FAIL: {args.simulator} environment contract: {len(failures)} violation(s)")
        for failure in failures[:20]:
            print(f"  {failure}")
        sys.stdout.flush()
        return 1

    if args.result_file is not None:
        args.result_file.write_text("OK\n")
    print(f"PASS: {args.simulator} environment batch shapes and episode bookkeeping", flush=True)
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
