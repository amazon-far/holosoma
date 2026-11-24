"""Run the fixed-base arm through reset, stepping, subset resets, and two PPO learning iterations.

Assert finite observations and rewards, articulated joint motion, and a stationary physical root."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path
from typing import Any

# tests/simulators/ has an isaacsim/ subpackage that would shadow the real IsaacSim package if it
# lands on sys.path[0] when run as a script -- drop it (mirrors the sibling harnesses).
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from holosoma.simulator.shared.root_states import root_pose_changed
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import torch
from tests.simulators._fixed_base_env import LEARNING_ITERATIONS, OBS_DIM, fixed_base_arm_experiment
from tests.simulators._robot_presets import test_arm_fixed
from tests.simulators._sim_harness import run_and_hard_exit

# Autotools-convention skip exit code; kept in sync with ``_run_harness.SKIP_EXIT_CODE``.
SKIP_EXIT_CODE = 77

POSE_ATOL = 1e-5  # robot_root_states vs the expected spawn state
ZERO_ACTION_STEPS = 50
DOF_DROOP_MIN = 1e-3  # rad the PD-held joints must have moved from default after the zero-action steps


def _fmt(t: torch.Tensor) -> list[float]:
    return [round(float(v), 4) for v in t.reshape(-1)]


def _root(env: Any) -> torch.Tensor:
    return env.simulator.robot_root_states[torch.arange(env.num_envs, device=env.device)].clone()


def _expected_root(env: Any) -> torch.Tensor:
    """``[num_envs, 13]``: terrain env origin + configured init pose (xyzw), zero velocity."""
    init = env.robot_config.init_state
    expected = torch.zeros(env.num_envs, 13, device=env.device)
    expected[:, :3] = env._get_env_origins() + torch.tensor(init.position, device=env.device)
    expected[:, 3:7] = torch.tensor(init.orientation_xyzw, device=env.device)
    return expected


def _base_body_pose(env: Any) -> torch.Tensor:
    """``[num_envs, 7]`` world pose (xyzw) of the welded base body read from the physics tensors."""
    sim = env.simulator
    base_idx = sim.find_rigid_body_indice("base_link")
    return torch.cat([sim.rigid_body_pos_w[:, base_idx], sim.rigid_body_quat_w[:, base_idx]], dim=1)


def _root_at_spawn(env: Any, expected: torch.Tensor, label: str, *, physics: bool = True) -> list[str]:
    """The root-state mirror and, with ``physics``, the base body's physics pose must equal ``expected``."""
    fails: list[str] = []
    root = _root(env)
    if not torch.allclose(root, expected, atol=POSE_ATOL):
        fails.append(f"{label}: robot_root_states {_fmt(root)} != spawn {_fmt(expected)}")
    if physics:
        pose = _base_body_pose(env)
        if root_pose_changed(pose, expected[:, :7]):
            fails.append(f"{label}: base body physics pose {_fmt(pose)} != spawn {_fmt(expected[:, :7])}")
    return fails


def _run_checks(env: Any, config: Any, device: str, log_dir: Path) -> list[str]:
    fails: list[str] = []
    num_envs = env.num_envs
    if not env.robot_config.asset.fix_base_link:
        return ["config resolved to a floating-base robot"]
    if env.dim_actions != len(test_arm_fixed.dof_names) or env.dof_names != list(test_arm_fixed.dof_names):
        fails.append("config resolved to a different joint or action order")
    if env.body_names != list(test_arm_fixed.body_names):
        fails.append("config resolved to a different body order")
    if env.terrain_manager.get_state("locomotion_terrain").feet_heights.shape[1] > 0:
        fails.append("feet height raycasts enabled on an arm without feet")

    env.reset_envs_idx()
    obs_dict = env.compute_reset_observations()
    expected = _expected_root(env)
    fails += _root_at_spawn(env, expected, "RESET", physics=False)
    print(f"  RESET root={_fmt(_root(env)[0])} ok={not fails}")

    actions = torch.zeros(num_envs, env.dim_actions, device=device)
    for step in range(ZERO_ACTION_STEPS):
        obs_dict, rewards, dones, infos = env.step({"actions": actions})
        for group in ("actor_obs", "critic_obs"):
            obs = obs_dict[group]
            if tuple(obs.shape) != (num_envs, OBS_DIM) or not bool(torch.isfinite(obs).all()):
                fails.append(
                    f"STEP {step}: {group} shape {tuple(obs.shape)} / finite {bool(torch.isfinite(obs).all())}"
                )
        if not bool(torch.isfinite(rewards).all()) or len(dones) != num_envs:
            fails.append(f"STEP {step}: rewards {_fmt(rewards)} dones={len(dones)}")
        if bool(torch.isnan(infos["to_log"]["average_episode_length"])):
            fails.append(f"STEP {step}: average_episode_length is NaN")
        fails += _root_at_spawn(env, expected, f"STEP {step}")
        if fails:
            break
    fails += _root_at_spawn(env, expected, "STEP")
    droop = float((env.simulator.dof_pos - env.default_dof_pos).abs().max())
    if droop <= DOF_DROOP_MIN:
        fails.append(f"STEP: joints did not move under gravity (max |dof_pos - default| = {droop:.2e})")
    print(f"  STEP root={_fmt(_root(env)[0])} dof_droop={droop:.4f} ok={not fails}")

    sim = env.simulator
    target = expected.clone()
    target[:, 2] += 0.5
    tip_idx = sim.find_rigid_body_indice("slider")
    tip_before, dof_before = sim.rigid_body_pos_w[:, tip_idx].clone(), sim.dof_pos.clone()
    env_ids = torch.arange(num_envs, device=device)
    sim.set_actor_root_state_tensor_robots(env_ids, target)
    env.step({"actions": actions})
    fails += _root_at_spawn(env, target, "TELEPORT")
    if not torch.allclose(
        sim.rigid_body_pos_w[:, tip_idx, 2] - tip_before[:, 2],
        torch.full((num_envs,), 0.5, device=device),
        atol=2e-3,
    ):
        fails.append("TELEPORT: the arm tip did not follow the mounted base")
    if not torch.allclose(sim.dof_pos, dof_before, atol=5e-2):
        fails.append("TELEPORT: joint positions changed beyond one control step")

    # Reset the moved environments, then exercise a subset reset at the spawn pose.
    env.reset_envs_idx(env_ids)
    env.step({"actions": actions})
    fails += _root_at_spawn(env, expected, "TELEPORT RESET")
    subset = torch.arange(0, num_envs, 2, device=device)
    env.reset_envs_idx(subset)
    env.step({"actions": actions})
    fails += _root_at_spawn(env, expected, "SUBSET")
    print(f"  SUBSET envs={subset.tolist()} root={_fmt(_root(env)[0])} ok={not fails}")

    agent = get_class(config.algo._target_)(env=env, config=config.algo.config, log_dir=str(log_dir), device=device)
    agent.setup()
    agent.attach_checkpoint_metadata(config, None)
    agent.learn()
    if agent.current_learning_iteration != LEARNING_ITERATIONS - 1:
        fails.append(
            f"LEARN: stopped at iteration {agent.current_learning_iteration}, expected {LEARNING_ITERATIONS - 1}"
        )
    if not (log_dir / f"model_{LEARNING_ITERATIONS - 1:05d}.pt").exists():
        fails.append("LEARN: the final checkpoint was not saved")
    fails += _root_at_spawn(env, expected, "LEARN")
    print(f"  LEARN iterations={agent.current_learning_iteration + 1} root={_fmt(_root(env)[0])} ok={not fails}")
    return fails


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-backend fixed-base robot env/PPO assertion harness.")
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacsim"])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--result-file", default=None, help="write 'OK' here after PASS (teardown-robust)")
    args = parser.parse_args()

    if args.simulator == "mujoco" and args.num_envs > 1:
        print("SKIP: MuJoCo ClassicBackend is single-env (use mjwarp for --num-envs > 1)")
        return SKIP_EXIT_CODE

    device = "cpu" if args.simulator == "mujoco" else "cuda:0"
    work_dir = Path(tempfile.mkdtemp(prefix="fixed_base_env_"))
    seeding(0)
    config = fixed_base_arm_experiment(args.simulator, args.num_envs, work_dir / "logs")
    with dataclasses.replace(config, device=device).create_task() as env:
        print(
            f"[{args.simulator}] robot={config.robot.asset.robot_type} "
            f"fixed_base={env.robot_config.asset.fix_base_link} dof={env.dof_names} "
            f"num_envs={env.num_envs} device={device}"
        )
        fails = _run_checks(env, config, device, work_dir / "ppo")
        # Report before the session closes: IsaacSim's app teardown can end the process (exit code and
        # late stdout are unreliable there), so the result file is written inside the session.
        if args.result_file:
            with open(args.result_file, "w") as fh:
                fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
        for msg in fails:
            print(f"FAIL: {msg}")
        if not fails:
            print(f"FIXED-BASE ENV OK: {args.simulator} ({args.num_envs} env(s)) -- reset, step, subset reset, PPO")
    return 1 if fails else 0


if __name__ == "__main__":
    run_and_hard_exit(main)
