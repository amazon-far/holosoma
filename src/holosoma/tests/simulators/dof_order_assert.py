"""Cross-backend check that DOF state and torques follow ``robot_config.dof_names``, not the asset's order.

Builds G1 with its config DOF order rotated away from the asset's joint order
(``g1-29dof-reordered``), then checks through physics rather than index bookkeeping:

  1. ORDER — ``simulator.dof_names`` equals the config's ``dof_names``.
  2. STATE — setting one joint by name and running forward kinematics moves that limb's distal body
     and not its opposite-side twin, and the joint reads back the written value.
  3. TORQUE — during free fall (gravity alone leaves joint velocities at zero), a torque on one
     named joint makes that joint the fastest-moving DOF, turning in the torque's direction.

Exits 0 on success, non-zero with a message on failure.

Usage:
  python dof_order_assert.py --simulator mujoco     # ClassicBackend, 1 env, cpu
  python dof_order_assert.py --simulator mjwarp     # WarpBackend, cuda
  python dof_order_assert.py --simulator isaacsim   # IsaacSim, cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

# tests/simulators/ has an isaacsim/ subpackage that would shadow the real IsaacSim package if it
# lands on sys.path[0] when run as a script.
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

import tyro

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.simulator import BridgeConfig, VirtualGantryCfg
from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators import _robot_presets
from tests.simulators._sim_harness import step

ROBOT = "g1-29dof-reordered"

# (joint set by name, body that must move, opposite-side body that must not).
STATE_CASES = [
    ("left_elbow_joint", "left_wrist_yaw_link", "right_wrist_yaw_link"),
    ("right_knee_joint", "right_ankle_roll_link", "left_ankle_roll_link"),
    ("left_hip_pitch_joint", "left_ankle_roll_link", "right_ankle_roll_link"),
    ("waist_yaw_joint", "left_wrist_yaw_link", "left_ankle_roll_link"),
]
TORQUE_JOINTS = ["left_elbow_joint", "right_wrist_roll_joint", "left_ankle_roll_joint"]
JOINT_OFFSET = 0.5  # rad
TORQUE = 1.0  # N·m
TORQUE_STEPS = 5


def _build(simulator: str, num_envs: int):
    _robot_presets.register()
    argv = [f"simulator:{simulator}", f"robot:{ROBOT}", "terrain:terrain_locomotion_plane", "scene:empty"]
    config = tyro.cli(RunSimConfig, args=argv)
    sim_cfg = dataclasses.replace(
        config.simulator,
        config=dataclasses.replace(
            config.simulator.config, bridge=BridgeConfig(enabled=False), virtual_gantry=VirtualGantryCfg(enabled=False)
        ),
    )
    config = dataclasses.replace(
        config,
        simulator=sim_cfg,
        device=("cpu" if simulator == "mujoco" else "cuda:0"),
        training=dataclasses.replace(config.training, num_envs=num_envs),
    )
    import torch

    env, device, _ = setup_simulation_environment(config, device=config.device)
    s = env.sim
    s.set_headless(True)
    s.setup()
    s.setup_terrain()
    s.load_assets()
    origins = torch.zeros(num_envs, 3, device=device)
    origins[:, 0] = torch.arange(num_envs, device=device, dtype=torch.float32) * 4.0
    init = config.robot.init_state
    base_init = torch.tensor(list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device)
    s.create_envs(num_envs, origins, base_init)
    s.prepare_sim()
    s.install_plugins()
    return s, config.robot


def _body_positions(sim):
    sim.forward_kinematics()
    sim.refresh_sim_tensors()
    return sim.rigid_body_pos_w.clone()


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-backend DOF-order assertion harness.")
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacsim"])
    parser.add_argument("--num-envs", type=int, default=None)
    args = parser.parse_args()
    num_envs = args.num_envs if args.num_envs is not None else (1 if args.simulator == "mujoco" else 2)

    import torch

    sim, robot = _build(args.simulator, num_envs)
    failures: list[str] = []

    # 1. ORDER
    if list(sim.dof_names) != list(robot.dof_names):
        print(f"FAIL: simulator dof_names {sim.dof_names} != config dof_names {robot.dof_names}")
        return 1

    # 2. STATE
    defaults = {name: robot.init_state.default_joint_angles[name] for name in robot.dof_names}
    zero_vel = dict.fromkeys(robot.dof_names, 0.0)
    sim.set_dof_state_by_name(None, defaults, zero_vel)
    baseline = _body_positions(sim)
    for joint, moves, stays in STATE_CASES:
        target = defaults[joint] + JOINT_OFFSET
        sim.set_dof_state_by_name(None, {joint: target})
        delta = (_body_positions(sim) - baseline).norm(dim=-1)  # [num_envs, num_bodies]
        moved = delta[:, sim.find_rigid_body_indice(moves)]
        still = delta[:, sim.find_rigid_body_indice(stays)]
        read_back = sim.get_dof_state_by_name([joint])[joint][0]
        print(f"  STATE {joint}: {moves} moved {moved.tolist()}, {stays} moved {still.tolist()}")
        if not bool((moved > 1e-2).all()) or not bool((still < 1e-4).all()):
            failures.append(f"setting {joint} moved the wrong bodies")
        if not torch.allclose(read_back, torch.full_like(read_back, target), atol=1e-5):
            failures.append(f"{joint} read back {read_back.tolist()}, wrote {target}")
        sim.set_dof_state_by_name(None, {joint: defaults[joint]})

    # 3. TORQUE
    for joint in TORQUE_JOINTS:
        sim.set_dof_state_by_name(None, defaults, zero_vel)
        for _ in range(TORQUE_STEPS):
            sim.apply_torques_at_dof(torch.zeros(num_envs, sim.num_dof, device=sim.sim_device))
            sim.apply_torques_by_name({joint: TORQUE})
            step(sim, 1)
        sim.refresh_sim_tensors()
        velocities = sim.dof_vel.abs()
        fastest = [sim.dof_names[i] for i in velocities.argmax(dim=-1).tolist()]
        joint_vel = sim.get_dof_state_by_name([joint])[joint][1]
        print(f"  TORQUE {joint}: fastest DOF per env {fastest}, its velocity {joint_vel.tolist()}")
        if any(name != joint for name in fastest) or not bool((joint_vel > 0).all()):
            failures.append(f"torque on {joint} did not drive {joint}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"PASS: {args.simulator} DOF state and torques follow robot_config.dof_names ({num_envs} env(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
