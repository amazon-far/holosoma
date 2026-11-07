"""Cross-backend check that DOF state and torques follow ``robot_config.dof_names``, not the asset's order.

Builds G1 with its config DOF order rotated away from the asset's joint order
(``g1-29dof-reordered``), then checks through physics rather than index bookkeeping:

  1. ORDER — ``simulator.dof_names`` equals the config's ``dof_names``.
  2. STATE — setting one joint by name and running forward kinematics moves that limb's distal body
     and not its opposite-side twin, and the joint reads back the written value.
  3. TORQUE — during free fall (gravity alone leaves joint velocities at zero), a torque on one
     named joint makes that joint the fastest-moving DOF, turning in the torque's direction.
  4. LIMITS — native solver effort limits preserve the URDF independently of model clipping.
  5. CLIPPING — native IsaacLab clips positive and negative elbow commands to the G1's 25 N·m limit
     and the joint moves in the commanded direction.

Exits 0 on success, non-zero with a message on failure.

Usage:
  python dof_order_assert.py --simulator mujoco     # ClassicBackend, 1 env, cpu
  python dof_order_assert.py --simulator mjwarp     # WarpBackend, cuda
  python dof_order_assert.py --simulator isaacsim   # IsaacSim, cuda
  python dof_order_assert.py --simulator isaacsim3  # Native Isaac Lab 3, cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# tests/simulators/ has an isaacsim/ subpackage that would shadow the real IsaacSim package if it
# lands on sys.path[0] when run as a script.
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

import tyro

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.simulator import BridgeConfig, IsaacSimBackend, VirtualGantryCfg
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
LOWER_MODEL_LIMIT_JOINT = "right_wrist_roll_joint"
LOWER_MODEL_LIMIT = 2.0  # N·m; the G1 URDF authors 25 N·m for this joint


def _build(simulator: str, num_envs: int, *, use_newton_actuators: bool = True):
    _robot_presets.register()
    argv = [f"simulator:{simulator}", f"robot:{ROBOT}", "terrain:terrain_locomotion_plane", "scene:empty"]
    config = tyro.cli(RunSimConfig, args=argv)
    if config.simulator.config.isaacsim_backend == IsaacSimBackend.ISAACLAB3:
        # Make model and authored solver limits deliberately different. The solver
        # readback must preserve the URDF while actuator commands clip to this limit.
        effort_limits = list(config.robot.dof_effort_limit_list)
        effort_limits[config.robot.dof_names.index(LOWER_MODEL_LIMIT_JOINT)] = LOWER_MODEL_LIMIT
        config = dataclasses.replace(
            config, robot=dataclasses.replace(config.robot, dof_effort_limit_list=effort_limits)
        )
    sim_cfg = dataclasses.replace(
        config.simulator,
        config=dataclasses.replace(
            config.simulator.config,
            bridge=BridgeConfig(enabled=False),
            virtual_gantry=VirtualGantryCfg(enabled=False),
            sim=dataclasses.replace(
                config.simulator.config.sim,
                isaaclab3=dataclasses.replace(
                    config.simulator.config.sim.isaaclab3, use_newton_actuators=use_newton_actuators
                ),
            ),
        ),
    )
    config = dataclasses.replace(
        config,
        simulator=sim_cfg,
        device=("cpu" if simulator == "mujoco" else "cuda:0"),
        training=dataclasses.replace(config.training, num_envs=num_envs, headless=True),
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
    return sim.rigid_body_pos_w.clone()


def _check_isaac_joint_properties(sim, robot) -> None:
    """Read the solver independently of the actuator config and Holosoma ordering."""
    import torch

    native = sim.backend.is_isaaclab3
    asset = sim._robot
    view = asset.root_view if native else asset.root_physx_view
    names = list(asset.backend_joint_names if native else asset.joint_names)
    if native and not sim.backend.uses_kit:
        from isaaclab_ov import tensor_types as TT

        armatures = view.get_attribute(TT.DOF_ARMATURE)
        friction = view.get_attribute(TT.DOF_FRICTION_PROPERTIES)
    else:
        armatures = view.get_dof_armatures()
        friction = view.get_dof_friction_properties()
    if native:
        import warp as wp

        armatures = wp.to_torch(armatures)
        friction = wp.to_torch(friction)
    for label, values, actual in (
        ("armature", robot.dof_armature_list, armatures),
        ("friction", robot.dof_joint_friction_list, friction[:, :, 0]),
    ):
        by_name = dict(zip(robot.dof_names, values, strict=True))
        expected = torch.tensor([by_name[name] for name in names], dtype=actual.dtype).expand(sim.num_envs, -1)
        torch.testing.assert_close(
            actual.cpu(),
            expected,
            rtol=1e-5,
            atol=1e-7,
            msg=f"{label} was not applied to the matching solver joint",
        )
    print("  PROPERTIES: solver armature and static friction follow joint names")
    if native:
        from holosoma.utils.path import resolve_asset_path

        # Read the source asset, not the actuator config or telemetry. A solver limit
        # widened to 1e9 or replaced with a model limit must fail this comparison.
        assert robot.asset.usd_file is None, "The authored-limit oracle requires the G1 URDF fixture"
        root = ET.parse(  # noqa: S314 - the oracle reads the repository's fixed G1 asset fixture
            resolve_asset_path(robot.asset.urdf_file, robot.asset.asset_root)
        ).getroot()
        authored_effort = {
            joint.attrib["name"]: float(limit.attrib["effort"])
            for joint in root.findall("joint")
            if joint.get("type") != "fixed" and (limit := joint.find("limit")) is not None
        }
        assert authored_effort[LOWER_MODEL_LIMIT_JOINT] == 25.0
        assert robot.dof_effort_limit_list[robot.dof_names.index(LOWER_MODEL_LIMIT_JOINT)] == LOWER_MODEL_LIMIT
        raw_limits = view.get_dof_max_forces() if sim.backend.uses_kit else view.get_attribute(TT.DOF_MAX_FORCE)
        solver_limits = wp.to_torch(raw_limits).cpu()
        expected_limits = torch.tensor([authored_effort[name] for name in names], dtype=solver_limits.dtype).expand(
            sim.num_envs, -1
        )
        torch.testing.assert_close(
            solver_limits,
            expected_limits,
            rtol=1e-5,
            atol=1e-6,
            msg="native solver effort limits no longer match the authored URDF joints",
        )
        print(f"  LIMITS: raw solver effort limits match all {len(names)} authored URDF joints")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-backend DOF-order assertion harness.")
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacsim", "isaacsim3"])
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--use-newton-actuators", choices=["true", "false"], default="true")
    parser.add_argument("--result-file", default=None, help="write 'OK' after all checks pass")
    args = parser.parse_args()
    num_envs = args.num_envs if args.num_envs is not None else (1 if args.simulator == "mujoco" else 2)

    import torch

    use_newton_actuators = args.use_newton_actuators == "true"
    sim, robot = _build(args.simulator, num_envs, use_newton_actuators=use_newton_actuators)
    failures: list[str] = []

    # 1. ORDER
    if list(sim.dof_names) != list(robot.dof_names):
        print(f"FAIL: simulator dof_names {sim.dof_names} != config dof_names {robot.dof_names}")
        return 1
    if args.simulator in ("isaacsim", "isaacsim3"):
        if sim.backend.is_isaaclab3:
            assert sim.sim.cfg.use_newton_actuators is use_newton_actuators, (
                "SimulationCfg did not preserve the requested actuator execution path"
            )
            print(f"  ACTUATORS: use_newton_actuators={sim.sim.cfg.use_newton_actuators}")
        _check_isaac_joint_properties(sim, robot)

    # 2. STATE
    defaults = {name: robot.init_state.default_joint_angles[name] for name in robot.dof_names}
    zero_vel = dict.fromkeys(robot.dof_names, 0.0)
    sim.set_dof_state_by_names(None, defaults, zero_vel)
    baseline = _body_positions(sim)
    for joint, moves, stays in STATE_CASES:
        target = defaults[joint] + JOINT_OFFSET
        sim.set_dof_state_by_names(None, {joint: target})
        delta = (_body_positions(sim) - baseline).norm(dim=-1)  # [num_envs, num_bodies]
        moved = delta[:, sim.find_rigid_body_indice(moves)]
        still = delta[:, sim.find_rigid_body_indice(stays)]
        read_back = sim.get_dof_state_by_names([joint])[joint][0]
        print(f"  STATE {joint}: {moves} moved {moved.tolist()}, {stays} moved {still.tolist()}")
        if not bool((moved > 1e-2).all()) or not bool((still < 1e-4).all()):
            failures.append(f"setting {joint} moved the wrong bodies")
        if not torch.allclose(read_back, torch.full_like(read_back, target), atol=1e-5):
            failures.append(f"{joint} read back {read_back.tolist()}, wrote {target}")
        sim.set_dof_state_by_names(None, {joint: defaults[joint]})

    # 3. TORQUE
    for joint in TORQUE_JOINTS:
        sim.set_dof_state_by_names(None, defaults, zero_vel)
        for _ in range(TORQUE_STEPS):
            sim.apply_torques_at_dof(torch.zeros(num_envs, sim.num_dof, device=sim.sim_device))
            sim.apply_torques_by_names({joint: TORQUE})
            step(sim, 1)
        velocities = sim.dof_vel.abs()
        fastest = [sim.dof_names[i] for i in velocities.argmax(dim=-1).tolist()]
        joint_vel = sim.get_dof_state_by_names([joint])[joint][1]
        print(f"  TORQUE {joint}: fastest DOF per env {fastest}, its velocity {joint_vel.tolist()}")
        if any(name != joint for name in fastest) or not bool((joint_vel > 0).all()):
            failures.append(f"torque on {joint} did not drive {joint}")
        if args.simulator in ("isaacsim", "isaacsim3") and sim.backend.is_isaaclab3:
            applied = torch.stack([sim.get_dof_forces(env_id) for env_id in range(num_envs)])
            expected = torch.zeros_like(applied)
            expected[:, sim.dof_names.index(joint)] = TORQUE
            if not torch.allclose(applied, expected, rtol=1e-5, atol=1e-6):
                failures.append(f"applied effort for {joint} did not match the commanded effort in every environment")

    # 4. CLIPPING (both native actuator execution paths).
    if args.simulator in ("isaacsim", "isaacsim3") and sim.backend.is_isaaclab3:
        import warp as wp

        free_fall_root = sim.robot_root_states.clone()
        free_fall_root[:, 2] = 3.0
        free_fall_root[:, 7:13] = 0.0
        for joint, magnitude, limit in (
            ("left_elbow_joint", 50.0, 25.0),
            (LOWER_MODEL_LIMIT_JOINT, 4.0, LOWER_MODEL_LIMIT),
        ):
            assert robot.dof_effort_limit_list[robot.dof_names.index(joint)] == limit
            for sign in (1.0, -1.0):
                command, clipped = sign * magnitude, sign * limit
                sim.robot_root_states[:] = free_fall_root
                sim.set_dof_state_by_names(None, defaults, zero_vel)
                sim.forward_kinematics()
                sim.apply_torques_at_dof(torch.zeros(num_envs, sim.num_dof, device=sim.sim_device))
                sim.apply_torques_by_names({joint: command})
                step(sim, 1)
                applied = torch.stack([sim.get_dof_forces(env_id) for env_id in range(num_envs)])
                expected = torch.zeros_like(applied)
                expected[:, sim.dof_names.index(joint)] = clipped
                print(
                    f"  CLIPPING {joint}: command {command}, applied {applied[:, sim.dof_names.index(joint)].tolist()}"
                )
                if not torch.equal(applied, expected):
                    failures.append(f"command {command} on {joint} did not clip to {clipped} N·m in every environment")
                # Read submitted solver commands independently of actuator telemetry.
                if sim.backend.uses_kit:
                    raw_command = sim._robot.root_view.get_dof_actuation_forces()
                else:
                    from isaaclab_ov import tensor_types as TT

                    raw_command = sim._robot.root_view.get_attribute(TT.DOF_ACTUATION_FORCE)
                raw_effort = wp.to_torch(raw_command)
                raw_joint_ids = [sim._robot.backend_joint_names.index(name) for name in sim.dof_names]
                submitted = raw_effort[:, raw_joint_ids].to(expected.device)
                if not torch.equal(submitted, expected):
                    failures.append(f"PhysX did not receive the clipped {clipped} N·m command on {joint}")
                joint_vel = sim.get_dof_state_by_names([joint])[joint][1]
                print(f"  RESPONSE {joint}: command {command}, velocity {joint_vel.tolist()}")
                if not bool(torch.isfinite(joint_vel).all()) or not bool((joint_vel * clipped > 0).all()):
                    failures.append(f"the clipped {clipped} N·m command did not move {joint} in its direction")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"PASS: {args.simulator} DOF state and torques follow robot_config.dof_names ({num_envs} env(s))")
    if args.result_file:
        Path(args.result_file).write_text("OK\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
