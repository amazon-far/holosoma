"""Headless cross-backend check that simulator state is current at every physics substep.

Settles the robot onto flat ground, then for several control frames compares every tensor the
simulator's state registry serves with a fresh read at ``PRE_STEP`` and ``POST_STEP`` of every
substep: joint state, root state, base orientation, body poses and contact forces through the
backend's own buffers, and the rest by re-running their bound read. A tensor that is only refreshed
at frame end would match at frame boundaries and fail mid-frame. Every served tensor must be float32 on
the simulator's device. Mid-frame it writes state in place and checks body state after FK against an
independent reference (MuJoCo: a CPU ``mj_forward``; IsaacSim: PhysX link poses, bypassing Isaac Lab's
caches), and on every backend that the root body's angular velocity follows a velocity-only write.

Success is reported via an ``OK`` sentinel in ``--result-file`` followed by ``os._exit``
(the _run_harness.py contract, as in contact_substep_assert.py).

Usage:
  python state_freshness_assert.py --simulator mujoco|mjwarp|isaacsim [--num-envs N] [--device D]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

# tests/simulators/ has an isaacsim/ subpackage that would shadow the real IsaacSim package if it
# lands on sys.path[0] when run as a script — drop it (mirrors contact_substep_assert.py).
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from tests.simulators.contact_substep_assert import _build

SETTLE_CONTROL_STEPS = 20
CHECKED_CONTROL_STEPS = 4
ATOL = 1e-5
# A substep must move some body by more than this, or a stale tensor could pass within ATOL.
MIN_SUBSTEP_MOTION = 10 * ATOL
JOINT_OFFSET = 0.02  # rad, written in place mid-frame
JOINT_VEL_OFFSET = 0.1  # rad/s
ROOT_ANG_VEL_OFFSET = 0.3  # rad/s
BODY_ATOL = 1e-4  # independent reference after FK
CONTACT_SLIDE_VELOCITY = 0.5  # m/s; forces ground contacts to produce friction on the native backend


def _fresh_isaacsim(sim):
    data = sim._robot.data
    if sim.backend.is_isaacsim6:
        import torch

        # Read the native data directly, independently of Holosoma's state adapter.
        root = torch.cat((data.root_link_pose_w.torch, data.root_com_vel_w.torch), dim=-1)
        contact_data = sim.contact_sensor.data
        forces = (
            contact_data.net_forces_w.torch
            if sim.simulator_config.sim.isaacsim6.contact_force_mode == "total"
            else contact_data.net_normal_forces_w.torch
        )
        return {
            "dof_pos": data.joint_pos.torch[:, sim.dof_ids],
            "dof_vel": data.joint_vel.torch[:, sim.dof_ids],
            "robot_root_states": root,
            "base_quat": root[:, 3:7],
            "_rigid_body_pos": data.body_link_pos_w.torch[:, sim.body_ids, :],
            "contact_forces": forces[:, sim._contact_to_robot_body_ids],
        }
    root = data.root_state_w.clone()
    root[:, 3:7] = root[:, [4, 5, 6, 3]]  # wxyz -> xyzw
    return {
        "dof_pos": data.joint_pos[:, sim.dof_ids],
        "dof_vel": data.joint_vel[:, sim.dof_ids],
        "robot_root_states": root,
        "base_quat": root[:, 3:7],
        "_rigid_body_pos": data.body_pos_w[:, sim.body_ids, :],
        "contact_forces": sim.contact_sensor.data.net_forces_w[:, sim._contact_to_robot_body_ids],
    }


def _fresh_mujoco(sim):
    import torch

    from holosoma.simulator.mujoco.backends.warp_backend import WarpBackend
    from holosoma.utils.rotations import quat_rotate

    if isinstance(sim.backend, WarpBackend):
        qpos, qvel = sim.backend.qpos_t, sim.backend.qvel_t
    else:
        qpos, qvel = torch.from_numpy(sim.root_data.qpos)[None], torch.from_numpy(sim.root_data.qvel)[None]

    qp, qv = sim.object_addrs["robot"]["qpos_addr"], sim.object_addrs["robot"]["qvel_addr"]
    quat = qpos[:, qp + 3 : qp + 7][:, [1, 2, 3, 0]].float()  # wxyz -> xyzw
    # Freejoint angular velocity is body-local; robot_root_states is world-frame.
    ang_vel = quat_rotate(quat, qvel[:, qv + 3 : qv + 6].float(), w_last=True)
    root = torch.cat([qpos[:, qp : qp + 3].float(), quat, qvel[:, qv : qv + 3].float(), ang_vel], dim=1)
    return {
        "dof_pos": qpos[:, sim.dof_qpos_addrs].clone(),
        "dof_vel": qvel[:, sim.dof_qvel_addrs].clone(),
        "robot_root_states": root,
        "base_quat": root[:, 3:7],
        "_rigid_body_pos": sim.backend.refresh_rigid_body_states()[0].clone(),
        "contact_forces": sim.backend.compute_contact_forces()[:, sim._body_ids_t].clone(),
    }


def _fresh(sim, simulator: str):
    """An independent read of the backend buffers per served name; other names re-run their bound read."""
    fresh = _fresh_isaacsim(sim) if simulator in ("isaacsim", "isaacsim6") else _fresh_mujoco(sim)
    for name in sim.state.names():
        if name not in fresh:
            fresh[name] = sim.state._read_fresh(name)
    return fresh


def _reference_bodies_isaacsim(sim):
    """Env 0's body poses read from PhysX, bypassing Isaac Lab's cached articulation data."""
    if sim.backend.is_isaacsim6:
        import warp as wp

        view = sim._robot.root_view
        if sim.backend.uses_kit:
            from isaaclab_physx.physics import PhysxManager

            PhysxManager.get_physics_sim_view().update_articulations_kinematic()
            poses = view.get_link_transforms()
        else:
            from isaaclab_ov import tensor_types as TT

            # forward_kinematics already flushed the write. Read the solver
            # binding directly, without refreshing Isaac Lab's cached data.
            poses = view.get_attribute(TT.LINK_POSE)
        poses = wp.to_torch(poses)
        # Native find_bodies returns public-order IDs; the raw PhysX view uses its
        # own solver order. Resolve names independently rather than reuse those IDs.
        body_ids = [sim._robot.backend_body_names.index(name) for name in sim.body_names]
    else:
        from isaacsim.core.simulation_manager import SimulationManager

        SimulationManager.get_physics_sim_view().update_articulations_kinematic()
        view = sim._robot.root_physx_view
        body_ids = sim.body_ids
        poses = view.get_link_transforms()
    poses = poses[0, body_ids].cpu()  # xyzw quaternions
    return {"_rigid_body_pos": poses[:, :3], "_rigid_body_rot": poses[:, 3:7]}


def _reference_bodies_mujoco(sim):
    """Env 0's body pose and velocity from an independent CPU ``mj_forward`` of its qpos/qvel."""
    import mujoco
    import numpy as np
    import torch

    from holosoma.simulator.mujoco.backends.warp_backend import WarpBackend

    if isinstance(sim.backend, WarpBackend):
        torch.cuda.synchronize()
        qpos, qvel = sim.backend.qpos_t[0].double().cpu().numpy(), sim.backend.qvel_t[0].double().cpu().numpy()
    else:
        qpos, qvel = sim.root_data.qpos.copy(), sim.root_data.qvel.copy()
    data = mujoco.MjData(sim.root_model)
    data.qpos[:], data.qvel[:] = qpos, qvel
    mujoco.mj_forward(sim.root_model, data)
    ids = list(sim.body_ids)
    velocity = np.zeros((len(ids), 6))
    for i, body in enumerate(ids):
        mujoco.mj_objectVelocity(sim.root_model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity[i], 0)
    return {
        "_rigid_body_pos": data.xpos[ids],
        "_rigid_body_rot": data.xquat[ids][:, [1, 2, 3, 0]],
        "_rigid_body_vel": velocity[:, 3:],
        "_rigid_body_ang_vel": velocity[:, :3],
    }


def _check(sim, simulator: str, *, contact_force_mode: str = "normal") -> tuple[list[str], list[str]]:
    import torch

    from holosoma.simulator.base_simulator.hooks import Phase

    decimation = sim.simulator_config.sim.control_decimation_steps
    worst: dict[str, float] = {}
    failures: list[str] = []
    motions: list[float] = []
    checks = 0
    previous: list[torch.Tensor] = []
    native_contacts = simulator in ("isaacsim", "isaacsim6") and sim.backend.is_isaacsim6
    contact_modes_distinct = False
    contact_mode_difference = 0.0

    device = torch.device(sim.sim_device)
    if native_contacts:
        configured_mode = sim.simulator_config.sim.isaacsim6.contact_force_mode
        if configured_mode != contact_force_mode:
            failures.append(f"requested contact_force_mode={contact_force_mode}, configured {configured_mode}")
        for field in ("joint_pos", "joint_vel", "body_link_pos_w", "body_link_quat_w"):
            proxy = getattr(sim._robot.data, field)
            native = proxy.torch
            exposed = sim.backend.tensor(proxy)
            if (
                exposed.data_ptr() != native.data_ptr()
                or exposed.shape != native.shape
                or exposed.stride() != native.stride()
            ):
                failures.append(f"{field} adapter copied or reshaped the native Torch view")

    def compare(phase: str) -> None:
        nonlocal checks, contact_modes_distinct, contact_mode_difference
        fresh = _fresh(sim, simulator)
        for name, fresh_value in fresh.items():
            checks += 1
            served = getattr(sim, name)[:]
            if served.device != device or served.dtype != torch.float32:
                worst[f"{name} served as {served.dtype} on {served.device}, not float32 on {device}"] = 0.0
            served = served.float()
            expected = fresh_value.to(served)
            if not torch.allclose(served, expected, atol=ATOL):
                error = (served - expected).abs().max().item()
                key = f"{name} stale at {phase}"
                worst[key] = max(worst.get(key, 0.0), error)
        if phase == "POST_STEP":
            if native_contacts:
                # A wrong selector could pass the direct-buffer comparison while both arrays
                # contain zero. Require actual friction evidence after the lateral velocity write.
                contact_data = sim.contact_sensor.data
                normal = contact_data.net_normal_forces_w.torch
                total = contact_data.net_forces_w.torch
                contact_modes_distinct |= not torch.allclose(normal, total, atol=ATOL)
                contact_mode_difference = max(contact_mode_difference, (total - normal).abs().max().item())
            # The root link is the robot's first body: its pose must be the integrated root pose, not the
            # pose the step started from.
            lag = (sim.state.get("_rigid_body_pos")[:, 0] - sim.robot_root_states[:, :3]).abs().max().item()
            if lag > ATOL:
                worst["root body position differs from robot_root_states after a step"] = max(
                    worst.get("root body position differs from robot_root_states after a step", 0.0), lag
                )
            positions = fresh["_rigid_body_pos"].float()
            if previous:
                motions.append((positions - previous[0]).abs().max().item())
            previous[:] = [positions.clone()]

    def check_close(label: str, served: torch.Tensor, expected: torch.Tensor) -> None:
        error = (served.float().cpu() - torch.as_tensor(expected, dtype=torch.float32)).abs().max().item()
        if error > BODY_ATOL:
            failures.append(f"{label} (max error {error:.3e})")

    def write_mid_frame() -> None:
        """Write joint and root state in place, never flushed by hand; the next FK pass must apply them."""
        target = sim.dof_pos[:].clone() + JOINT_OFFSET
        sim.dof_pos[:] = target
        sim.dof_vel[:] += JOINT_VEL_OFFSET
        sim.robot_root_states[:, 10:13] += ROOT_ANG_VEL_OFFSET
        sim.forward_kinematics()
        reached = _fresh(sim, simulator)["dof_pos"].to(target)
        if not torch.allclose(reached, target, atol=ATOL):
            error = (reached - target).abs().max().item()
            failures.append(f"in-place dof_pos write did not reach the simulator (max error {error:.3e})")
        references = (
            _reference_bodies_isaacsim(sim) if simulator in ("isaacsim", "isaacsim6") else _reference_bodies_mujoco(sim)
        )
        for name, reference in references.items():
            expected = torch.as_tensor(reference, dtype=torch.float32)
            served = sim.state.get(name)[0].float().cpu()
            if name == "_rigid_body_rot":  # q and -q are the same rotation
                served = served * torch.sign((served * expected).sum(-1, keepdim=True))
            check_close(f"{name} after FK differs from an independent reference", served, expected)
        compare("FK after an in-place write")

    def write_velocity_only() -> None:
        """A velocity-only write: FK must update body velocities, not only poses."""
        root_ang_vel = sim.robot_root_states[:, 10:13] + ROOT_ANG_VEL_OFFSET
        sim.robot_root_states[:, 10:13] = root_ang_vel
        sim.forward_kinematics()
        # The root link is the robot's first body.
        check_close(
            "root body angular velocity after a velocity-only write and FK",
            sim._rigid_body_ang_vel[:, 0],
            root_ang_vel.cpu(),
        )
        compare("FK after a velocity-only write")

    def control_step(checked: bool, write: bool = False) -> None:
        sim.hooks.emit(Phase.FRAME_BEGIN)
        for substep in range(decimation):
            if checked:
                compare("PRE_STEP")
            if write and substep == 1:
                write_mid_frame()
            if write and substep == 2:
                write_velocity_only()
            sim.hooks.emit(Phase.PRE_STEP)
            sim.simulate_at_each_physics_step()
            sim.hooks.emit(Phase.POST_STEP)
            if checked:
                compare("POST_STEP")

    for _ in range(SETTLE_CONTROL_STEPS):
        control_step(checked=False)
    if native_contacts:
        # Sliding a settled robot produces tangential contact forces. Both mode cells retain
        # the same physical stimulus, so their direct SDK oracles distinguish the selected buffer.
        sim.robot_root_states[:, 7] = CONTACT_SLIDE_VELOCITY
        sim.forward_kinematics()
    for frame in range(CHECKED_CONTROL_STEPS):
        control_step(checked=True, write=frame == 1)

    failures += [f"{key} (max error {error:.3e})" for key, error in sorted(worst.items())]
    slowest = min(motions)
    if slowest <= MIN_SUBSTEP_MOTION:
        failures.append(f"VACUOUS: a substep moved no body by more than {MIN_SUBSTEP_MOTION:g} m ({slowest:.2e})")
    report = [f"decimation={decimation} checks={checks} slowest substep body motion={slowest:.2e} m"]
    if native_contacts:
        if not contact_modes_distinct:
            failures.append("VACUOUS: native normal and total contact forces never differed during the sliding probe")
        report.append(
            f"contact_force_mode={sim.simulator_config.sim.isaacsim6.contact_force_mode} "
            f"maximum total-minus-normal force={contact_mode_difference:.3e} N"
        )
    return failures, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacsim", "isaacsim6"])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--contact-force-mode", choices=["normal", "total"], default="normal")
    parser.add_argument("--result-file", default=None)
    args, _ = parser.parse_known_args()
    sys.argv = [sys.argv[0]]  # AppLauncher re-parses argv

    sim = _build(args.simulator, args.num_envs, args.device, contact_force_mode=args.contact_force_mode)
    failures, report = _check(sim, args.simulator, contact_force_mode=args.contact_force_mode)

    lines = [f"==== state freshness on {args.simulator} ====", *report]
    lines += [f"FAIL {failure}" for failure in failures]
    if not failures:
        lines.append("OK: served state matches a fresh backend read at every substep")
    print("\n".join(lines), flush=True)
    if args.result_file and not failures:
        pathlib.Path(args.result_file).write_text("OK")

    # Hard-exit past teardown (see contact_substep_assert.py).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failures else 0)


if __name__ == "__main__":
    main()
