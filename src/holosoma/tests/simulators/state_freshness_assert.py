"""Headless cross-backend check that simulator state is current at every physics substep.

Settles the robot onto flat ground, then for several control frames compares every tensor the
simulator's state registry serves with a fresh read at ``PRE_STEP`` and ``POST_STEP`` of every
substep: joint state, root state, base orientation, body poses and contact forces through the
backend's own buffers, and the rest by re-running their bound read. A
tensor that is only refreshed at frame end would match at frame boundaries and fail mid-frame.

Success is reported via an ``OK`` sentinel in ``--result-file`` followed by ``os._exit``
(the _run_harness.py contract, as in contact_substep_assert.py).

Usage:
  python state_freshness_assert.py --simulator mujoco|mjwarp|isaacsim [--num-envs N]
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


def _fresh_isaacsim(sim):
    data = sim._robot.data
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

    if isinstance(sim.backend, WarpBackend):
        qpos, qvel = sim.backend.qpos_t, sim.backend.qvel_t
    else:
        qpos, qvel = torch.from_numpy(sim.root_data.qpos)[None], torch.from_numpy(sim.root_data.qvel)[None]
    from holosoma.utils.rotations import quat_rotate

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
    fresh = _fresh_isaacsim(sim) if simulator == "isaacsim" else _fresh_mujoco(sim)
    for name in sim.state.names():
        if name not in fresh:
            fresh[name] = sim.state.read_fresh(name)
    return fresh


def _check(sim, simulator: str) -> tuple[list[str], list[str]]:
    import torch

    from holosoma.simulator.base_simulator.hooks import Phase

    decimation = sim.simulator_config.sim.control_decimation_steps
    worst: dict[str, float] = {}
    failures: list[str] = []
    motions: list[float] = []
    checks = 0
    previous: list[torch.Tensor] = []

    def compare(phase: str) -> None:
        nonlocal checks
        fresh = _fresh(sim, simulator)
        for name, fresh_value in fresh.items():
            checks += 1
            served = sim.state.get(name)[:].float()
            expected = fresh_value.to(served)
            if not torch.allclose(served, expected, atol=ATOL):
                error = (served - expected).abs().max().item()
                key = f"{name} stale at {phase}"
                worst[key] = max(worst.get(key, 0.0), error)
        if phase == "POST_STEP":
            positions = fresh["_rigid_body_pos"].float()
            if previous:
                motions.append((positions - previous[0]).abs().max().item())
            previous[:] = [positions.clone()]

    def write_mid_frame() -> None:
        """Write joint positions in place, never flushed by hand; the next FK pass must apply them."""
        target = sim.dof_pos[:].clone() + JOINT_OFFSET
        sim.dof_pos[:] = target
        sim.forward_kinematics()
        reached = _fresh(sim, simulator)["dof_pos"].to(target)
        if not torch.allclose(reached, target, atol=ATOL):
            error = (reached - target).abs().max().item()
            failures.append(f"in-place dof_pos write did not reach the simulator (max error {error:.3e})")
        compare("FK after an in-place write")

    def control_step(checked: bool, write: bool = False) -> None:
        sim.hooks.emit(Phase.FRAME_BEGIN)
        for substep in range(decimation):
            if checked:
                compare("PRE_STEP")
            if write and substep == 1:
                write_mid_frame()
            sim.hooks.emit(Phase.PRE_STEP)
            sim.simulate_at_each_physics_step()
            sim.hooks.emit(Phase.POST_STEP)
            if checked:
                compare("POST_STEP")
        sim.refresh_sim_tensors()

    for _ in range(SETTLE_CONTROL_STEPS):
        control_step(checked=False)
    for frame in range(CHECKED_CONTROL_STEPS):
        control_step(checked=True, write=frame == 1)

    failures += [f"{key} (max error {error:.3e})" for key, error in sorted(worst.items())]
    # Without motion between substeps a frame-end snapshot would pass, so require it.
    slowest = min(motions)
    if slowest <= MIN_SUBSTEP_MOTION:
        failures.append(f"VACUOUS: a substep moved no body by more than {MIN_SUBSTEP_MOTION:g} m ({slowest:.2e})")
    live = [name for name in sim.state.names() if sim.state.is_live(name)]
    report = [
        f"decimation={decimation} checks={checks} slowest substep body motion={slowest:.2e} m",
        f"live views: {', '.join(live) or 'none'}",
    ]
    return failures, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacsim"])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--result-file", default=None)
    args, _ = parser.parse_known_args()
    sys.argv = [sys.argv[0]]  # AppLauncher re-parses argv

    sim = _build(args.simulator, args.num_envs)
    failures, report = _check(sim, args.simulator)

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
