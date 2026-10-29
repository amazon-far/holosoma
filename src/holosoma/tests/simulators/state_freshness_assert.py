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
    env_ids = torch.arange(sim.num_envs, device=sim.sim_device)
    root = sim.get_actor_states(["robot"], env_ids)
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
    body_z: list[float] = []
    checks = 0

    def compare(phase: str) -> None:
        nonlocal checks
        fresh = _fresh(sim, simulator)
        for name, fresh_value in fresh.items():
            checks += 1
            served = sim.state.get(name)[:].float()
            expected = fresh_value.to(served)
            if not torch.allclose(served, expected, atol=1e-5):
                error = (served - expected).abs().max().item()
                key = f"{name} stale at {phase}"
                worst[key] = max(worst.get(key, 0.0), error)
        body_z.append(float(fresh["_rigid_body_pos"][0, 0, 2]))

    def control_step(checked: bool) -> None:
        sim.hooks.emit(Phase.FRAME_BEGIN)
        for _ in range(decimation):
            if checked:
                compare("PRE_STEP")
            sim.hooks.emit(Phase.PRE_STEP)
            sim.simulate_at_each_physics_step()
            sim.hooks.emit(Phase.POST_STEP)
            if checked:
                compare("POST_STEP")
        sim.refresh_sim_tensors()

    for _ in range(SETTLE_CONTROL_STEPS):
        control_step(checked=False)
    for _ in range(CHECKED_CONTROL_STEPS):
        control_step(checked=True)

    failures = [f"{key} (max error {error:.3e})" for key, error in sorted(worst.items())]
    # Without motion between substeps a frame-end snapshot would pass, so require it.
    if len(set(body_z)) < decimation:
        failures.append(f"VACUOUS: the pelvis barely moved between substeps ({len(set(body_z))} distinct heights)")
    live = [name for name in sim.state.names() if sim.state.is_live(name)]
    report = [
        f"decimation={decimation} checks={checks} distinct pelvis heights={len(set(body_z))}",
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
