"""Headless cross-backend assertion harness for ``simulator.contact_recorder``.

Settles the robot onto flat ground, then asserts the buffer's contract over one control step.
The properties are the ones a reward term reading the buffer depends on:

  1. SHAPE      — exactly ``control_decimation_steps`` frames wide, and ``recorded_forces``
                  exposes all of them once the control step completes.
  2. FILLED     — every slot is written; an unwritten slot's zero is indistinguishable from a
                  real zero-force sample.
  3. ORDERED    — slot i is written at substep i, not before, and not touched again.
  4. DISTINCT   — the slots are not all one frame. A tensor not re-fetched mid-step, or a sensor
                  whose update period is coarser than the physics step, yields one sample repeated.
  5. AGREES     — the last slot equals ``simulator.contact_forces`` at the following refresh.
  6. REWRITTEN  — the next control step writes every slot again, which is why the buffer needs no
                  clearing on reset: no pre-reset sample can survive into a reward.
  7. RESTARTS   — after a deliberately short step desyncs the slot index, FRAME_BEGIN re-anchors
                  slot 0 to the first substep of the control step, and ``recorded_forces`` narrows
                  to just that sample rather than exposing the previous step's slots.
  8. STABLE     — ``refresh_sim_tensors`` does not touch the buffer, mid-step or once the step is
                  complete, however many times it is called. It runs a variable number of times per
                  control step (the task reset path calls it a second time; run_sim calls it per
                  physics step on the Isaac backends), which is why recording cannot live there.

"Written" is checked by poisoning the buffer with NaN first, rather than by looking for non-zero
forces: a slot is legitimately zero whenever the body is airborne at that substep.

The robot is unactuated, so it settles into a resting sprawl — sustained ground contact, with
enough solver jitter for consecutive substeps to differ.

Teardown on some backends can kill the process before stdout flushes and can corrupt the exit
code, so success is reported via an ``OK`` sentinel in ``--result-file`` followed by ``os._exit``
(the _run_harness.py contract, as in behavior_assert.py).

Usage:
  python contact_substep_assert.py --simulator mujoco                 # ClassicBackend, 1 env, cpu
  python contact_substep_assert.py --simulator mjwarp   --num-envs 4  # WarpBackend, cuda
  python contact_substep_assert.py --simulator isaacgym --num-envs 4  # IsaacGym, cuda
  python contact_substep_assert.py --simulator isaacsim --num-envs 4  # IsaacSim, cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import sys

# tests/simulators/ has an isaacsim/ subpackage that would shadow the real IsaacSim package if it
# lands on sys.path[0] when run as a script — drop it (mirrors static_move_assert.py).
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators._sim_harness import build_run_sim_config

SETTLE_CONTROL_STEPS = 60
ROBOT = "g1-29dof"
TERRAIN = "terrain_locomotion_plane"


def _build(simulator: str, num_envs: int):
    """Build and prepare a sim with the robot dropped on flat ground."""
    import torch

    config = build_run_sim_config(simulator, "empty", ROBOT, TERRAIN)
    config = dataclasses.replace(
        config,
        device=("cpu" if simulator == "mujoco" else "cuda:0"),
        training=dataclasses.replace(config.training, num_envs=num_envs, headless=True),
    )

    env, device, _ = setup_simulation_environment(config, device=config.device)
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    origins = torch.zeros(num_envs, 3, device=device)
    if num_envs > 1:
        origins[:, 0] = torch.arange(num_envs, device=device, dtype=torch.float32) * 8.0
    init_state = config.robot.init_state
    base_init = torch.tensor(
        list(init_state.pos) + list(init_state.rot) + list(init_state.lin_vel) + list(init_state.ang_vel), device=device
    )
    sim.create_envs(num_envs, origins, base_init)
    sim.prepare_sim()
    return sim


def _check(sim, num_envs: int) -> tuple[list[str], list[str]]:
    """Run the checks. Returns (failures, report lines)."""
    import torch

    from holosoma.simulator.base_simulator.hooks import Phase

    decimation = sim.simulator_config.sim.control_decimation_steps
    recorder = sim.contact_recorder
    buffer = recorder.buffer  # never reallocated, so one binding stays valid
    failures = []

    # The recorder records on POST_STEP, which the real loops (base_task, run_sim) emit right after
    # the physics step; this harness drives the simulator directly, so it emits the phase itself.
    def physics_step():
        sim.simulate_at_each_physics_step()
        sim.hooks.emit(Phase.POST_STEP)

    def control_step():
        sim.hooks.emit(Phase.FRAME_BEGIN)
        for _ in range(decimation):
            physics_step()
        sim.refresh_sim_tensors()

    want = (num_envs, decimation, sim.num_bodies, 3)
    if tuple(buffer.shape) != want:
        failures.append(f"SHAPE: {tuple(buffer.shape)} != {want}")

    for _ in range(SETTLE_CONTROL_STEPS):
        control_step()

    if tuple(recorder.recorded_forces.shape) != want:
        failures.append(f"SHAPE: recorded_forces {tuple(recorder.recorded_forces.shape)} != {want} after a full step")

    # Poison every slot first, so "written this control step" is exact (not "happens to be
    # non-zero"): a slot CAN legitimately be zero when the body is airborne, which is precisely the
    # zero/no-data conflation this buffer exists to avoid.
    buffer.fill_(float("nan"))

    def refresh_is_stable(when: str) -> None:
        """Clone the live buffer across one refresh. Must read `buffer`, not a snapshot: comparing
        two clones (as AGREES does) cannot see a refresh that mutates the buffer in place."""
        before = buffer.clone()
        sim.refresh_sim_tensors()
        # Compare the poison mask and the written values separately: nan_to_num would map a refresh
        # that ZEROED an unwritten slot onto the NaN it replaced, hiding the mutation.
        poisoned = before.isnan()
        if not torch.equal(poisoned, buffer.isnan()) or not torch.equal(before[~poisoned], buffer[~poisoned]):
            failures.append(f"STABLE: refresh_sim_tensors mutated the buffer ({when})")

    # One control step, snapshotting after every substep, so ORDERED can see when each slot moves.
    sim.hooks.emit(Phase.FRAME_BEGIN)
    snapshots = []
    for substep in range(decimation):
        physics_step()
        if substep == 0:
            refresh_is_stable("mid-step, most slots still unwritten")
        snapshots.append(buffer.clone())
    # The completed buffer is what a reward term reads, and the task reset path refreshes a second
    # time in the same control step (locomotion_manager, wbt_manager) — so check both calls.
    refresh_is_stable("step complete")
    refresh_is_stable("step complete, second refresh")

    for i in range(decimation):
        written_at_i = not snapshots[i][:, i].isnan().any()
        if not written_at_i:
            failures.append(f"ORDERED: slot {i} was not written at substep {i}")
        if i > 0 and not snapshots[i - 1][:, i].isnan().all():
            failures.append(f"ORDERED: slot {i} was written before substep {i}")
        for later in range(i + 1, decimation):
            if not torch.equal(snapshots[i][:, i], snapshots[later][:, i]):
                failures.append(f"ORDERED: slot {i} changed again at substep {later}")
                break

    final = snapshots[-1]
    force_magnitude = final.norm(dim=-1).sum(dim=-1)  # [num_envs, decimation]

    if bool(final.isnan().any()):
        failures.append(f"FILLED: some slot went unwritten; per-slot magnitude {force_magnitude.tolist()}")

    # Not "every consecutive pair differs": a fully settled robot could legitimately produce two
    # bitwise-equal substeps. The defect this guards against makes the WHOLE step one frame.
    distinct = sum(1 for i in range(decimation - 1) if not torch.equal(final[:, i], final[:, i + 1]))
    if decimation > 1 and distinct == 0:
        failures.append(f"DISTINCT: all {decimation} slots identical — the same frame recorded every substep")

    # Tautological on backends that record contact_forces itself; it still pins IsaacSim's
    # separate sensor-to-robot body-order gather against the one refresh_sim_tensors uses.
    if not torch.allclose(final[:, decimation - 1], sim.contact_forces, atol=1e-4):
        max_abs_diff = (final[:, decimation - 1] - sim.contact_forces).abs().max()
        failures.append(f"AGREES: last slot != contact_forces at refresh (max abs diff {float(max_abs_diff):.4g})")

    buffer.fill_(float("nan"))
    control_step()
    if bool(buffer.isnan().any()):
        failures.append("REWRITTEN: a slot survived unwritten into the next control step")

    # RESTARTS — deliberately desync the slot index with a short step, then check FRAME_BEGIN
    # re-anchors slot 0 to the first substep. Without this the index only stays phase-aligned
    # because the decimation divides evenly, so a dead frame-begin hook would go unnoticed.
    if decimation > 1:
        for _ in range(decimation - 1):
            physics_step()
        buffer.fill_(float("nan"))
        sim.hooks.emit(Phase.FRAME_BEGIN)
        physics_step()
        if bool(buffer[:, 0].isnan().any()):
            failures.append("RESTARTS: FRAME_BEGIN did not re-anchor the next substep to slot 0")
        for i in range(1, decimation):
            if not bool(buffer[:, i].isnan().all()):
                failures.append(f"RESTARTS: slot {i} was written instead of slot 0 after FRAME_BEGIN")
                break
        # The poisoned slots are still in the allocation, so a mid-step read must not reach them.
        if tuple(recorder.recorded_forces.shape)[1] != 1:
            failures.append(f"RESTARTS: recorded_forces is {recorder.recorded_forces.shape[1]} wide after 1 substep")

    report = [
        f"num_envs={num_envs} decimation={decimation} num_bodies={sim.num_bodies}",
        f"per-slot |force| (env 0): {[round(v, 2) for v in force_magnitude[0].tolist()]}",
        f"distinct consecutive slot pairs: {distinct}/{max(decimation - 1, 0)}",
    ]
    return failures, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--result-file", default=None)
    args, _ = parser.parse_known_args()
    sys.argv = [sys.argv[0]]  # AppLauncher re-parses argv

    sim = _build(args.simulator, args.num_envs)
    failures, report = _check(sim, args.num_envs)

    lines = [f"==== contact_recorder on {args.simulator} ====", *report]
    lines += [f"FAIL {failure}" for failure in failures]
    if not failures:
        lines.append("OK: shape, filled, ordered, distinct, agrees, rewritten, restarts, stable")
    print("\n".join(lines), flush=True)
    if args.result_file and not failures:
        pathlib.Path(args.result_file).write_text("OK")

    # Hard-exit past teardown: IsaacLab's stop handler re-enters render() and wedges there, so a
    # normal return (or SimulationApp.close()) never reaches process exit and the runner would
    # time out on a passing run.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failures else 0)


if __name__ == "__main__":
    main()
