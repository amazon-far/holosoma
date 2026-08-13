"""Probe: a NON-CONTIGUOUS env_ids subset write to set_static_body_pose must not scramble poses.

Builds the same freebox+pillar scene as static_move_assert.py at num_envs=4, settles, then relocates
the pillar ONLY in envs [0, 2] to PER-ENV-DISTINCT world targets. Asserts:
  (a) WRITTEN envs [0,2] each land at their OWN distinct target (no transposition),
  (b) UNWRITTEN envs [1,3] are UNTOUCHED (still at the spawn pose),
exercising the index encode/decode (get_object_indices / resolve_indices / write_object_states)
for a strided subset rather than the full arange.

Exits 0 on success, non-zero with a message on failure. Run under each backend's launcher.

Usage:
  python subset_write_assert.py --simulator isaacsim
  python subset_write_assert.py --simulator isaacgym
  python subset_write_assert.py --simulator mjwarp
"""

from __future__ import annotations

import argparse
import sys

if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

# Reuse static_move_assert's scene/build/step helpers so the scene is identical to the passing test.
from holosoma.simulator.shared.object_registry import ObjectType
from tests.simulators.static_move_assert import (
    PILLAR_Z,
    SETTLE_SECONDS,
    _build,
    _step,
    steps_for_seconds,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-contiguous env_ids subset write probe.")
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--num-envs", type=int, default=4)
    args = parser.parse_args()
    num_envs = args.num_envs
    if num_envs < 4:
        print("FAIL: subset probe needs num_envs>=4 (write a strided subset, leave others)")
        return 1

    import torch

    sim, origins = _build(args.simulator, num_envs)
    free = sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
    static = sim.object_registry.get_names_by_type(ObjectType.SCENE)
    print(f"[{args.simulator}] free={free} static={static} sim_dt={sim.sim_dt}")
    if "pillar" not in static:
        print("FAIL: scene did not register pillar(SCENE)")
        return 1

    all_ids = torch.arange(num_envs, device=sim.sim_device)
    settle = steps_for_seconds(sim, SETTLE_SECONDS)
    _step(sim, settle)

    # Spawn (world) pillar pose per env: x = origin_x + BOX_X column, z = origin_z + PILLAR_Z.
    spawn = sim.get_actor_states(["pillar"], all_ids)[:, :3].clone()  # [num_envs, 3] WORLD
    print(f"  spawn pillar x={[round(float(v), 3) for v in spawn[:, 0]]}")

    # Relocate ONLY envs [0, 2] to per-env-distinct world x targets; leave [1, 3] alone.
    subset = torch.tensor([0, 2], device=sim.sim_device)
    # Per-env-distinct world targets: env0 -> spawn_x[0]+10, env2 -> spawn_x[2]+20 (distinct deltas
    # so a transposition between the two subset envs would be caught).
    deltas = torch.tensor([10.0, 20.0], device=sim.sim_device)
    poses = torch.zeros(len(subset), 7, device=sim.sim_device, dtype=torch.float32)
    poses[:, :3] = spawn[subset]
    poses[:, 0] += deltas
    poses[:, 2] = origins[subset, 2] + PILLAR_Z  # keep it kinematic-stable at pillar height
    poses[:, 6] = 1.0  # identity quat (xyzw)
    sim.set_static_body_pose(["pillar"], subset, poses)

    after = sim.get_actor_states(["pillar"], all_ids)[:, :3]  # [num_envs, 3] WORLD
    print(f"  after pillar x={[round(float(v), 3) for v in after[:, 0]]}")

    # (a) WRITTEN envs [0,2] each at their OWN target.
    want0 = float(spawn[0, 0] + 10.0)
    want2 = float(spawn[2, 0] + 20.0)
    got0, got2 = float(after[0, 0]), float(after[2, 0])
    written_ok = abs(got0 - want0) < 2e-2 and abs(got2 - want2) < 2e-2

    # (b) UNWRITTEN envs [1,3] untouched (still at spawn x).
    unwritten_ok = (
        abs(float(after[1, 0]) - float(spawn[1, 0])) < 2e-2 and abs(float(after[3, 0]) - float(spawn[3, 0])) < 2e-2
    )

    print(f"  WRITTEN env0 {got0:.3f} (want {want0:.3f}) env2 {got2:.3f} (want {want2:.3f}) ok={written_ok}")
    print(
        f"  UNWRITTEN env1 {float(after[1, 0]):.3f} (spawn {float(spawn[1, 0]):.3f}) "
        f"env3 {float(after[3, 0]):.3f} (spawn {float(spawn[3, 0]):.3f}) ok={unwritten_ok}"
    )

    if not written_ok:
        print("FAIL: subset write did not place written envs at their distinct targets (SCRAMBLE)")
        return 1
    if not unwritten_ok:
        print("FAIL: subset write disturbed envs NOT in the subset (SCRAMBLE)")
        return 1
    print("PASS: non-contiguous subset write is correctly indexed (no scramble)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
