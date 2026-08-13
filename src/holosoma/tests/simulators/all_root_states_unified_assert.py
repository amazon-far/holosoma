"""Cross-backend assertion harness: ``all_root_states`` spans robot AND objects, uniformly.

``sim.all_root_states[sim.get_actor_indices(names, env_ids)]`` gathers/scatters the unified
13-vector world-frame state for any actor — robot, free object, or static scene body — on every
backend, and duck-types like a tensor for the ``[indices]`` / ``[indices, cols]`` / ``.shape`` /
``.clone()`` surface.

Builds the static_move scene (g1 robot + free box + static pillar) and asserts, for names
including BOTH ``"robot"`` and the free object, in EVERY env:

  1. INDICES — get_actor_indices(["robot", "freebox"], env_ids) has the expected length and the
     robot rows resolve to robot_root_states.
  2. READ PARITY — all_root_states[idx] equals get_actor_states(names, env_ids) for pose columns,
     and the robot slice equals robot_root_states[env_ids] (pose).
  3. WRITE ROUNDTRIP — scatter a perturbed pose for robot+object via all_root_states[idx]=...,
     flush, and read it back through get_actor_states.
  4. PARTIAL-COLUMN WRITE — all_root_states[idx, 0:3] = new_xyz changes only position, leaving
     the quaternion intact.
  5. CLONE / SHAPE — .clone() returns [total_actors, 13] matching .shape and equals a full-range
     gather.
  6. WHOLESALE-PASS INVARIANCE — set_actor_root_state_tensor(env_ids, all_root_states) moves only
     the robot (the object is untouched).
  7. UNIFIED set_actor_states — set_actor_states(["robot", "freebox"], ...) (the method, not the
     proxy) writes robot AND object uniformly in WORLD frame; both move, no origin double-count.
     This is the path the old IsaacGym "Cannot set 'robot' state" guard rejected.
  8. WORLD-FRAME VELOCITY ROUND-TRIP — a known world-frame LINEAR and ANGULAR velocity written via
     set_actor_states reads back (after a few free-flight cache-refresh steps; x/y conserved, z by
     gravity) through get_actor_states for robot AND object. Guards the freejoint
     body-local<->world angular conversion for the robot (not just objects) AND that linear
     velocity round-trips in world frame for both — linear is asserted, not dropped.

Exits 0 on success, non-zero with a message on failure. Run under each backend's launcher.

Usage:
  python all_root_states_unified_assert.py --simulator mujoco                 # ClassicBackend, 1 env, cpu
  python all_root_states_unified_assert.py --simulator mjwarp   --num-envs 4  # WarpBackend, cuda
  python all_root_states_unified_assert.py --simulator isaacgym --num-envs 4  # IsaacGym, cuda
  python all_root_states_unified_assert.py --simulator isaacsim --num-envs 4  # IsaacSim, cuda
"""

from __future__ import annotations

import argparse
import sys

# tests/simulators/ has an isaacsim/ subpackage that would shadow the real IsaacSim package if it
# lands on sys.path[0] when run as a script — drop it (mirrors the sibling harnesses).
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

# Reuse static_move_assert's scene/build/step helpers so the scene (robot + freebox + pillar) is
# identical to the passing static-move test.
from holosoma.simulator.shared.object_registry import ObjectType
from tests.simulators.static_move_assert import (
    SETTLE_SECONDS,
    _build,
    _step,
    steps_for_seconds,
)

POS_ATOL = 2e-2  # world-position tolerance (m)
QUAT_ATOL = 2e-2  # quaternion-component tolerance
# Steps to advance after a write before reading back. MuJoCo (mj_forward in write_state_updates)
# and IsaacGym (the read tensor is the live buffer) reflect writes immediately; IsaacSim refreshes
# its read cache (.data.root_state_w) from PhysX on a sim step. A zero-velocity body drifts
# ~5e-4 m over 2 steps, well within POS_ATOL.
STEPS_AFTER_WRITE = 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified all_root_states (robot+object) probe.")
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--num-envs", type=int, default=None)
    args = parser.parse_args()
    num_envs = args.num_envs if args.num_envs is not None else (1 if args.simulator == "mujoco" else 4)

    import torch

    sim, _origins = _build(args.simulator, num_envs)
    free = sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
    static = sim.object_registry.get_names_by_type(ObjectType.SCENE)
    print(f"[{args.simulator}] free={free} static={static} num_envs={num_envs} sim_dt={sim.sim_dt}")
    if "freebox" not in free:
        print("FAIL: scene did not register freebox(INDIVIDUAL)")
        return 1

    env_ids = torch.arange(num_envs, device=sim.sim_device)
    # Settle so the box is at rest (a deterministic live state to read back).
    _step(sim, steps_for_seconds(sim, SETTLE_SECONDS))

    names = ["robot", "freebox"]
    n_actors = len(names) * num_envs

    # ---- 1. INDICES ------------------------------------------------------------------------
    idx = sim.get_actor_indices(names, env_ids)
    if idx.shape[0] != n_actors:
        print(f"FAIL: get_actor_indices returned {idx.shape[0]} rows, expected {n_actors}")
        return 1
    robot_idx = sim.get_actor_indices(["robot"], env_ids)
    if robot_idx.shape[0] != num_envs:
        print(f"FAIL: get_actor_indices(['robot']) returned {robot_idx.shape[0]} rows, expected {num_envs}")
        return 1
    print(f"  INDICES ok: idx.len={idx.shape[0]} robot_idx.len={robot_idx.shape[0]}")

    # ---- 1b. LIST INDEX COERCION -----------------------------------------------------------
    # all_root_states accepts a python list of ints (coerced to a tensor), not only a tensor.
    list_idx = idx.tolist()
    list_read = sim.all_root_states[list_idx]
    list_ok = list_read.shape == (n_actors, 13) and torch.allclose(
        list_read[:, :7], sim.all_root_states[idx][:, :7], atol=POS_ATOL
    )
    print(f"  LIST-INDEX list==tensor(pose)={bool(list_ok)}")
    if not list_ok:
        print("FAIL: all_root_states[list] did not match all_root_states[tensor]")
        return 1

    # ---- 2. READ PARITY --------------------------------------------------------------------
    via_proxy = sim.all_root_states[idx]  # [n_actors, 13]
    via_byname = sim.get_actor_states(names, env_ids)  # [n_actors, 13]
    if via_proxy.shape != (n_actors, 13):
        print(f"FAIL: all_root_states[idx] shape {tuple(via_proxy.shape)} != ({n_actors}, 13)")
        return 1
    read_pose_ok = torch.allclose(via_proxy[:, :7], via_byname[:, :7], atol=POS_ATOL)
    # Robot rows (name-major: first num_envs rows) match robot_root_states pose.
    robot_via_proxy = sim.all_root_states[robot_idx]  # [num_envs, 13]
    robot_ref_pose = sim.robot_root_states[env_ids][:, :7]
    robot_read_ok = torch.allclose(robot_via_proxy[:, :7], robot_ref_pose, atol=POS_ATOL)
    print(
        f"  READ-PARITY proxy==by_name(pose)={bool(read_pose_ok)} robot==robot_root_states(pose)={bool(robot_read_ok)}"
    )
    if not (read_pose_ok and robot_read_ok):
        print("FAIL: all_root_states read did not match the by-name / robot_root_states paths")
        return 1

    # ---- 3. WRITE ROUNDTRIP (robot + object) ----------------------------------------------
    # Nudge every actor +0.05 m in z via the unified proxy scatter, then read back by name.
    before = sim.get_actor_states(names, env_ids).clone()
    perturbed = before.clone()
    perturbed[:, 2] += 0.05  # raise z
    perturbed[:, 7:] = 0.0  # zero velocities for a clean static read-back
    sim.all_root_states[idx] = perturbed
    sim.write_state_updates()
    _step(sim, STEPS_AFTER_WRITE)  # refresh the read cache (see STEPS_AFTER_WRITE)
    after = sim.get_actor_states(names, env_ids)
    write_ok = torch.allclose(after[:, :3], perturbed[:, :3], atol=POS_ATOL)
    # Specifically confirm the ROBOT row moved (name-major first block).
    robot_after = after[:num_envs]
    robot_before = before[:num_envs]
    robot_moved = bool(((robot_after[:, 2] - robot_before[:, 2]) > 0.02).all())
    print(f"  WRITE-ROUNDTRIP pos_ok={bool(write_ok)} robot_z_moved={robot_moved}")
    if not (write_ok and robot_moved):
        print("FAIL: unified write via all_root_states[idx]=... did not reach the sim for robot+object")
        return 1

    # ---- 4. PARTIAL-COLUMN WRITE (position only, quat preserved) ---------------------------
    pre = sim.get_actor_states(names, env_ids).clone()
    new_xyz = pre[:, :3].clone()
    new_xyz[:, 0] += 0.10  # shift x only
    sim.all_root_states[idx, 0:3] = new_xyz
    sim.write_state_updates()
    _step(sim, STEPS_AFTER_WRITE)  # refresh the read cache (see STEPS_AFTER_WRITE)
    post = sim.get_actor_states(names, env_ids)
    x_ok = torch.allclose(post[:, 0], new_xyz[:, 0], atol=POS_ATOL)
    quat_ok = torch.allclose(post[:, 3:7], pre[:, 3:7], atol=QUAT_ATOL)
    print(f"  PARTIAL-COLUMN x_moved={bool(x_ok)} quat_preserved={bool(quat_ok)}")
    if not (x_ok and quat_ok):
        print("FAIL: partial-column write changed the wrong columns (column_slice not honored)")
        return 1

    # ---- 5. CLONE / SHAPE ------------------------------------------------------------------
    total_actors = len(sim.object_registry.objects) * num_envs
    shape = sim.all_root_states.shape
    if tuple(shape) != (total_actors, 13):
        print(f"FAIL: all_root_states.shape {tuple(shape)} != ({total_actors}, 13)")
        return 1
    cloned = sim.all_root_states.clone()
    full = sim.all_root_states[torch.arange(total_actors, device=sim.sim_device)]
    clone_ok = cloned.shape == full.shape and torch.allclose(cloned[:, :7], full[:, :7], atol=POS_ATOL)
    print(f"  CLONE/SHAPE shape={tuple(shape)} clone==full_gather(pose)={bool(clone_ok)}")
    if not clone_ok:
        print("FAIL: clone() did not match a full-range gather")
        return 1

    # ---- 6. WHOLESALE-PASS INVARIANCE ------------------------------------------------------
    # set_actor_root_state_tensor(env_ids, all_root_states) must move only the robot: raise the
    # robot z via robot_root_states, push the wholesale tensor, confirm the object is untouched.
    obj_idx = sim.get_actor_indices(["freebox"], env_ids)
    obj_before = sim.all_root_states[obj_idx][:, :3].clone()
    sim.robot_root_states[env_ids, 2] += 0.07
    sim.set_actor_root_state_tensor(env_ids, sim.all_root_states)
    sim.write_state_updates()
    obj_after = sim.all_root_states[obj_idx][:, :3]
    obj_untouched = torch.allclose(obj_after, obj_before, atol=POS_ATOL)
    print(f"  WHOLESALE-PASS object_untouched={bool(obj_untouched)}")
    if not obj_untouched:
        print("FAIL: wholesale set_actor_root_state_tensor(all_root_states) disturbed the object")
        return 1

    # ---- 7. UNIFIED set_actor_states(robot + object) ---------------------------------------
    # The unified set_actor_states method (NOT the proxy) must accept the ROBOT alongside an
    # object in one name-major call and write both straight through in WORLD frame. This is the
    # path the old IsaacGym "Cannot set 'robot' state" guard rejected; it now writes the robot's
    # world state straight to the world-frame buffer (no env_origins re-add), exactly like an
    # object. Pose round-trip catches an origin double-count / robot-vs-object asymmetry.
    base = sim.get_actor_states(names, env_ids).clone()  # name-major: [robot envs..., freebox envs...]
    target = base.clone()
    target[:, 2] += 0.06  # raise z for every actor (robot AND object)
    target[:, 7:] = 0.0  # zero velocity for a clean static read-back
    sim.set_actor_states(names, env_ids, target)  # <-- the method whose guard we removed
    sim.write_state_updates()
    _step(sim, STEPS_AFTER_WRITE)
    got = sim.get_actor_states(names, env_ids)
    set_pos_ok = torch.allclose(got[:, :3], target[:, :3], atol=POS_ATOL)
    robot_block, obj_block = got[:num_envs], got[num_envs:]
    base_robot, base_obj = base[:num_envs], base[num_envs:]
    robot_set_moved = bool(((robot_block[:, 2] - base_robot[:, 2]) > 0.02).all())
    obj_set_moved = bool(((obj_block[:, 2] - base_obj[:, 2]) > 0.02).all())
    print(
        f"  UNIFIED-SET-ACTOR-STATES pos_ok={bool(set_pos_ok)} "
        f"robot_moved={robot_set_moved} object_moved={obj_set_moved}"
    )
    if not (set_pos_ok and robot_set_moved and obj_set_moved):
        print("FAIL: set_actor_states(['robot','freebox'], ...) did not write robot+object uniformly in world frame")
        return 1

    # ---- 8. WORLD-FRAME VELOCITY ROUND-TRIP (robot + object, BOTH linear and angular) ------
    # The unified 13-vector contract is WORLD-frame for BOTH linear AND angular velocity on every
    # backend. We assert BOTH and EXACTLY (the freejoint body-local<->world angular conversion is
    # the headline, but linear must round-trip too — do NOT silently drop it). MuJoCo stores
    # freejoint angular velocity body-local, so the angular check specifically guards that
    # conversion FOR THE ROBOT, not just objects.
    #
    # Read-back timing (verified on-cluster, all 4 backends):
    #   - MuJoCo (classic + warp) and IsaacGym reflect a set_actor_states velocity write IMMEDIATELY
    #     (pre-step) for BOTH robot and object — so we read back with no step and compare EXACTLY,
    #     no dynamics/gravity drift to mask a frame bug. (Note: IsaacGym has a known one-step read
    #     latency where a free object's velocity transiently reads 0 at exactly step 1 then recovers
    #     from step 2 — irrelevant here since we read pre-step, but it's why we do NOT step on the
    #     immediate path.)
    #   - IsaacSim refreshes its read cache (.data.root_state_w) from PhysX only on a sim step, so a
    #     pre-step read is stale. For IsaacSim only, lift the actors clear of the ground (free
    #     flight) and step a couple times. After a physics step the velocity is no longer the exact
    #     written value (dynamics integrate), so for IsaacSim we verify the FRAME rather than an
    #     exact value: x/y are conserved (a body-local<->world frame bug WOULD corrupt the
    #     horizontal components, so this still catches a frame error) and z decreases under gravity
    #     (sign + lower-bounded magnitude), while angular is ~conserved for a torque-free rigid body.
    is_isaacsim = sim.get_simulator_type().value == "isaacsim"
    world_lin = torch.tensor([0.5, -0.3, 0.0], device=sim.sim_device)  # z=0 so gravity is the only z term
    world_ang = torch.tensor([0.4, 0.5, -0.6], device=sim.sim_device)
    vel_state = sim.get_actor_states(names, env_ids).clone()
    if is_isaacsim:
        vel_state[:, 2] += 1.5  # lift clear of the ground -> free flight over the refresh steps
    vel_state[:, 7:10] = world_lin
    vel_state[:, 10:13] = world_ang
    sim.set_actor_states(names, env_ids, vel_state)
    sim.write_state_updates()

    if is_isaacsim:
        REFRESH_STEPS = 2  # clear IsaacSim's PhysX read-cache latency (introduces small dynamics drift)
        _step(sim, REFRESH_STEPS)
        LIN_ATOL, ANG_ATOL = 3e-2, 3e-2
        vel_back = sim.get_actor_states(names, env_ids)
        # Frame check: x/y conserved; z fell under gravity (bounded by g*dt*steps, sign negative).
        max_z_drop = 9.81 * float(sim.sim_dt) * REFRESH_STEPS + 2e-2
        lin_xy_ok = torch.allclose(vel_back[:, 7:9], world_lin[:2].expand(n_actors, 2), atol=LIN_ATOL)
        lin_z_ok = bool(((vel_back[:, 9] <= 1e-3) & (vel_back[:, 9] >= -max_z_drop)).all())
        robot_lin_ok = torch.allclose(vel_back[:num_envs, 7:9], world_lin[:2].expand(num_envs, 2), atol=LIN_ATOL)
    else:
        # MuJoCo (classic + warp) and IsaacGym reflect the velocity write IMMEDIATELY -> read back
        # with NO step and require the write to round-trip EXACTLY (no dynamics/gravity to mask a bug).
        LIN_ATOL, ANG_ATOL = 1e-3, 1e-3
        vel_back = sim.get_actor_states(names, env_ids)
        lin_xy_ok = torch.allclose(vel_back[:, 7:10], world_lin.expand(n_actors, 3), atol=LIN_ATOL)
        lin_z_ok = bool(lin_xy_ok)  # z is part of the exact 3-vector compare above
        robot_lin_ok = torch.allclose(vel_back[:num_envs, 7:10], world_lin.expand(num_envs, 3), atol=LIN_ATOL)

    ang_rt_ok = torch.allclose(vel_back[:, 10:13], world_ang.expand(n_actors, 3), atol=ANG_ATOL)
    # Explicitly include the robot rows (name-major first block) so a robot-only frame bug is caught.
    robot_ang_ok = torch.allclose(vel_back[:num_envs, 10:13], world_ang.expand(num_envs, 3), atol=ANG_ATOL)
    lin_max_xy_err = (vel_back[:, 7:9] - world_lin[:2]).abs().max().item()
    ang_max_err = (vel_back[:, 10:13] - world_ang).abs().max().item()
    print(
        f"  WORLD-VEL-ROUNDTRIP lin_xy_ok={bool(lin_xy_ok)} lin_z_ok={bool(lin_z_ok)} "
        f"ang_ok={bool(ang_rt_ok)} robot_lin_ok={bool(robot_lin_ok)} robot_ang_ok={bool(robot_ang_ok)} "
        f"(lin_xy_err={lin_max_xy_err:.4f} ang_err={ang_max_err:.4f})"
    )
    if not (lin_xy_ok and lin_z_ok and ang_rt_ok and robot_lin_ok and robot_ang_ok):
        print(
            "FAIL: world-frame velocity (linear and/or angular) did not round-trip through "
            "set/get_actor_states for robot+object"
        )
        return 1

    print(
        f"ALL-ROOT-STATES-UNIFIED OK: {args.simulator} ({num_envs} env(s)) — "
        "robot+object read/write/clone/sentinel consistent"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
