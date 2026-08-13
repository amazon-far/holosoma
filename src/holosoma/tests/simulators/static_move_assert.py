"""Headless cross-backend assertion harness for RUNTIME kinematic moves of static scene bodies.

Builds a sim for a chosen backend with a free box dropped onto a STATIC pillar, then mid-rollout
teleports the pillar away via the unified ``set_static_body_pose`` API. The proof is a single-sim
before/after on the free box's height (no second CONTROL sim, no inter-run gap threshold):

  1. RESTS-ON — after settling, the free box comes to rest ON the pillar: its z sits at
     ``pillar_top + box_half`` and well above the floor. This proves the static body is solid at
     its spawned pose (the box did not pass through it).
  2. MOVED + FALLS-THROUGH — ``get_actor_states([pillar])`` reports the teleported world pose
     (position + applied yaw), proving the kinematic write took on the live backend; and after
     the pillar is yanked out from under it, the box falls BELOW the pillar's old top toward the
     floor. The within-run drop (settled-on-pillar height -> below-old-top) is the portable
     "physics honors the static, and stops honoring it once moved" signal — it needs no
     per-backend contact-tensor plumbing.

Settling uses :func:`steps_for_seconds` so each backend runs the SAME amount of SIM-TIME
regardless of its per-step ``sim_dt`` (every backend's ``simulate_at_each_physics_step`` /
``backend.step()`` advances exactly one ``sim_dt`` physics step).

Exits 0 on success, non-zero with a message on failure, so it runs as a live integration test
under each backend's launcher (MuJoCo venv, IsaacGym, IsaacSim). Mirrors scene_spawn_assert.py.

Usage:
  python static_move_assert.py --simulator mujoco                 # ClassicBackend, 1 env, cpu
  python static_move_assert.py --simulator mjwarp   --num-envs 4  # WarpBackend, cuda
  python static_move_assert.py --simulator isaacgym --num-envs 4  # IsaacGym, cuda
  python static_move_assert.py --simulator isaacsim --num-envs 4  # IsaacSim, cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys

# tests/simulators/ has an isaacsim/ subpackage that would shadow the real IsaacSim package if it
# lands on sys.path[0] when run as a script — drop it (mirrors scene_spawn_assert.py).
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

import tyro

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.scene import RigidObjectConfig, SceneConfig
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.utils.sim_utils import setup_simulation_environment

# steps_for_seconds re-exported (subset_write_assert imports it from here).
from tests.simulators._sim_harness import step, steps_for_seconds

SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"
BOX_HALF = 0.05  # small_box is a 0.1 m cube.

# Free box well clear of the robot at the origin, dropped from just above the pillar's top so it
# lands and settles quickly. The pillar (also a 0.1 m cube) spawns directly under the box; its top
# is PILLAR_Z + BOX_HALF, and the box rests centered at PILLAR_Z + 2*BOX_HALF.
BOX_X = 2.0
PILLAR_Z = 0.30
PILLAR_TOP = PILLAR_Z + BOX_HALF  # 0.35
BOX_REST_Z = PILLAR_Z + 2 * BOX_HALF  # 0.40 — box center when resting on the pillar
BOX_DROP_Z = BOX_REST_Z + 0.05  # small drop so it settles fast
PILLAR_REMOVED_X = 6.0  # where the pillar is teleported to (far from the box's column)

SETTLE_SECONDS = 1.5  # sim-time to let the box settle on the pillar / fall after removal


def _scene() -> SceneConfig:
    return SceneConfig(
        rigid_objects={
            "freebox": RigidObjectConfig(urdf_file=SMALL_BOX, position=[BOX_X, 0.0, BOX_DROP_Z]),
            "pillar": RigidObjectConfig(urdf_file=SMALL_BOX, position=[BOX_X, 0.0, PILLAR_Z], fixed=True),
        }
    )


def _build(simulator: str, num_envs: int):
    from holosoma.config_types.simulator import BridgeConfig, VirtualGantryCfg

    argv = [f"simulator:{simulator}", "robot:g1-29dof", "terrain:terrain_locomotion_plane", "scene:empty"]
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
        scene=_scene(),
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
    if num_envs > 1:
        origins[:, 0] = torch.arange(num_envs, device=device, dtype=torch.float32) * 8.0
    init = config.robot.init_state
    base_init = torch.tensor(list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device)
    s.create_envs(num_envs, origins, base_init)
    s.prepare_sim()
    return s, origins


def _step(sim, n):
    """Advance ``n`` physics steps, no render (this harness asserts physics, not video)."""
    step(sim, n, render=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-backend runtime static-move assertion harness.")
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--num-envs", type=int, default=None)
    args = parser.parse_args()
    num_envs = args.num_envs if args.num_envs is not None else (1 if args.simulator == "mujoco" else 4)

    import torch

    sim, origins = _build(args.simulator, num_envs)
    free = sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
    static = sim.object_registry.get_names_by_type(ObjectType.SCENE)
    print(f"[{args.simulator}] free={free} static={static} sim_dt={sim.sim_dt}")
    if "freebox" not in free or "pillar" not in static:
        print("FAIL: scene did not register freebox(INDIVIDUAL)+pillar(SCENE)")
        return 1

    env_ids = torch.arange(num_envs, device=sim.sim_device)
    settle = steps_for_seconds(sim, SETTLE_SECONDS)

    # PHASE 1 — let the box settle onto the pillar. It must come to rest ON TOP of the pillar
    # (z ~= BOX_REST_Z) and clearly above the floor, proving the static body is solid at its pose.
    _step(sim, settle)
    z_on = sim.get_actor_states(["freebox"], env_ids)[:, 2]  # [num_envs]
    rest_ok = torch.allclose(z_on, torch.full_like(z_on, BOX_REST_Z), atol=3e-2)
    above_floor = bool((z_on > PILLAR_TOP - 1e-2).all())  # well clear of the ground
    print(
        f"  RESTS-ON freebox z={[round(float(v), 3) for v in z_on]} (expect ~{BOX_REST_Z}) "
        f"ok={bool(rest_ok)} above_floor={above_floor}"
    )
    if not (rest_ok and above_floor):
        print(f"FAIL: box did not rest on the static pillar (z {z_on.tolist()} != ~{BOX_REST_Z})")
        return 1

    # PHASE 2 — teleport the pillar out from under the box (with a 90deg yaw, to prove orientation
    # writes too). set_static_body_pose takes a WORLD pose on EVERY backend (the uniform frame
    # contract), so build the world target — env origin + local removed-x — and pass it as-is; no
    # `if simulator == ...` branch.
    c, sn = math.cos(math.pi / 4), math.sin(math.pi / 4)
    base = torch.tensor([PILLAR_REMOVED_X, 0.0, PILLAR_Z, 0.0, 0.0, sn, c], device=sim.sim_device, dtype=torch.float32)
    poses = base.unsqueeze(0).repeat(num_envs, 1)
    poses[:, :3] += origins
    sim.set_static_body_pose(["pillar"], env_ids, poses)
    moved = sim.get_actor_states(["pillar"], env_ids)  # [num_envs, 13]

    # MOVED: the pillar reports the teleport target in WORLD frame. get_actor_states returns WORLD
    # on every backend, so the expected x is origin + target.
    want_x = origins[:, 0] + PILLAR_REMOVED_X
    moved_ok = torch.allclose(moved[:, 0], want_x.to(moved.device), atol=2e-2)
    yaw_ok = (moved[:, 5] - sn).abs().max() < 5e-2 and (moved[:, 6] - c).abs().max() < 5e-2
    print(
        f"  MOVED pillar x={[round(float(v), 3) for v in moved[:, 0]]} "
        f"want {[round(float(v), 3) for v in want_x]} ok={bool(moved_ok)} yaw_ok={bool(yaw_ok)}"
    )
    if not (moved_ok and yaw_ok):
        print("FAIL: static body did not report the relocated pose")
        return 1

    # FALLS-THROUGH: with the pillar gone, the box is unsupported and falls BELOW the pillar's old
    # top toward the floor. This is the within-run signal that the kinematic move actually removed
    # the support (not just updated a reported pose).
    _step(sim, settle)
    z_after = sim.get_actor_states(["freebox"], env_ids)[:, 2]
    fell_ok = bool((z_after < PILLAR_TOP - 0.05).all())
    print(
        f"  FALLS-THROUGH freebox z={[round(float(v), 3) for v in z_after]} "
        f"(must drop below old pillar top {PILLAR_TOP}) ok={fell_ok}"
    )
    if not fell_ok:
        print(
            f"FAIL: box did not fall after the static was removed (z {z_after.tolist()} still >= {PILLAR_TOP - 0.05})"
        )
        return 1

    print(
        f"STATIC-MOVE OK: {args.simulator} ({num_envs} env(s)) — "
        f"box rested on the static, which moved AND stopped supporting it"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
