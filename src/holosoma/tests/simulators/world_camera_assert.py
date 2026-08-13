"""Cross-backend free-floating (``target_kind="world"``) camera behavioral assertion harness.

A world camera anchors to each env's frame, NOT to any body, so moving or yawing the robot must
NOT change what it sees. This is the behavioral inverse of ``camera_follow_assert`` (which asserts
a robot_link camera DOES track its body).

Setup mirrors the follow test: a world camera fixed at the robot's spawn spot looks forward (+X) at
a red panel 1.0 m ahead. We render (baseline), then (1) advance the robot toward the panel and
(2) yaw it 40deg. A world camera's panel depth and red-fraction must stay ~constant through both
(the robot moving in front of it may perturb pixels slightly, but the panel stays framed at the
same depth). For num_envs>1 we also check the per-env views are independent (each env's world
camera sits at that env's origin). Exits 0 PASS, 1 FAIL, 77 SKIP.

  python world_camera_assert.py --simulator mujoco
  python world_camera_assert.py --simulator mjwarp   --num-envs 3
  python world_camera_assert.py --simulator isaacgym --num-envs 3
  python world_camera_assert.py --simulator isaacsim --num-envs 3
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators._sim_harness import build_run_sim_config, step, steps_for_seconds

SKIP_EXIT_CODE = 77
_CAM = "world_cam"


def _red_mask(rgb):
    return (rgb[..., 0].to(int) > rgb[..., 1].to(int) + 40) & (rgb[..., 0].to(int) > rgb[..., 2].to(int) + 40)


def _red_fraction(rgb):
    """Fraction of red-panel pixels in one [H,W,3] frame."""
    return float(_red_mask(rgb).float().mean())


def _panel_median_depth(rgb, depth):
    """Median depth (meters) over the red-panel pixels of one env frame, or None if not visible."""
    import torch

    mask = _red_mask(rgb)
    if int(mask.sum()) < 0.02 * rgb.shape[0] * rgb.shape[1]:
        return None
    panel = depth[..., 0][mask]
    finite = panel[torch.isfinite(panel)]
    return float(finite.median()) if finite.numel() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--robot", default="g1-29dof")
    parser.add_argument("--terrain", default="terrain_locomotion_plane")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", choices=["true", "false"], default="true")
    parser.add_argument("--result-file", default=None, help="write OK/FAIL here before teardown")
    args = parser.parse_args()

    from tests.simulators import _camera_presets

    headless = args.headless == "true"
    sim_arg = "mujoco" if args.simulator == "mjwarp" else args.simulator
    config = build_run_sim_config(sim_arg, "panel-target", args.robot, args.terrain, sensors=_camera_presets.world_cam)
    if args.simulator == "mjwarp":
        config = _camera_presets.as_mjwarp(config)

    device = "cuda:0" if args.simulator != "mujoco" else "cpu"
    config = dataclasses.replace(
        config, device=device, training=dataclasses.replace(config.training, num_envs=args.num_envs)
    )

    env, device, _app = setup_simulation_environment(config, device=device)
    sim = env.sim
    sim.set_headless(headless)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import math

    import torch

    n = args.num_envs
    # Spread envs along +X so each gets its own panel copy at its own origin. The world camera must
    # sit at each env's origin (per-env frame), so each env sees its own panel.
    env_origins = torch.zeros(n, 3, device=device)
    if n > 1:
        env_origins[:, 0] = torch.arange(n, device=device, dtype=torch.float32) * 10.0
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device, dtype=torch.float32
    )
    sim.create_envs(n, env_origins, base_init)
    sim.prepare_sim()

    if not sim.get_sensor_names():
        print(f"[{args.simulator}] FAIL: no sensors created")
        return 1

    move = 0.3  # meters to advance the robot toward the panel (+X base frame)
    yaw = math.radians(40.0)  # body yaw
    base_rot = list(init.rot)  # (x, y, z, w) actor-state quaternion order

    def _quat_mul_yaw(q_xyzw, yaw_rad):
        """Compose a world-Z yaw onto the base orientation (xyzw)."""
        cz, sz = math.cos(yaw_rad / 2), math.sin(yaw_rad / 2)
        yq = [0.0, 0.0, sz, cz]  # yaw about +Z (x, y, z, w)
        x1, y1, z1, w1 = yq
        x2, y2, z2, w2 = q_xyzw
        return [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]

    # Keep the robot OUT of the world camera's forward sightline. The camera sits on the camera↔panel
    # axis (y=0) looking +X; the robot spawns on that same axis, so advancing it +X toward the panel —
    # or the un-actuated robot collapsing under gravity during the settle step — puts a limb between
    # the (fixed) camera and the panel and occludes it, which is exactly what this test must NOT
    # conflate with camera motion. Shove the robot 1.5 m to the side (clear of the ~0.6 m half-width
    # 60° frustum at the ~1 m panel distance) so it still moves/yaws to prove the world camera doesn't
    # follow it, without ever occluding the panel.
    robot_y_offset = -1.5

    def _place(x_offset: float, rot_xyzw) -> None:
        all_ids = torch.arange(n, device=device)
        states = sim.get_actor_states(["robot"], all_ids).clone()
        states[:, :3] = env_origins + torch.tensor(list(init.pos), device=device)
        states[:, 0] += x_offset
        states[:, 1] += robot_y_offset
        states[:, 3:7] = torch.tensor(rot_xyzw, device=device)
        states[:, 7:] = 0.0
        sim.set_actor_states(["robot"], all_ids, states)
        step(sim, max(2, steps_for_seconds(sim, 0.05)))
        sim.render_sensors()

    def _panel_depths():
        return [
            _panel_median_depth(sim.get_camera_data(_CAM, "rgb")[e], sim.get_camera_data(_CAM, "depth")[e])
            for e in range(n)
        ]

    def _reds():
        return [_red_fraction(sim.get_camera_data(_CAM, "rgb")[e]) for e in range(n)]

    # Baseline: robot at spawn, world camera framing the panel head-on.
    _place(0.0, base_rot)
    base_depth = _panel_depths()
    base_red = _reds()

    fails: list[str] = [
        f"{args.simulator}/env{e}: world camera does not see the panel at baseline"
        for e in range(n)
        if base_depth[e] is None
    ]
    if fails:
        for f in fails:
            print(f"[{args.simulator}] FAIL: {f}")
        if args.result_file:
            with open(args.result_file, "w") as fh:
                fh.write("FAIL\n" + "\n".join(fails))
        return 1

    # 1. Advance the robot toward the panel. A WORLD camera does not move, so the panel's depth
    #    from the (fixed) camera must stay ~constant — unlike a follow camera, whose depth would
    #    drop by ~move.
    _place(move, base_rot)
    moved_depth = _panel_depths()

    # 2. Yaw the robot 40deg. A world camera keeps framing the panel: red fraction stays ~constant
    #    (a follow camera would swing the panel out of frame).
    _place(0.0, _quat_mul_yaw(base_rot, yaw))
    yawed_red = _reds()

    for e in range(n):
        print(
            f"[{args.simulator}] env{e}: panel depth base {base_depth[e]:.3f}m -> after-robot-move "
            f"{moved_depth[e] if moved_depth[e] is None else round(moved_depth[e], 3)}m; "
            f"red {base_red[e]:.3f} -> after-40deg-yaw {yawed_red[e]:.3f}"
        )
        if moved_depth[e] is None:
            fails.append(f"{args.simulator}/env{e}: panel left the world-camera frame after the robot moved")
            continue
        # The world camera is fixed, so its panel depth must be ~unchanged (tolerance 5 cm; the
        # robot passing in front can nibble panel pixels but not shift the panel's distance).
        if abs(moved_depth[e] - base_depth[e]) > 0.05:
            fails.append(
                f"{args.simulator}/env{e}: world-camera panel depth changed {base_depth[e]:.3f}m -> "
                f"{moved_depth[e]:.3f}m when the ROBOT moved {move}m; a world camera must not follow the robot."
            )
        # Panel must stay framed after the yaw: red fraction at least half its baseline (a follow
        # camera would collapse it).
        if base_red[e] > 0.02 and yawed_red[e] < 0.5 * base_red[e]:
            fails.append(
                f"{args.simulator}/env{e}: world-camera red fraction {base_red[e]:.3f} -> {yawed_red[e]:.3f} after a "
                f"robot yaw; a world camera must keep framing the panel (it does not follow the robot)."
            )

    # 3. Per-env independence (n > 1): every env's world camera should see its own panel at the same
    #    baseline depth (each sits at its own env origin, not env 0's).
    if n > 1 and not fails:
        spread = max(base_depth) - min(base_depth)
        print(f"[{args.simulator}] per-env baseline panel depths: {[round(d, 3) for d in base_depth]}")
        if spread > 0.05:
            fails.append(
                f"{args.simulator}: per-env world-camera panel depths vary by {spread:.3f}m "
                f"({[round(d, 3) for d in base_depth]}); each env's world camera should sit at its own origin."
            )

    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        for f in fails:
            print(f"[{args.simulator}] FAIL: {f}")
        return 1
    print(f"[{args.simulator}] PASS: world camera stays fixed in the env frame across {n} env(s)")
    return 0


if __name__ == "__main__":
    # IsaacSim teardown deadlocks in carbOnPluginShutdown tearing down the
    # omni.syntheticdata/OmniGraph render-product graph a TiledCamera creates (native
    # py-spy stack), so a normal interpreter exit hangs until the parent's subprocess
    # timeout SIGKILLs it -- turning a PASS (verdict already written to --result-file) into
    # a spurious timeout failure. Hard-exit past the atexit teardown, mirroring
    # behavior_assert / scene_spawn_assert. Rendering itself is fine; only exit hangs.
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
