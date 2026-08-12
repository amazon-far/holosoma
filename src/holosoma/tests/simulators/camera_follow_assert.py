"""Cross-backend camera-follow behavioral assertion harness.

Asserts a mounted camera tracks its mount body (MuJoCo spec child, IsaacGym
FOLLOW_TRANSFORM, IsaacSim child prim).

A depth camera is mounted on the robot base looking forward at a red panel at a known
distance. Render, move the robot a known ``d`` toward the panel via ``set_actor_states``,
re-render, and assert the panel's median depth drops by ~``d``. Depth is sign-independent
across backends. Exits 0 PASS, 1 FAIL, 77 SKIP.
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


def _red_mask(rgb):
    return (rgb[..., 0].to(int) > rgb[..., 1].to(int) + 40) & (rgb[..., 0].to(int) > rgb[..., 2].to(int) + 40)


def _panel_median_depth(rgb, depth):
    """Median depth (meters) over the red-panel pixels of one env frame, or None if not visible."""
    import torch

    mask = _red_mask(rgb)
    if int(mask.sum()) < 0.02 * rgb.shape[0] * rgb.shape[1]:
        return None
    panel = depth[..., 0][mask]
    finite = panel[torch.isfinite(panel)]
    return float(finite.median()) if finite.numel() else None


def _red_fraction(rgb):
    """Fraction of red-panel pixels in one [H,W,3] frame."""
    return float(_red_mask(rgb).float().mean())


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
    config = build_run_sim_config(
        sim_arg, "panel-target", args.robot, args.terrain, sensors=_camera_presets.front_cam_depth
    )
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
    import torch

    n = args.num_envs
    # Spread envs along +X so each gets its own panel copy at its own origin.
    env_origins = torch.zeros(n, 3, device=device)
    if n > 1:
        env_origins[:, 0] = torch.arange(n, device=device, dtype=torch.float32) * 10.0
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device, dtype=torch.float32
    )
    sim.create_envs(n, env_origins, base_init)
    sim.prepare_sim()
    # Use the origins the simulator ACTUALLY placed the envs at: IsaacSim ignores the requested
    # env_origins and clones onto its own env_spacing grid, so pinning the robot relative to the
    # requested spread would land it away from its (grid-placed) panel. sim.env_origins is reconciled
    # to the real placement on every backend.
    env_origins = sim.env_origins

    if not sim.get_sensor_names():
        print(f"[{args.simulator}] FAIL: no sensors created")
        return 1

    _all_ids = torch.arange(n, device=device)
    # Capture the spawn joint pose so each _place can hold the articulation rigid: the robot is
    # un-actuated, so over the settle steps the joints sag and the pelvis (hence its mounted camera)
    # tilts a few degrees — enough to flake a random env's frame out of tolerance in multi-env. The
    # test wants the body to move RIGIDLY (root translate/yaw) while the camera follows it, so holding
    # the joints at spawn is exactly the intended behavior, not a cheat.
    _spawn_dof_pos = sim.dof_pos.clone()

    def _hold_dof() -> None:
        # Restore joints to the spawn pose with zero velocity; per-backend DOF-state tensor shapes
        # (IsaacGym flat [n*ndof, 2], IsaacSim 3D [n, ndof, 2]). MuJoCo is unaffected by this flake.
        ndof = sim.num_dof
        if args.simulator == "isaacgym":
            ds = torch.zeros(n * ndof, 2, device=device)
            ds[:, 0] = _spawn_dof_pos.reshape(-1)
            sim.set_dof_state_tensor_robots(_all_ids, ds)
        elif args.simulator == "isaacsim":
            ds = torch.zeros(n, ndof, 2, device=device)
            ds[:, :, 0] = _spawn_dof_pos
            sim.set_dof_state_tensor_robots(_all_ids, ds)

    import math

    move = 0.3  # meters to advance the robot toward the panel (+X base frame)
    yaw = math.radians(40.0)  # body yaw that swings the forward panel out of frame

    base_rot = list(init.rot)  # (x, y, z, w) actor-state quaternion order

    def _quat_mul_yaw(q_xyzw, yaw_rad):
        """Compose a world-Z yaw onto the base orientation (xyzw) so the robot turns in place."""
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

    def _set_root(x_offset: float, rot_xyzw) -> None:
        states = sim.get_actor_states(["robot"], _all_ids).clone()
        states[:, :3] = env_origins + torch.tensor(list(init.pos), device=device)
        states[:, 0] += x_offset
        states[:, 3:7] = torch.tensor(rot_xyzw, device=device)
        states[:, 7:] = 0.0
        sim.set_actor_states(["robot"], _all_ids, states)

    def _place(x_offset: float, rot_xyzw) -> None:
        # Set the root pose + hold joints rigid, settle, then re-assert both right before the render:
        # holding the DOF pose stops the un-actuated joints from sagging the pelsvis/camera over the
        # settle, and re-setting immediately before render_sensors lands the exact pose in the frame
        # (on IsaacSim a pose write needs a few steps to propagate to the render, which the settle
        # provides). Robust against the multi-env drift flake.
        _set_root(x_offset, rot_xyzw)
        _hold_dof()
        step(sim, max(2, steps_for_seconds(sim, 0.05)))
        _set_root(x_offset, rot_xyzw)
        _hold_dof()
        step(sim, 2)
        sim.render_sensors()

    cam_name, _cam = next(iter(config.sensor.items()))
    _place(0.0, base_rot)
    before = [
        _panel_median_depth(sim.get_camera_data(cam_name, "rgb")[e], sim.get_camera_data(cam_name, "depth")[e])
        for e in range(n)
    ]
    before_red = [_red_fraction(sim.get_camera_data(cam_name, "rgb")[e]) for e in range(n)]

    # 1. Translation: advance +X toward the panel; its depth should drop by ~move.
    _place(move, base_rot)
    after = [
        _panel_median_depth(sim.get_camera_data(cam_name, "rgb")[e], sim.get_camera_data(cam_name, "depth")[e])
        for e in range(n)
    ]

    # 2. Rotation: from the original spot, yaw the robot 40deg. A camera that follows body
    #    orientation swings the forward panel out of frame (red fraction collapses).
    _place(0.0, _quat_mul_yaw(base_rot, yaw))
    yawed_red = [_red_fraction(sim.get_camera_data(cam_name, "rgb")[e]) for e in range(n)]

    fails: list[str] = []
    for e in range(n):
        if before[e] is None or after[e] is None:
            fails.append(f"{args.simulator}/env{e}: panel not visible before/after move ({before[e]}, {after[e]})")
            continue
        drop = before[e] - after[e]
        print(
            f"[{args.simulator}] env{e}: panel depth {before[e]:.3f}m -> {after[e]:.3f}m "
            f"(drop {drop:.3f}m, moved {move}m); "
            f"red {before_red[e]:.3f} -> {yawed_red[e]:.3f} after 40deg yaw"
        )
        if abs(drop - move) > 0.25 * move:
            fails.append(
                f"{args.simulator}/env{e}: panel depth dropped {drop:.3f}m, expected ~{move}m after moving the "
                f"mount body (camera did not follow its body in translation)"
            )
        # After a 40deg yaw the forward panel should largely leave frame: red fraction at most a
        # third of its head-on value.
        if before_red[e] > 0.02 and yawed_red[e] > 0.34 * before_red[e]:
            fails.append(
                f"{args.simulator}/env{e}: red fraction {before_red[e]:.3f} -> {yawed_red[e]:.3f} after 40deg yaw; "
                f"panel did not leave frame (camera did not follow its body in rotation)"
            )

    # 3. Per-env independence (n > 1): set every robot head-on, then yaw only env 0. Env 0's red
    # should collapse while the other envs stay framed.
    if n > 1 and not fails:
        all_ids = torch.arange(n, device=device)
        states = sim.get_actor_states(["robot"], all_ids).clone()
        states[:, :3] = env_origins + torch.tensor(list(init.pos), device=device)
        states[:, 3:7] = torch.tensor(base_rot, device=device)
        states[0, 3:7] = torch.tensor(_quat_mul_yaw(base_rot, yaw), device=device)  # env 0 only
        states[:, 7:] = 0.0
        sim.set_actor_states(["robot"], all_ids, states)
        step(sim, max(2, steps_for_seconds(sim, 0.05)))
        sim.render_sensors()
        per_env_red = [_red_fraction(sim.get_camera_data(cam_name, "rgb")[e]) for e in range(n)]
        print(f"[{args.simulator}] per-env red after yawing ONLY env0: {[round(r, 3) for r in per_env_red]}")
        if per_env_red[0] > 0.34 * before_red[0]:
            fails.append(
                f"{args.simulator}: env0 was yawed but still sees the panel "
                f"(red {per_env_red[0]:.3f}); per-env pose not applied"
            )
        fails.extend(
            f"{args.simulator}/env{e}: red {per_env_red[e]:.3f} dropped though only env0 moved "
            f"(env frames not independent; backend broadcasts one env's render)"
            for e in range(1, n)
            if before_red[e] > 0.02 and per_env_red[e] < 0.5 * before_red[e]
        )

    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        for f in fails:
            print(f"[{args.simulator}] FAIL: {f}")
        return 1
    print(f"[{args.simulator}] PASS: camera follows its mount body across {n} env(s)")
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
