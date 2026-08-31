"""Cross-backend camera-orientation (handedness) assertion harness.

A non-square camera (width != height) views a red panel off the optical axis, to the robot's left
(+Y body) and up (+Z body). A correct forward-looking camera images that panel in the top-right
quadrant; a left-right mirror, top-bottom flip, or width/height transpose moves it elsewhere and
fails.

Reference mapping (MuJoCo-classic, identity optical basis): a panel at body +Y, +Z, seen by a
camera looking down body +X, lands top-right (small row, large column).

Exits 0 PASS / 1 FAIL / 77 SKIP.
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


def _red_centroid(img):
    """(row, col, count) of the red-panel pixels in one [H,W,3] frame, or (None, None, 0)."""
    import torch

    r, g, b = img[..., 0].to(int), img[..., 1].to(int), img[..., 2].to(int)
    mask = (r > g + 40) & (r > b + 40)
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() == 0:
        return None, None, 0
    return float(ys.float().mean()), float(xs.float().mean()), int(ys.numel())


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
    # Non-square camera (96x64) with an off-axis panel (left and up): the red lands in one quadrant.
    config = build_run_sim_config(
        sim_arg, "panel-offaxis", args.robot, args.terrain, sensors=_camera_presets.front_cam_wide
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
    env_origins = torch.zeros(n, 3, device=device)
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device, dtype=torch.float32
    )
    sim.create_envs(n, env_origins, base_init)
    sim.prepare_sim()

    # Pin the robot upright at each origin so the camera aligns with the off-axis panel ahead. Use
    # the origins the simulator ACTUALLY placed the envs at (IsaacSim clones onto its own grid and
    # ignores the requested env_origins; sim.env_origins is reconciled to the real placement).
    env_origins = sim.env_origins

    _all_ids = torch.arange(n, device=device)
    # Capture the spawn joint pose once so the pin can restore it: the robot is un-actuated, so
    # holding only the ROOT lets the JOINTS sag over the settle step, tilting/yawing the pelvis a
    # few degrees and swinging the pelvis-mounted camera enough to shift the off-axis panel across
    # the vertical centerline — a near-boundary env images TOP-LEFT instead of TOP-RIGHT (the
    # isaacgym single-gpu flake). Holding the joints rigid too removes the settle at its source.
    _spawn_dof_pos = sim.dof_pos.clone()

    def _hold_dof() -> None:
        # Restore joints to the spawn pose with zero velocity. The cross-backend DOF setter takes
        # different tensor shapes (IsaacGym flattened [n*ndof, 2], IsaacSim 3D [n, ndof, 2]); build
        # whichever this backend expects. MuJoCo isn't affected by this flake, so only the two
        # articulation backends are handled.
        ndof = sim.num_dof
        if args.simulator == "isaacgym":
            ds = torch.zeros(n * ndof, 2, device=device)
            ds[:, 0] = _spawn_dof_pos.reshape(-1)
            sim.set_dof_state_tensor_robots(_all_ids, ds)
        elif args.simulator == "isaacsim":
            ds = torch.zeros(n, ndof, 2, device=device)
            ds[:, :, 0] = _spawn_dof_pos
            sim.set_dof_state_tensor_robots(_all_ids, ds)

    def _pin_robot() -> None:
        # Set the target root pose AND zero all velocities, hold the joints rigid, then read back to
        # verify the write landed before we rely on it: any residual root velocity, joint sag, or a
        # dropped write lets the pelvis drift and drags its mounted camera off the panel. Retry until
        # it sticks.
        target_pos = env_origins + torch.tensor(list(init.pos), device=device)
        target_rot = torch.tensor(list(init.rot), device=device)
        for _ in range(3):
            robot_states = sim.get_actor_states(["robot"], _all_ids).clone()
            robot_states[:, :3] = target_pos
            robot_states[:, 3:7] = target_rot
            robot_states[:, 7:] = 0.0
            sim.set_actor_states(["robot"], _all_ids, robot_states)
            _hold_dof()
            back = sim.get_actor_states(["robot"], _all_ids)
            if torch.allclose(back[:, :3], target_pos, atol=1e-3) and back[:, 7:].abs().max() < 1e-3:
                break

    _pin_robot()
    step(sim, max(2, steps_for_seconds(sim, 0.05)))
    # Re-pin immediately before capture, then step a few frames so the corrected pose reaches the
    # render (on IsaacSim a root/joint write needs a few physics steps to propagate; other backends
    # are immediate) while the joints are held rigid so the un-actuated robot can't drift again.
    _pin_robot()
    step(sim, 4)
    sim.render_sensors()

    cam_name, cam = next(iter(config.sensor.items()))
    fails: list[str] = []
    img_all = sim.get_camera_data(cam_name, "rgb")  # [N,H,W,3]
    # Shape (N, height, width, 3) with height != width; catches a W/H transpose.
    if tuple(img_all.shape) != (n, cam.height, cam.width, 3) or img_all.dtype != torch.uint8:
        msg = (
            f"{args.simulator}: shape/dtype {tuple(img_all.shape)} {img_all.dtype} "
            f"!= ({n},{cam.height},{cam.width},3) uint8"
        )
        print(f"[{args.simulator}] FAIL: {msg}")
        if args.result_file:
            with open(args.result_file, "w") as fh:
                fh.write("FAIL\n" + msg)
        return 1

    h, w = cam.height, cam.width
    for e in range(n):
        rc, cc, count = _red_centroid(img_all[e])
        if count < 0.01 * h * w:
            fails.append(f"{args.simulator}/env{e}: off-axis panel not visible ({count} red px)")
            continue
        # Panel is left (+Y) and up (+Z) -> TOP (row < H/2) and RIGHT (col > W/2) of the image.
        top = rc < h / 2
        right = cc > w / 2
        print(
            f"[{args.simulator}] env{e}: red centroid (row={rc:.1f}/{h}, col={cc:.1f}/{w}) -> "
            f"{'TOP' if top else 'BOTTOM'}-{'RIGHT' if right else 'LEFT'}"
        )
        if not (top and right):
            fails.append(
                f"{args.simulator}/env{e}: off-axis (+Y left, +Z up) panel imaged "
                f"{'TOP' if top else 'BOTTOM'}-{'RIGHT' if right else 'LEFT'}, expected TOP-RIGHT "
                f"(mirror/flip/transpose in the image buffer)"
            )

    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        for f in fails:
            print(f"[{args.simulator}] FAIL: {f}")
        return 1
    print(f"[{args.simulator}] PASS: off-axis panel lands TOP-RIGHT on a non-square frame across {n} env(s)")
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
