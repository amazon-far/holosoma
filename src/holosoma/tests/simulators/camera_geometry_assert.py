"""Quantitative cross-backend camera-geometry assertions.

Verifies the camera's projection is geometrically correct, with numeric thresholds applied
identically on every backend:

  1. Centering: the panel's silhouette centroid sits at the image center (within tol).
  2. Resolution: the frame is exactly W x H.
  3. Field of view: the panel's measured pixel half-height matches the pinhole projection
     predicted from ``vertical_fov``, the camera-to-panel distance, and the panel's real size:
         px_half = (H/2) * (s / d) / tan(fovy/2)
  4. No distortion: the silhouette is left-right and top-bottom symmetric about the center,
     and its aspect ratio matches the (square) panel.

The panel is a flat bright-red rectangle of known size at a known distance directly ahead,
segmentable by R dominance.

Run per backend:
    python tests/simulators/camera_geometry_assert.py --simulator mujoco
    python tests/simulators/camera_geometry_assert.py --simulator mjwarp --num-envs 2
Exits 0 PASS, 1 FAIL, 77 SKIP.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators._sim_harness import build_run_sim_config, step, steps_for_seconds

SKIP_EXIT_CODE = 77


def _red_mask(env_img):
    """Boolean [H,W] mask of red-panel pixels (R clearly dominant over G and B)."""
    r = env_img[..., 0].to(int)
    g = env_img[..., 1].to(int)
    b = env_img[..., 2].to(int)
    return (r > g + 40) & (r > b + 40)


def _measure(mask):
    """Return (count, row_centroid, col_centroid, row_extent, col_extent) of a boolean mask."""
    import torch

    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    return (
        int(ys.numel()),
        float(ys.float().mean()),
        float(xs.float().mean()),
        float(ys.max() - ys.min() + 1),
        float(xs.max() - xs.min() + 1),
    )


def _check_geometry(img, cam, cam_to_panel_dist, panel_half_size, label):
    """All geometry assertions on one env's frame. Returns list of failure strings."""
    fails = []
    h, w, _ = img.shape
    mask = _red_mask(img)
    count, rc, cc, r_ext, c_ext = _measure(mask)

    # Require the panel to cover at least 2% of the frame before measuring geometry.
    if count < 0.02 * h * w:
        return [f"{label}: panel barely visible ({count}/{h * w} red px); cannot assert geometry"]

    # 1. Centering: silhouette centroid at image center within 8% of the dimension.
    cx_tol, cy_tol = 0.08 * w, 0.08 * h
    if abs(cc - (w - 1) / 2) > cx_tol:
        fails.append(f"{label}: horizontal centroid {cc:.1f} not centered (center {(w - 1) / 2:.1f}, tol {cx_tol:.1f})")
    if abs(rc - (h - 1) / 2) > cy_tol:
        fails.append(f"{label}: vertical centroid {rc:.1f} not centered (center {(h - 1) / 2:.1f}, tol {cy_tol:.1f})")

    # 3. Field of view: predicted pixel half-height for a pinhole camera.
    #    full_px_height = H * (panel_full_size) / (2 * d * tan(fovy/2))
    fovy = math.radians(cam.vertical_fov)
    predicted_px_height = h * (2 * panel_half_size) / (2 * cam_to_panel_dist * math.tan(fovy / 2))
    # 15% tolerance absorbs per-engine silhouette bleed of ~1-2px/side at 64px.
    if abs(r_ext - predicted_px_height) > 0.15 * predicted_px_height:
        fails.append(
            f"{label}: panel pixel height {r_ext:.1f} != FOV-predicted {predicted_px_height:.1f} "
            f"(fovy={cam.vertical_fov}deg, d={cam_to_panel_dist:.2f}m, size={2 * panel_half_size}m); "
            f"FOV/projection mismatch"
        )

    # 4. No distortion: square panel gives a near-square silhouette; symmetric margins.
    aspect = c_ext / max(1.0, r_ext)
    if not (0.85 <= aspect <= 1.18):
        fails.append(f"{label}: silhouette aspect {aspect:.2f} not ~1.0 (square panel) -> distortion/shear")
    # Left vs right margin and top vs bottom margin should match (symmetry about center).
    ys, xs = mask.nonzero(as_tuple=True)
    left_m, right_m = float(xs.min()), float(w - 1 - xs.max())
    top_m, bot_m = float(ys.min()), float(h - 1 - ys.max())
    if abs(left_m - right_m) > 0.10 * w:
        fails.append(f"{label}: L/R margins {left_m:.1f}/{right_m:.1f} asymmetric -> off-axis/distortion")
    if abs(top_m - bot_m) > 0.10 * h:
        fails.append(f"{label}: T/B margins {top_m:.1f}/{bot_m:.1f} asymmetric -> off-axis/distortion")
    return fails


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--robot", default="g1-29dof")
    parser.add_argument("--terrain", default="terrain_locomotion_plane")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", choices=["true", "false"], default="true")
    parser.add_argument("--result-file", default=None, help="write PASS/FAIL here before teardown")
    args = parser.parse_args()

    from tests.simulators import _camera_presets

    headless = args.headless == "true"
    sim_arg = "mujoco" if args.simulator == "mjwarp" else args.simulator
    config = build_run_sim_config(sim_arg, "panel-target", args.robot, args.terrain, sensors=_camera_presets.front_cam)
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
    # All envs co-located at the origin so every env's robot and panel align.
    env_origins = torch.zeros(n, 3, device=device)
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device, dtype=torch.float32
    )
    sim.create_envs(n, env_origins, base_init)
    sim.prepare_sim()

    # Pin the robot to a known upright pose at each env origin so the (pelvis-mounted) camera aligns
    # with the panel placed ahead (IsaacGym jitters the spawn xy; other backends are already at origin).
    # Use the origins the simulator ACTUALLY placed the envs at (IsaacSim clones onto its own grid and
    # ignores the requested env_origins; sim.env_origins is reconciled to the real placement).
    env_origins = sim.env_origins
    import torch as _torch

    _all_ids = _torch.arange(n, device=device)
    # Capture the spawn joint pose once so the pin can restore it: the robot is un-actuated, so
    # holding only the ROOT still lets the JOINTS sag over the settle steps, tilting the pelvis a
    # few degrees and shifting the pelvis-mounted camera a few px off-axis (a rare multi-env L/R
    # asymmetry flake). Holding the joints rigid too removes the settle at its source.
    _spawn_dof_pos = sim.dof_pos.clone()

    def _hold_dof() -> None:
        # Restore joints to the spawn pose with zero velocity. The cross-backend DOF setter takes
        # different tensor shapes (IsaacGym flattened [n*ndof, 2], IsaacSim 3D [n, ndof, 2]); build
        # whichever this backend expects. MuJoCo isn't affected by this flake, so only the two
        # articulation backends are handled.
        ndof = sim.num_dof
        if args.simulator == "isaacgym":
            ds = _torch.zeros(n * ndof, 2, device=device)
            ds[:, 0] = _spawn_dof_pos.reshape(-1)
            sim.set_dof_state_tensor_robots(_all_ids, ds)
        elif args.simulator == "isaacsim":
            ds = _torch.zeros(n, ndof, 2, device=device)
            ds[:, :, 0] = _spawn_dof_pos
            sim.set_dof_state_tensor_robots(_all_ids, ds)

    def _pin_robot() -> None:
        # Set the target root pose AND zero all velocities, hold the joints rigid, then read back to
        # verify the write landed before we rely on it: any residual root velocity, joint sag, or a
        # dropped write lets the pelvis drift and drags its mounted camera off the panel (the
        # multi-env flake where a random env's panel leaves frame / lands off-center). Retry until it
        # sticks.
        target_pos = env_origins + _torch.tensor(list(init.pos), device=device)
        target_rot = _torch.tensor(list(init.rot), device=device)
        for _ in range(3):
            robot_states = sim.get_actor_states(["robot"], _all_ids).clone()
            robot_states[:, :3] = target_pos
            robot_states[:, 3:7] = target_rot
            robot_states[:, 7:] = 0.0
            sim.set_actor_states(["robot"], _all_ids, robot_states)
            _hold_dof()
            back = sim.get_actor_states(["robot"], _all_ids)
            if _torch.allclose(back[:, :3], target_pos, atol=1e-3) and back[:, 7:].abs().max() < 1e-3:
                break

    _pin_robot()
    step(sim, 2)

    names = sim.get_sensor_names()
    if not names:
        print(f"[{args.simulator}] FAIL: no sensors created")
        return 1

    step(sim, max(2, steps_for_seconds(sim, 0.05)))
    # Re-pin before capture, then step a few frames: the robot is un-actuated, so the settle steps
    # above let the pelvis drift and drag its mounted camera off the panel. Re-pinning restores the
    # exact pose, but on IsaacSim a root/joint write does NOT reach the render until physics has
    # stepped a few times to propagate it (an immediate render_sensors() after the write still shows
    # the pre-pin drifted pose — measured ~4px off, tripping the margin-symmetry check). Stepping ~4
    # frames with the joints held rigid (see _pin_robot) lands the corrected, centered pose in the
    # render on every backend without letting the robot drift again.
    _pin_robot()
    step(sim, 4)
    sim.render_sensors()

    cam_name, cam = next(iter(config.sensor.items()))
    # Camera-to-panel-face distance: panel center at x=_PANEL_DISTANCE, minus the camera mount x
    # offset, minus the panel half-thickness (0.01 m).
    cam_to_panel = _camera_presets._PANEL_DISTANCE - cam.mount.position[0] - 0.01
    panel_half = _camera_presets._PANEL_HALF_SIZE

    fails: list[str] = []
    img_all = sim.get_camera_data(cam_name, "rgb")  # [N,H,W,3]
    if tuple(img_all.shape) != (n, cam.height, cam.width, 3) or img_all.dtype != torch.uint8:
        print(f"[{args.simulator}] FAIL: shape/dtype {tuple(img_all.shape)} {img_all.dtype}")
        return 1
    for e in range(n):
        env_fails = _check_geometry(img_all[e], cam, cam_to_panel, panel_half, f"{args.simulator}/env{e}")
        for f in env_fails:
            print(f"[{args.simulator}] FAIL: {f}")
        fails += env_fails
        if not env_fails:
            print(f"[{args.simulator}] env{e}: geometry OK (centered, square, FOV-consistent silhouette)")

    # Persist the verdict before teardown (it can mask the exit code): "OK" on success per the
    # run_harness convention, else the failure detail.
    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        return 1
    print(f"[{args.simulator}] PASS: camera geometry correct across {n} env(s)")
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
