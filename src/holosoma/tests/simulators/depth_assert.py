"""Cross-backend depth behavioral assertion harness.

Same red-panel setup as the RGB geometry test. Panel pixels are located by RGB segmentation (the
panel is bright red), then their depth is checked against the known distance. Asserts:

  1. shape ``[num_envs, H, W, 1]``, dtype float32, device == sim device.
  2. Metric: panel pixel depth ~= known camera-to-panel face distance (meters).
  3. No-hit clipping: identical background no-returns become ``+inf``, per-camera ``far``, or
     zero for the ``none``, ``max``, and ``zero`` policies; a panel beyond a short camera's range
     follows the same rule.
  4. Image-plane convention: a flat fronto-parallel panel reads ~constant depth, not increasing
     toward the edges (which distance-to-camera/radial depth would show).

Checks metric agreement, never bit-exact (renderers differ). Exits 0 PASS / 1 FAIL / 77 SKIP.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from typing import TYPE_CHECKING, cast

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit, step, steps_for_seconds

if TYPE_CHECKING:
    from holosoma.config_types.sensor import CameraSensorConfig


def _red_mask(env_rgb):
    """Boolean [H,W] mask of red-panel pixels (R clearly dominant), same rule as the RGB test."""
    r, g, b = env_rgb[..., 0].to(int), env_rgb[..., 1].to(int), env_rgb[..., 2].to(int)
    return (r > g + 40) & (r > b + 40)


def _check_depth(rgb, depth, cam_to_panel, panel_half, fov_deg, label, *, expect_infinite_no_hit: bool):
    """Depth shape/dtype + metric assertions on one env's frame. Returns failure strings."""
    import torch

    fails = []
    h, w, _ = rgb.shape
    if depth.shape != (h, w, 1) or depth.dtype != torch.float32:
        return [f"{label}: depth shape/dtype {tuple(depth.shape)} {depth.dtype} != ({h},{w},1) float32"]

    dep = depth[..., 0].float()
    mask = _red_mask(rgb)
    if int(mask.sum()) < 0.02 * h * w:
        return [f"{label}: panel barely visible in RGB ({int(mask.sum())} px); cannot locate depth surface"]

    panel_depth = dep[mask]
    finite = panel_depth[torch.isfinite(panel_depth)]
    if finite.numel() < 0.5 * panel_depth.numel():
        return [f"{label}: panel depth mostly non-finite ({finite.numel()}/{panel_depth.numel()} finite)"]
    med = float(finite.median())
    # Inlier core: pixels within 20% of the median, excluding anti-aliased edge pixels whose depth
    # falls through to the background. Used for the metric and image-plane spread checks.
    inliers = finite[(finite - med).abs() <= 0.2 * cam_to_panel]

    # 2. Metric: panel depth ~= camera-to-panel face distance, 15% tolerance.
    if abs(med - cam_to_panel) > 0.15 * cam_to_panel:
        fails.append(
            f"{label}: panel median depth {med:.3f}m != expected {cam_to_panel:.3f}m (>15%); "
            f"per-backend depth NOT normalized to meters"
        )

    # 3. The "none" camera supplies a known no-return mask for the other policy cameras below.
    if expect_infinite_no_hit:
        bg = dep[~mask]
        if not bool(torch.isinf(bg).any()):
            fails.append(f"{label}: no +inf no-hit pixels in the background (sentinel missing or finite-far)")
        bg_finite = bg[torch.isfinite(bg)]
        bg_ref = float(bg_finite.median()) if bg_finite.numel() else math.inf
        if bg_ref <= cam_to_panel * 1.2:
            fails.append(
                f"{label}: finite background depth {bg_ref:.3f}m not clearly beyond the panel {cam_to_panel:.3f}m "
                f"(depth inverted/uncalibrated)"
            )

    # 4. Image-plane: a fronto-parallel flat panel reads ~constant depth (distance-to-camera would
    #    grow toward the edges by 1/cos(angle)). Inlier-core spread must be small relative to distance.
    spread = float(inliers.max() - inliers.min())
    if spread > 0.10 * cam_to_panel:
        fails.append(
            f"{label}: panel depth spread {spread:.3f}m > 10% of {cam_to_panel:.3f}m -> not image-plane "
            f"(looks distance-to-camera/radial)"
        )
    return fails


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
    # The clipping rig has otherwise-identical cameras with every shared no-return policy.
    config = build_run_sim_config(
        sim_arg, "panel-target", args.robot, args.terrain, sensors=_camera_presets.front_cam_depth_clipping
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
    sim.install_plugins()

    # Pin the robot upright at the origin so the camera aligns with the panel ahead.
    robot_states = sim.get_actor_states(["robot"], torch.arange(n, device=device)).clone()
    robot_states[:, :3] = env_origins + torch.tensor(list(init.pos), device=device)
    robot_states[:, 3:7] = torch.tensor(list(init.rot), device=device)
    robot_states[:, 7:] = 0.0
    sim.set_actor_states(["robot"], torch.arange(n, device=device), robot_states)
    step(sim, 2)

    if not sim.get_sensor_names():
        print(f"[{args.simulator}] FAIL: no sensors created")
        return 1

    step(sim, max(2, steps_for_seconds(sim, 0.05)))
    sim.render_sensors()

    fails: list[str] = []
    cameras = cast("dict[str, CameraSensorConfig]", config.sensor)
    depth_outputs = {}
    rgb_outputs = {}
    for cam_name, cam in cameras.items():
        rgb_all = sim.get_camera_data(cam_name, "rgb")  # [N,H,W,3] uint8
        depth_all = sim.get_camera_data(cam_name, "depth")  # [N,H,W,1] float32 meters
        rgb_outputs[cam_name] = rgb_all
        depth_outputs[cam_name] = depth_all
        if str(rgb_all.device) != str(sim.sim_device) or str(depth_all.device) != str(sim.sim_device):
            fails.append(
                f"{args.simulator}/{cam_name}: camera buffers on {rgb_all.device}/{depth_all.device} "
                f"!= sim_device {sim.sim_device}"
            )

        cam_to_panel = _camera_presets._PANEL_DISTANCE - cam.mount.position[0] - 0.01
        print(
            f"[{args.simulator}/{cam_name}] depth shape={tuple(depth_all.shape)} dtype={depth_all.dtype} "
            f"expected_panel_depth={cam_to_panel:.3f}m"
        )
        if cam_name == "short_max":
            expected_shape = (n, cam.height, cam.width, 1)
            if tuple(depth_all.shape) != expected_shape or depth_all.dtype != torch.float32:
                fails.append(
                    f"{args.simulator}/{cam_name}: depth shape/dtype {tuple(depth_all.shape)} {depth_all.dtype} "
                    f"!= {expected_shape} float32"
                )
            continue

        for e in range(n):
            env_fails = _check_depth(
                rgb_all[e],
                depth_all[e],
                cam_to_panel,
                _camera_presets._PANEL_HALF_SIZE,
                cam.vertical_fov,
                f"{args.simulator}/{cam_name}/env{e}",
                expect_infinite_no_hit=cam_name == "none",
            )
            for failure in env_fails:
                print(f"[{args.simulator}] FAIL: {failure}")
            fails += env_fails

    # The raw "none" camera gives a scene-derived no-return mask. Each policy camera must report
    # its exact sentinel at those same pixels, while the short-range camera also clips the known
    # panel return. This catches a backend that bypasses CameraRecord's shared clipping boundary.
    for e in range(n):
        none_rgb = rgb_outputs["none"][e]
        none_depth = depth_outputs["none"][e, ..., 0]
        panel_mask = _red_mask(none_rgb)
        no_return_mask = ~panel_mask & torch.isinf(none_depth)
        if not bool(no_return_mask.any()):
            fails.append(f"{args.simulator}/env{e}: no scene-derived no-return pixels for clipping checks")
            continue

        for cam_name, sentinel in (("max", cameras["max"].far), ("zero", 0.0)):
            actual = depth_outputs[cam_name][e, ..., 0][no_return_mask]
            expected = torch.full_like(actual, sentinel)
            if not torch.allclose(actual, expected, atol=1e-5, rtol=0.0):
                fails.append(
                    f"{args.simulator}/{cam_name}/env{e}: no-return pixels do not use the configured "
                    f"{cameras[cam_name].depth_clipping_behavior!r} sentinel"
                )

        short_panel = depth_outputs["short_max"][e, ..., 0][panel_mask]
        expected_short = torch.full_like(short_panel, cameras["short_max"].far)
        if not torch.allclose(short_panel, expected_short, atol=1e-5, rtol=0.0):
            fails.append(
                f"{args.simulator}/short_max/env{e}: the panel beyond this camera's far plane was not re-clipped"
            )

    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        return 1
    print(f"[{args.simulator}] PASS: depth correct across {n} env(s)")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
