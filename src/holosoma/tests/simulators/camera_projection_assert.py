"""Cross-backend camera projection-model assertions.

Two co-located cameras view the same red panel. Isaac Sim must author the requested f-theta
calibration and produce a visibly wider fisheye image. Isaac Gym and both MuJoCo backends must
ignore the Isaac-Sim-only channel and preserve the shared 60-degree pinhole fallback.
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
from tests.simulators.camera_assert import _check_contract
from tests.simulators.camera_geometry_assert import _check_geometry, _measure, _red_mask

if TYPE_CHECKING:
    from holosoma.config_types.sensor import CameraSensorConfig, IsaacSimCameraConfig

_PINHOLE = "pinhole"
_FISHEYE = "fisheye"
_USD_FISHEYE_ATTRS = {
    "cameraProjectionType": "projection_type",
    "fthetaWidth": "nominal_width",
    "fthetaHeight": "nominal_height",
    "fthetaCx": "optical_centre_x",
    "fthetaCy": "optical_centre_y",
    "fthetaMaxFov": "max_fov",
    "fthetaPolyA": "polynomial_a",
    "fthetaPolyB": "polynomial_b",
    "fthetaPolyC": "polynomial_c",
    "fthetaPolyD": "polynomial_d",
    "fthetaPolyE": "polynomial_e",
    "fthetaPolyF": "polynomial_f",
}


def _check_isaacsim_authored_calibration(sim, config: IsaacSimCameraConfig) -> list[str]:
    """Check every cloned USD camera prim, not only the pre-spawn Python config."""
    fails = []
    native_camera = sim.tiled_cameras[_FISHEYE]
    for sensor_prim in native_camera._sensor_prims:
        prim = sensor_prim.GetPrim()
        for usd_name, config_name in _USD_FISHEYE_ATTRS.items():
            actual = prim.GetAttribute(usd_name).Get()
            if config_name == "projection_type":
                expected = config.projection_type
                config_path = config_name
            else:
                expected = getattr(config.fisheye, config_name)
                config_path = f"fisheye.{config_name}"
            if isinstance(expected, str):
                matches = actual == expected
            else:
                matches = actual is not None and math.isclose(
                    float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-7
                )
            if not matches:
                fails.append(
                    f"isaacsim/{prim.GetPath()}: {usd_name}={actual!r}, expected {expected!r} "
                    f"from isaacsim.{config_path}"
                )
    return fails


def _check_fisheye_geometry(pinhole_img, fisheye_img, label: str) -> list[str]:
    """Require the calibrated 120-degree fisheye to differ materially from the 60-degree pinhole."""
    pin = _measure(_red_mask(pinhole_img))
    fish = _measure(_red_mask(fisheye_img))
    pin_count, _pin_row, _pin_col, pin_h, pin_w = pin
    fish_count, fish_row, fish_col, fish_h, fish_w = fish
    h, w, _ = fisheye_img.shape
    fails = []

    if fish_count < 0.01 * h * w:
        return [f"{label}: fisheye panel barely visible ({fish_count}/{h * w} red px)"]
    if abs(fish_col - (w - 1) / 2) > 0.10 * w or abs(fish_row - (h - 1) / 2) > 0.10 * h:
        fails.append(f"{label}: fisheye panel centroid ({fish_row:.1f}, {fish_col:.1f}) is not centered")
    if fish_h >= 0.80 * pin_h or fish_w >= 0.80 * pin_w:
        fails.append(
            f"{label}: 120-degree fisheye panel extent {fish_h:.0f}x{fish_w:.0f} is not materially smaller "
            f"than 60-degree pinhole {pin_h:.0f}x{pin_w:.0f}; projection may have been ignored"
        )
    if fish_count >= 0.70 * pin_count:
        fails.append(
            f"{label}: fisheye panel area {fish_count}px is not materially smaller than pinhole "
            f"{pin_count}px; projection may have been ignored"
        )
    return fails


def _check_pinhole_fallback(pinhole_img, configured_img, label: str) -> list[str]:
    """Require the backend-specific channel to leave a near-identical pinhole render."""
    import torch

    pin_mask = _red_mask(pinhole_img)
    configured_mask = _red_mask(configured_img)
    mask_disagreement = float((pin_mask != configured_mask).float().mean())
    mean_abs_difference = float((pinhole_img.to(torch.float32) - configured_img.to(torch.float32)).abs().mean())
    fails = []
    if mask_disagreement > 0.01:
        fails.append(
            f"{label}: fallback red silhouettes disagree on {mask_disagreement:.3%} of pixels "
            "(expected the same pinhole projection)"
        )
    if mean_abs_difference > 1.0:
        fails.append(
            f"{label}: fallback images differ by mean {mean_abs_difference:.2f} intensity levels "
            "(expected the Isaac Sim channel to be ignored)"
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

    sim_arg = "mujoco" if args.simulator == "mjwarp" else args.simulator
    config = build_run_sim_config(
        sim_arg,
        "panel-target",
        args.robot,
        args.terrain,
        sensors=_camera_presets.projection_pair,
    )
    if args.simulator == "mjwarp":
        config = _camera_presets.as_mjwarp(config)

    device = "cuda:0" if args.simulator != "mujoco" else "cpu"
    config = dataclasses.replace(
        config,
        device=device,
        training=dataclasses.replace(config.training, num_envs=args.num_envs),
    )

    env, device, _app = setup_simulation_environment(config, device=device)
    sim = env.sim
    sim.set_headless(args.headless == "true")
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()

    import torch

    n = args.num_envs
    env_origins = torch.zeros(n, 3, device=device)
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel),
        device=device,
        dtype=torch.float32,
    )
    sim.create_envs(n, env_origins, base_init)
    sim.prepare_sim()
    sim.install_plugins()

    names = set(sim.get_sensor_names())
    fails: list[str] = []
    if names != {_PINHOLE, _FISHEYE}:
        fails.append(f"{args.simulator}: expected pinhole+fisheye cameras, got {sorted(names)}")

    # Move the robot clear of the fixed world cameras' sightline before rendering the panel.
    actual_origins = sim.env_origins
    all_ids = torch.arange(n, device=device)
    states = sim.get_actor_states(["robot"], all_ids).clone()
    states[:, :3] = actual_origins + torch.tensor(list(init.pos), device=device)
    states[:, 1] -= 1.5
    states[:, 3:7] = torch.tensor(list(init.rot), device=device)
    states[:, 7:] = 0.0
    sim.set_actor_states(["robot"], all_ids, states)

    step(sim, max(2, steps_for_seconds(sim, 0.05)))
    sim.render_sensors()

    camera_configs = cast("dict[str, CameraSensorConfig]", dict(config.sensor))
    frames = {name: sim.get_camera_data(name, "rgb") for name in (_PINHOLE, _FISHEYE)}
    for name, frame in frames.items():
        camera = camera_configs[name]
        fails += _check_contract(frame, n, camera.width, camera.height, f"{args.simulator}/{name}")

    cam_to_panel = _camera_presets._PANEL_DISTANCE - _camera_presets._PROJECTION_MOUNT.position[0] - 0.01
    pinhole_config = camera_configs[_PINHOLE]
    fisheye_config = camera_configs[_FISHEYE]
    for env_id in range(n):
        pinhole_img = frames[_PINHOLE][env_id]
        fisheye_img = frames[_FISHEYE][env_id]
        pin_measure = _measure(_red_mask(pinhole_img))
        fish_measure = _measure(_red_mask(fisheye_img))
        print(
            f"[{args.simulator}] env{env_id}: pinhole red={pin_measure[0]} extent={pin_measure[3]:.0f}x"
            f"{pin_measure[4]:.0f}; configured red={fish_measure[0]} extent={fish_measure[3]:.0f}x"
            f"{fish_measure[4]:.0f}"
        )
        fails += _check_geometry(
            pinhole_img,
            pinhole_config,
            cam_to_panel,
            _camera_presets._PANEL_HALF_SIZE,
            f"{args.simulator}/pinhole/env{env_id}",
        )
        if args.simulator == "isaacsim":
            fails += _check_fisheye_geometry(
                pinhole_img,
                fisheye_img,
                f"{args.simulator}/fisheye/env{env_id}",
            )
        else:
            fails += _check_geometry(
                fisheye_img,
                fisheye_config,
                cam_to_panel,
                _camera_presets._PANEL_HALF_SIZE,
                f"{args.simulator}/fallback/env{env_id}",
            )
            fails += _check_pinhole_fallback(
                pinhole_img,
                fisheye_img,
                f"{args.simulator}/fallback/env{env_id}",
            )

    if args.simulator == "isaacsim":
        fails += _check_isaacsim_authored_calibration(sim, fisheye_config.isaacsim)

    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        for failure in fails:
            print(f"[{args.simulator}] FAIL: {failure}")
        return 1
    print(f"[{args.simulator}] PASS: projection behavior is correct across {n} env(s)")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
