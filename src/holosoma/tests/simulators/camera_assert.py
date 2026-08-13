"""Cross-backend camera-sensor assertion harness.

Builds a sim under one backend, creates the mounted camera(s) from the ``--sensors`` preset,
renders, and asserts:

  1. ``get_camera_data`` returns shape ``[num_envs, H, W, 3]``, dtype uint8,
     values in ``[0, 255]``;
  2. the rendered frame is not uniformly blank;
  3. the visualization recorder wiring, when ``--check-recorder`` is set.

Centering, optical-axis, and FOV validation live in ``camera_geometry_assert.py``.

Run per backend (each in its own env):
    python tests/simulators/camera_assert.py --simulator mujoco --scene <preset> --sensors <preset>
    python tests/simulators/camera_assert.py --simulator {mjwarp,isaacgym,isaacsim} ...

Exits 0 on PASS, 1 on FAIL, 77 on SKIP.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

# Pop this dir off sys.path[0] so tests/simulators/isaacsim/ can't shadow the real isaacsim pkg.
if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators._sim_harness import build_run_sim_config, step, steps_for_seconds

SKIP_EXIT_CODE = 77


def _check_contract(img, num_envs: int, width: int, height: int, label: str) -> list[str]:
    """Shape/dtype/range assertions on a get_camera_data tensor. Returns failure messages."""
    import torch

    fails = []
    if not isinstance(img, torch.Tensor):
        return [f"{label}: get_camera_data did not return a torch.Tensor (got {type(img)})"]
    if tuple(img.shape) != (num_envs, height, width, 3):
        fails.append(f"{label}: shape {tuple(img.shape)} != expected ({num_envs},{height},{width},3)")
    if img.dtype != torch.uint8:
        fails.append(f"{label}: dtype {img.dtype} != torch.uint8")
    if img.numel() and (int(img.min()) < 0 or int(img.max()) > 255):
        fails.append(f"{label}: values out of [0,255] (min={int(img.min())}, max={int(img.max())})")
    return fails


def _check_not_blank(img, label: str) -> list[str]:
    """Return a failure message if the frame is ~uniform. Returns failure messages."""
    lum = img[0].float().mean(dim=-1)  # [H, W]
    spread = float(lum.max()) - float(lum.min())
    if spread < 1.0:
        return [f"{label}: frame is ~uniform (max-min lum {spread:.2f}); renderer saw nothing"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--scene", default="camera-target", help="scene preset (front-facing bright object)")
    parser.add_argument("--sensors", default="front-cam", help="sensors preset (one robot-base camera)")
    parser.add_argument("--robot", default="g1-29dof")
    parser.add_argument("--terrain", default="terrain_locomotion_plane")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--headless", choices=["true", "false"], default="true")
    parser.add_argument(
        "--check-recorder",
        action="store_true",
        help="install a CameraVizPlugin recorder hook (record_video) and assert it was installed + buffered "
        "a frame, regardless of backend; guards that camera-consumer hooks fire on FRAME_END.",
    )
    parser.add_argument("--result-file", default=None, help="write PASS/FAIL here before teardown")
    args = parser.parse_args()

    # Register the test-only camera scene/sensor presets into the CLI DEFAULTS.
    from tests.simulators import _camera_presets

    headless = args.headless == "true"

    sim_arg = "mujoco" if args.simulator == "mjwarp" else args.simulator
    config = build_run_sim_config(
        sim_arg, args.scene, args.robot, args.terrain, sensors=_camera_presets.SENSOR_PRESETS[args.sensors]
    )
    if args.simulator == "mjwarp":
        config = _camera_presets.as_mjwarp(config)

    device = "cuda:0" if args.simulator != "mujoco" else "cpu"
    config = dataclasses.replace(
        config,
        device=device,
        training=dataclasses.replace(config.training, num_envs=args.num_envs),
    )

    if args.check_recorder:
        # Attach a CameraVizPlugin recorder as a hook plugin with record_video (not live_window) so the
        # check runs without a DISPLAY. Keyed by an arbitrary label ("rec"); the simulator builds it
        # in __init__ and it registers its publish on FRAME_END (after the render hook).
        from holosoma.config_types.plugin import CameraVizPluginConfig

        config = dataclasses.replace(config, plugin={**config.plugin, "rec": CameraVizPluginConfig(record_video=True)})

    env, device, _app = setup_simulation_environment(config, device=device)
    sim = env.sim
    sim.set_headless(headless)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import torch

    n = args.num_envs
    env_origins = torch.zeros(n, 3, device=device)
    if n > 1:
        env_origins[:, 0] = torch.arange(n, device=device, dtype=torch.float32) * 5.0
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel),
        device=device,
        dtype=torch.float32,
    )
    sim.create_envs(n, env_origins, base_init)
    sim.prepare_sim()

    # Hooks (incl. the --check-recorder CameraVizPlugin) were installed by the simulator in __init__ from
    # FullSimConfig.plugin; nothing to install here.
    from holosoma.simulator.base_simulator.hooks import Phase

    names = sim.get_sensor_names()
    print(f"[{args.simulator}] sensors created: {names}")
    if not names:
        print(f"[{args.simulator}] FAIL: no sensors created from preset '{args.sensors}'")
        return 1

    # Step so the renderer warms up and body poses settle, then fire the control-post-refresh hooks
    # (cameras render, egress consumers publish) once — the same emission the task loop drives.
    step(sim, max(args.steps, steps_for_seconds(sim, 0.05)))
    sim.hooks.emit(Phase.FRAME_END)

    fails: list[str] = []
    cam_cfgs = dict(config.sensor)
    for name in names:
        cam = cam_cfgs[name]
        img = sim.get_camera_data(name, "rgb")
        print(
            f"[{args.simulator}] {name}: shape={tuple(img.shape)} dtype={img.dtype} "
            f"min={int(img.min())} max={int(img.max())}"
        )
        fails += _check_contract(img, n, cam.width, cam.height, f"{args.simulator}/{name}")
        fails += _check_not_blank(img, f"{args.simulator}/{name}")

    if args.check_recorder:
        # The CameraVizPlugin recorder hook (installed from the `plugin` dict) must have registered on
        # FRAME_END and buffered a frame from the emission above. Find it among the
        # installed hooks and assert it captured at least one grid.
        from holosoma.simulator.plugins.viz.viz_plugin import CameraVizPlugin

        recorders = [h for h in sim.installed_plugins.values() if isinstance(h, CameraVizPlugin)]
        if not recorders:
            fails.append(
                f"{args.simulator}: recorder requested (plugin.rec:viz-record) but no CameraVizPlugin was "
                f"installed; the simulator did not build hooks from FullSimConfig.plugin in __init__."
            )
        else:
            n_frames = max(len(getattr(r, "_frames_video", [])) for r in recorders)
            print(f"[{args.simulator}] recorder buffered {n_frames} frame(s) after one emission")
            if n_frames < 1:
                fails.append(f"{args.simulator}: recorder hook installed but buffered no frame ({n_frames}).")

    if args.result_file:
        # Persist the verdict before teardown: IsaacSim teardown can hard-kill the process and
        # swallow the exit code. "OK" on success per the run_harness sentinel convention.
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        for f in fails:
            print(f"[{args.simulator}] FAIL: {f}")
        return 1
    print(f"[{args.simulator}] PASS: {len(names)} camera(s) satisfy the shape/dtype + non-blank check")
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
