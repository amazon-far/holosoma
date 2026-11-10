"""Check completed native camera renders, image freshness and held observations."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.config_types.sensor import SensorMountConfig
from holosoma.config_types.simulator import IsaacLab3Config
from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators import _camera_presets
from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit, step


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", choices=["isaacsim3"], required=True)
    parser.add_argument("--physics", choices=["physx", "newton"], required=True)
    parser.add_argument("--renderer", choices=["isaac_rtx", "ovrtx"], required=True)
    parser.add_argument("--decimations", default="1,5")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--all-camera-renders", action="store_true", help="Benchmark the SDK's original render policy.")
    parser.add_argument("--record", help="Check movie recording alongside scheduled perception cameras.")
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()
    decimations = [int(value) for value in args.decimations.split(",")]
    camera = dataclasses.replace(
        _camera_presets.front_cam_depth["front_cam"],
        width=args.resolution,
        height=args.resolution,
        mount=SensorMountConfig(
            target_kind="world",
            position=[0.1, 0.0, _camera_presets._BASE_Z],
            orientation=_camera_presets._LOOK_FORWARD_WXYZ,
        ),
    )
    sensors = {
        f"camera_{index}": dataclasses.replace(camera, update_decimation=value)
        for index, value in enumerate(decimations)
    }
    from holosoma.config_types.video import FixedCameraConfig

    config = build_run_sim_config(
        args.simulator,
        "panel-target",
        "g1-29dof",
        "terrain_locomotion_plane",
        sensors=sensors,
        record_dir=args.record,
        video_camera=FixedCameraConfig(position=[0.1, 0.0, 0.79], target=[1.0, 0.0, 0.79]),
        show_command_overlay=False,
    )
    config = dataclasses.replace(
        config,
        logger=dataclasses.replace(
            config.logger,
            video=dataclasses.replace(
                config.logger.video,
                width=args.resolution,
                height=args.resolution,
                vertical_fov=camera.vertical_fov,
                use_recording_thread=False,
            ),
        ),
    )
    config = dataclasses.replace(
        config,
        simulator=dataclasses.replace(
            config.simulator,
            config=dataclasses.replace(
                config.simulator.config,
                sim=dataclasses.replace(
                    config.simulator.config.sim,
                    fps=200,
                    control_decimation=4,
                    # Deliberately does not divide the control interval:
                    # native captures must not depend on viewer draws.
                    render_interval=7,
                    isaaclab3=IsaacLab3Config(physics_backend=args.physics, renderer=args.renderer),
                ),
            ),
        ),
        device="cuda:0",
        training=dataclasses.replace(config.training, num_envs=args.num_envs),
    )
    if args.all_camera_renders:
        from holosoma.simulator.isaacsim.backends.isaaclab3 import IsaacLab3Backend

        # Retain batching and the exact same delivered observation cadence.
        # Only the native render policy changes for this benchmark baseline.
        IsaacLab3Backend.configure_camera_cadence = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    env, device, _app = setup_simulation_environment(config, device=config.device)
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import torch

    initial = config.robot.init_state
    state = torch.tensor(
        list(initial.pos) + list(initial.rot) + list(initial.lin_vel) + list(initial.ang_vel), device=device
    )
    sim.create_envs(args.num_envs, torch.zeros(args.num_envs, 3, device=device), state)
    sim.prepare_sim()
    recorder = sim.video_recorder
    snapshots = {}
    if args.record:
        assert recorder is not None
        recorder.on_episode_start(env_id=0)
    cameras = sim.tiled_cameras
    completed = dict.fromkeys(cameras, 0)
    subscriptions = []
    if args.renderer == "isaac_rtx":
        import omni.hydratexture
        from carb.eventdispatcher import get_eventdispatcher

        for name, cam in cameras.items():
            texture = cam._render_data.render_product.hydra_texture

            def on_render(event, name=name):
                completed[name] += 1

            subscriptions.append(
                get_eventdispatcher().observe_event(
                    observer_name=f"cadence_test_{name}",
                    event_name=omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED,
                    on_event=on_render,
                    filter=texture.get_event_key(),
                )
            )
    else:
        from isaaclab.utils.version import has_kit

        assert not has_kit(), "OVRTX capture imported Kit."
        renderer = next(iter(cameras.values()))._renderer
        engine = renderer.backend.renderer._engine
        native_step = engine.step
        names = {cam._render_data.render_product_path: name for name, cam in cameras.items()}

        def observed_step(**kwargs):
            result = native_step(**kwargs)
            for path, output in result.items():
                if path in names:
                    completed[names[path]] += len(output.frames)
            return result

        # Instrument the real native operation and its completed results;
        # neither submissions nor camera bookkeeping stand in for GPU work.
        engine.step = observed_step

    ids = torch.arange(args.num_envs, device=device)
    poses = torch.tensor([[1.0, 0.0, _camera_presets._BASE_Z, 0.0, 0.0, 0.0, 1.0]], device=device).repeat(
        args.num_envs, 1
    )
    poses[:, :3] += sim.env_origins
    previous: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    previous_completed = completed.copy()
    count_before = sim.sim.get_physics_step_count()
    time_before = sim.backend.physics_time(sim.sim)
    capture_seconds = []
    warmup = 10
    for frame in range(args.frames):
        # Large, non-periodic-at-the-tested-rates jumps expose old images,
        # including the first capture, without relying on noisy pixel diffs.
        panel_y = [-0.18, 0.18, -0.06, 0.12, -0.12, 0.06, 0.0][frame % 7]
        poses[:, 1] = sim.env_origins[:, 1] + panel_y
        sim.set_static_body_pose(["panel"], ids, poses)
        step(sim, 4)
        if frame == args.frames // 2:
            # Reset one native tile without restarting the shared cadence.
            for cam in cameras.values():
                cam.reset(env_ids=[0])
        count = sim.sim.get_physics_step_count()
        torch.cuda.synchronize()
        start = time.perf_counter()
        sim.render_sensors()
        torch.cuda.synchronize()
        if frame >= warmup:
            capture_seconds.append(time.perf_counter() - start)
        if args.record:
            from tests.simulators.scene_spawn_assert import _video_panel_centroid

            recorder.capture_frame(env_id=0)
            movie_frame = recorder.video_frames[-1]
            center = _video_panel_centroid(movie_frame)[1]
            expected = (args.resolution - 1) / 2 - panel_y * args.resolution / (
                2 * math.tan(math.radians(camera.vertical_fov) / 2) * 0.89
            )
            assert abs(center - expected) < max(1.2, args.resolution * 0.02), (
                f"Movie: stale panel image at frame {frame}: column {center}, expected {expected}."
            )
            if frame in (0, args.frames - 1):
                snapshots[frame] = movie_frame.copy()
        assert sim.sim.get_physics_step_count() == count, "Capture integrated physics."
        for name, decimation in zip(cameras, decimations, strict=True):
            due = frame % decimation == 0
            rgb = sim.get_camera_data(name)
            depth = sim.get_camera_data(name, data_type="depth")
            assert rgb.shape == (args.num_envs, args.resolution, args.resolution, 3)
            assert rgb.dtype == torch.uint8
            assert depth.shape == (args.num_envs, args.resolution, args.resolution, 1)
            if due:
                red = rgb[0, ..., 0].short() > rgb[0, ..., 1].short() + 30
                assert red.any(), f"{name}: the panel was not rendered at frame {frame}."
                center = red.nonzero().float().mean(dim=0)[0].item()
                expected = (args.resolution - 1) / 2 - panel_y * args.resolution / (
                    2 * math.tan(math.radians(camera.vertical_fov) / 2) * 0.89
                )
                assert abs(center - expected) < max(1.2, args.resolution * 0.02), (
                    f"{name}: stale panel image at frame {frame}: row {center}, expected {expected}."
                )
                panel_depth = depth[0, ..., 0][red].median().item()
                assert abs(panel_depth - 0.89) < 0.03, (name, panel_depth)
            else:
                assert torch.equal(rgb, previous[name][0]), f"{name}: RGB changed between captures."
                assert torch.equal(depth, previous[name][1]), f"{name}: depth changed between captures."
            if not args.all_camera_renders:
                rendered = completed[name] - previous_completed[name]
                assert rendered == int(due), f"{name}: frame {frame}, completed {rendered}, due={due}."
            previous[name] = (rgb.clone(), depth.clone())
        previous_completed = completed.copy()
    assert sim.sim.get_physics_step_count() - count_before == args.frames * 4
    assert abs(sim.backend.physics_time(sim.sim) - time_before - args.frames / 50) < 1e-6
    if args.record:
        from tests.simulators.scene_spawn_assert import _check_saved_video

        assert len(recorder.video_frames) == args.frames
        recorder.on_episode_end(env_id=0)
        movies = list(Path(args.record).glob("**/*.mp4"))
        assert movies and all(movie.stat().st_size > 0 for movie in movies)
        for movie in movies:
            _check_saved_video(movie, snapshots, args.frames)
        recorder.cleanup()
    if args.renderer == "isaac_rtx":
        subscriptions.clear()
        renderer = next(iter(cameras.values()))._renderer
        for cam in cameras.values():
            renderer.cleanup(cam._render_data)
            renderer.cleanup(cam._render_data)
        renderer.close()
        renderer.close()
    print(
        json.dumps(
            {
                "completed_renders": completed,
                "decimations": decimations,
                "capture_seconds": sum(capture_seconds),
                "measured_cycles": len(capture_seconds),
                "num_envs": args.num_envs,
                "resolution": args.resolution,
                "policy": "all" if args.all_camera_renders else "scheduled",
            }
        ),
        flush=True,
    )
    Path(args.result_file).write_text("OK\n")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
