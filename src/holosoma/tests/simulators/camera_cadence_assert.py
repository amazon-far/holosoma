"""Check completed native camera renders, image freshness and held observations."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.config_types.sensor import SensorMountConfig
from holosoma.config_types.simulator import IsaacSim6Config
from tests.simulators import _camera_presets
from tests.simulators._sim_harness import build_run_sim_config, launch_simulator, run_and_hard_exit, step


def gpu_process_memory() -> dict[str, int]:
    """Deduplicate native graphics/compute entries for the same host process."""
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True
    )
    memory: dict[str, int] = {}
    for row in rows.splitlines():
        pid, value = row.split(",")
        if value.strip().isdigit():
            memory[pid.strip()] = max(memory.get(pid.strip(), 0), int(value.strip()))
    return memory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", choices=["isaacsim6"], required=True)
    parser.add_argument("--physics", choices=["physx", "newton"], required=True)
    parser.add_argument("--renderer", choices=["isaac_rtx", "ovrtx", "newton_warp"], required=True)
    parser.add_argument("--decimations", default="1,5")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--record", help="Check movie recording alongside perception cameras.")
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()
    if args.frames <= 10:
        parser.error("--frames must exceed the 10-frame warmup.")
    if args.num_envs <= 0 or args.resolution <= 0:
        parser.error("--num-envs and --resolution must be positive.")
    try:
        decimations = [int(value) for value in args.decimations.split(",")]
    except ValueError:
        parser.error("--decimations must be comma-separated positive integers.")
    if any(value <= 0 for value in decimations):
        parser.error("--decimations must be comma-separated positive integers.")
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

    def orientation(index: int) -> list[float]:
        # Distinct optical-axis rotations in 90-degree increments.
        angle = -index * math.pi / 4
        cosine, sine = math.cos(angle), math.sin(angle)
        c, _, pitch, _ = camera.mount.orientation
        return [c * cosine, pitch * sine, pitch * cosine, c * sine]

    sensors = {
        f"camera_{index}": dataclasses.replace(
            camera,
            update_decimation=value,
            mount=dataclasses.replace(
                camera.mount,
                orientation=orientation(index),
            ),
        )
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
                    isaacsim6=IsaacSim6Config(physics_backend=args.physics, renderer=args.renderer),
                ),
            ),
        ),
        device="cuda:0",
        training=dataclasses.replace(config.training, num_envs=args.num_envs),
    )
    # Keep the simulated robot behind the camera. Its falling limbs can
    # otherwise occlude the panel and shift the measured red centroid.
    initial_pos = config.robot.init_state.pos
    config = dataclasses.replace(
        config,
        robot=dataclasses.replace(
            config.robot,
            init_state=dataclasses.replace(
                config.robot.init_state, pos=[initial_pos[0] - 5, initial_pos[1], initial_pos[2]]
            ),
        ),
    )
    if args.record:
        from holosoma.simulator.isaacsim.backends.isaacsim6 import IsaacSim6Backend

        original_create_camera_cfg = IsaacSim6Backend.create_camera_cfg

        def movie_depth_cfg(self: Any, **kwargs: Any) -> Any:
            # Test-only AOV on the same movie product: locate the current surface
            # independently of the default renderer's temporal RGB trails.
            if kwargs["prim_path"].endswith("/holosoma_video_camera"):
                kwargs["data_types"] = ["rgb", "distance_to_image_plane"]
            return original_create_camera_cfg(self, **kwargs)

        IsaacSim6Backend.create_camera_cfg = movie_depth_cfg  # type: ignore[method-assign]

    memory_before = gpu_process_memory()
    startup_start = time.perf_counter()
    sim, device, _app = launch_simulator(config, device=config.device)
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

    def physics_state() -> tuple[torch.Tensor, ...]:
        """Read native dynamics directly so SDK clock bookkeeping cannot hide integration."""
        import warp as wp

        if args.physics == "newton":
            state = sim.sim.physics_manager.get_state_0()
            return tuple(
                wp.to_torch(getattr(state, name)).clone() for name in ("body_q", "body_qd", "joint_q", "joint_qd")
            )
        view = sim._robot.root_view
        if args.renderer == "isaac_rtx":
            return tuple(
                wp.to_torch(getattr(view, name)()).clone()
                for name in ("get_root_transforms", "get_root_velocities", "get_dof_positions", "get_dof_velocities")
            )
        from isaaclab_ov import tensor_types as TT

        return tuple(
            wp.to_torch(view.get_attribute(attribute)).clone()
            for attribute in (TT.ROOT_POSE, TT.ROOT_VELOCITY, TT.DOF_POSITION, TT.DOF_VELOCITY)
        )

    startup_seconds = time.perf_counter() - startup_start
    recorder = sim.video_recorder
    snapshots: dict[int, Any] = {}
    if args.record:
        assert recorder is not None
        origin = sim.env_origins[recorder.config.record_env_id].cpu().tolist()
        recorder.camera_controller.config = FixedCameraConfig(
            # Distinguish the movie view from every perception camera.
            position=[origin[0] + 0.1, origin[1] + 0.06, origin[2] + 0.83],
            target=[origin[0] + 1.0, origin[1] + 0.06, origin[2] + 0.83],
        )
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
    elif args.renderer == "ovrtx":
        from isaaclab.utils.version import has_kit

        assert not has_kit(), "OVRTX capture imported Kit."
        renderer = next(iter(cameras.values()))._renderer
        engine = renderer.backend.renderer
        names = {cam._render_data.render_product_path: name for name, cam in cameras.items()}
        native_step = engine.step

        def observed_step(**kwargs):
            result = native_step(**kwargs)
            for path, output in result.items():
                if path in names:
                    completed[names[path]] += len(output.frames)
            return result

        engine.step = observed_step
    else:
        renderer = next(iter(cameras.values()))._renderer
        native_render = renderer.render
        names = {id(cam._render_data): name for name, cam in cameras.items()}

        def observed_render(data):
            native_render(data)
            if id(data) in names:
                completed[names[id(data)]] += 1

        renderer.render = observed_render

    ids = torch.arange(args.num_envs, device=device)
    poses = torch.tensor([[1.0, 0.0, _camera_presets._BASE_Z, 0.0, 0.0, 0.0, 1.0]], device=device).repeat(
        args.num_envs, 1
    )
    poses[:, :3] += sim.env_origins
    previous: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    borrowed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    previous_completed = completed.copy()
    count_before = sim.sim.get_physics_step_count()
    time_before = sim.backend.physics_time(sim.sim)
    capture_seconds = []
    measured_completed = completed.copy()
    warmup = 10
    for frame in range(args.frames):
        # Large, non-periodic-at-the-tested-rates jumps expose old images,
        # including the first capture, without relying on noisy pixel diffs.
        panel_ys = [[-0.18, 0.18, -0.06, 0.12, -0.12, 0.06, 0.0][(frame + tile) % 7] for tile in range(args.num_envs)]
        panel_distances = [0.89 + 0.02 * ((frame + 3 * tile) % 11 - 5) for tile in range(args.num_envs)]
        poses[:, 0] = sim.env_origins[:, 0] + torch.tensor(panel_distances, device=device) + 0.11
        poses[:, 1] = sim.env_origins[:, 1] + torch.tensor(panel_ys, device=device)
        sim.set_static_body_pose(["panel"], ids, poses)
        step(sim, 4)
        if frame == args.frames // 2:
            # Reset one native tile without restarting the shared cadence.
            for cam in cameras.values():
                cam.reset(env_ids=[0])
        count = sim.sim.get_physics_step_count()
        dynamics_before = physics_state()
        torch.cuda.synchronize()
        start = time.perf_counter()
        sim.render_sensors()
        torch.cuda.synchronize()
        if frame >= warmup:
            capture_seconds.append(time.perf_counter() - start)
        if args.record:
            recorder.capture_frame(env_id=0)
            movie_frame = recorder.video_frames[-1]
            movie_depth = recorder.camera.data.output["distance_to_image_plane"].torch[0, ..., 0]
            scale = args.resolution / (2 * math.tan(math.radians(camera.vertical_fov) / 2) * panel_distances[0])
            expected = torch.tensor(
                [(args.resolution - 1) / 2 + 0.04 * scale, (args.resolution - 1) / 2 - (panel_ys[0] - 0.06) * scale],
                device=device,
            )
            panel = (movie_depth - panel_distances[0]).abs() < 0.015
            movie_rgb = torch.as_tensor(movie_frame, device=device)
            red = movie_rgb[..., 0].short() > movie_rgb[..., 1].short() + 30
            current = panel & red
            assert panel.any() and current.any(), f"Movie: no current surface at frame {frame}."
            tolerance = max(1.2, args.resolution * 0.02)
            for label, mask in (("depth", panel), ("RGB", current)):
                center = mask.nonzero().float().mean(dim=0)
                assert (center - expected).abs().max().item() < tolerance, (
                    f"Movie: stale {label} at frame {frame}: {center.tolist()}, expected {expected.tolist()}."
                )
            if frame == 1:
                old_rgb = torch.as_tensor(snapshots[0], device=device)
                stale = (old_rgb[..., 0].short() > old_rgb[..., 1].short() + 30) & panel
                assert (
                    not stale.any() or (stale.nonzero().float().mean(dim=0) - expected).abs().max().item() >= tolerance
                ), "Movie freshness criterion accepted the previous RGB image."
            if frame in (0, args.frames - 1):
                snapshots[frame] = movie_frame.copy()
        assert sim.sim.get_physics_step_count() == count, "Capture integrated physics."
        for before, after in zip(dynamics_before, physics_state(), strict=True):
            assert torch.equal(before, after), "Rendering changed native dynamics without a physics step."
        shared_capture = args.record is not None or any(frame % value == 0 for value in decimations)
        for name, decimation in zip(cameras, decimations, strict=True):
            due = frame % decimation == 0
            rgb = sim.get_camera_data(name)
            depth = sim.get_camera_data(name, data_type="depth")
            assert rgb.shape == (args.num_envs, args.resolution, args.resolution, 3)
            assert rgb.dtype == torch.uint8
            assert depth.shape == (args.num_envs, args.resolution, args.resolution, 1)
            if name in borrowed:
                assert torch.equal(borrowed[name][0], previous[name][0]), f"{name}: retained RGB buffer mutated."
                assert torch.equal(borrowed[name][1], previous[name][1]), f"{name}: retained depth buffer mutated."
            if due:
                index = int(name.removeprefix("camera_"))
                axis = 1 if index % 2 else 0
                tolerance = max(1.2, args.resolution * 0.02)
                for tile, (panel_y, distance) in enumerate(zip(panel_ys, panel_distances, strict=True)):
                    label = f"{name}/env{tile}/frame{frame}"
                    red = rgb[tile, ..., 0].short() > rgb[tile, ..., 1].short() + 30
                    assert red.any(), f"{label}: the panel was not rendered."
                    direction = -1 if index % 4 < 2 else 1
                    expected = (args.resolution - 1) / 2 + direction * panel_y * args.resolution / (
                        2 * math.tan(math.radians(camera.vertical_fov) / 2) * distance
                    )
                    # Color trails can extend beyond today's silhouette. Check
                    # moving depth independently, then RGB on current geometry.
                    panel = (depth[tile, ..., 0] - distance).abs() < 0.015
                    assert panel.any(), f"{label}: no current panel depth ({distance} m)."
                    depth_center = panel.nonzero().float().mean(dim=0)[axis].item()
                    assert abs(depth_center - expected) < tolerance, (
                        f"{label}: stale panel depth: {depth_center}, expected {expected}."
                    )
                    current_red = red & panel
                    assert current_red.any(), f"{label}: no red pixels on current geometry."
                    center = current_red.nonzero().float().mean(dim=0)[axis].item()
                    assert abs(center - expected) < tolerance, (
                        f"{label}: stale panel image: axis {axis} coordinate {center}, expected {expected}."
                    )
                    if frame == decimation and tile == 0:
                        # Previous real RGB paired with fresh depth must fail.
                        old_rgb = previous[name][0][tile]
                        stale_red = (old_rgb[..., 0].short() > old_rgb[..., 1].short() + 30) & panel
                        if stale_red.any():
                            stale_center = stale_red.nonzero().float().mean(dim=0)[axis].item()
                            assert abs(stale_center - expected) >= tolerance, (
                                f"{label}: freshness criterion accepted previous RGB."
                            )
                    panel_depth = depth[tile, ..., 0][current_red].median().item()
                    assert abs(panel_depth - distance) < 0.015, (label, panel_depth, distance)
            else:
                assert torch.equal(rgb, previous[name][0]), f"{name}: RGB changed between captures."
                assert torch.equal(depth, previous[name][1]), f"{name}: depth changed between captures."
            rendered = completed[name] - previous_completed[name]
            expected_capture = shared_capture if sim.backend.shared_camera_rendering else due
            if args.renderer == "isaac_rtx" and frame < warmup and expected_capture:
                # The ordinary SDK pumps extra frames while textures stream.
                # Keep all pixel checks strict and record every completed draw.
                assert rendered >= 1, f"{name}: no completed startup capture at frame {frame}."
            else:
                assert rendered == int(expected_capture), (
                    f"{name}: frame {frame}, completed {rendered}, expected={expected_capture}, due={due}."
                )
            borrowed[name] = (rgb, depth)
            previous[name] = (rgb.clone(), depth.clone())
        previous_completed = completed.copy()
        if frame == warmup - 1:
            measured_completed = completed.copy()
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
    memory_after = gpu_process_memory()
    subscriptions.clear()
    # STOP invalidates scene-owned camera data before the registry closes its
    # renderers. Direct renderer.cleanup is not an idempotent SDK lifecycle API.
    sim.sim.stop()
    sim.sim.clear_instance()
    sim.sim.clear_instance()
    assert all(not cam.is_initialized for cam in cameras.values())
    for name, (rgb, depth) in borrowed.items():
        assert torch.equal(rgb, previous[name][0]), f"{name}: RGB did not survive SDK teardown."
        assert torch.equal(depth, previous[name][1]), f"{name}: depth did not survive SDK teardown."
    print(
        json.dumps(
            {
                "completed_renders": completed,
                "measured_completed_renders": {
                    name: count - measured_completed[name] for name, count in completed.items()
                },
                "decimations": decimations,
                "capture_seconds": sum(capture_seconds),
                "measured_cycles": len(capture_seconds),
                "capture_p50_ms": statistics.median(capture_seconds) * 1000,
                "capture_p95_ms": sorted(capture_seconds)[math.ceil(0.95 * len(capture_seconds)) - 1] * 1000,
                "startup_seconds": startup_seconds,
                "gpu_process_memory_mib": sum(
                    memory for pid, memory in memory_after.items() if pid not in memory_before
                ),
                "num_envs": args.num_envs,
                "resolution": args.resolution,
                "policy": "shared" if sim.backend.shared_camera_rendering else "independent",
            }
        ),
        flush=True,
    )
    Path(args.result_file).write_text("OK\n")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
