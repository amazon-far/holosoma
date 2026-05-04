"""Render one mp4 per motion for a multi-terrain checkpoint in a single session.

Starts IsaacSim once, loads the checkpoint, iterates every .npz motion in the
motion_library, sets env 0 to play that motion on its matching terrain tile,
records the playback as an individual mp4, and renames the file so the output
maps 1:1 to the input motion directory.

Designed for videomimic multi-terrain training (bind_terrain_tiles=True,
pool_ordered=True), so env_origins must track the tile grid column of whichever
motion env 0 is currently playing.

Usage:
    python src/holosoma/holosoma/render_per_motion.py \
        --checkpoint /path/to/model.pt \
        --motion_dir src/.../videomimic_captures_29dof \
        --output_dir videos/model_10000
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from loguru import logger

from holosoma.simulator.shared.camera_controller import CameraParameters
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_saved_experiment_config,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment


# Azimuths (degrees) we consider when looking for an unoccluded camera
# direction. 12 candidates at 30° spacing give dense coverage without making
# per-frame raycasts expensive.
_CAMERA_AZIMUTH_CANDIDATES = list(range(0, 360, 30))


def _load_tile_mesh_world(npz_path: str, env_origin: np.ndarray) -> trimesh.Trimesh | None:
    """Load the tile mesh embedded in the motion npz and shift to world coords."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "mesh_vertices" not in data.files or "mesh_faces" not in data.files:
            return None
        verts = np.asarray(data["mesh_vertices"], dtype=np.float64) + env_origin
        faces = np.asarray(data["mesh_faces"], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        return mesh
    except Exception as exc:
        logger.warning(f"Could not load tile mesh from {npz_path}: {exc}")
        return None


def _pick_unoccluded_azimuth(
    mesh: trimesh.Trimesh | None,
    target: np.ndarray,
    distance: float,
    elevation_deg: float,
    prev_az_deg: float | None,
    hysteresis_margin: float = 0.3,
) -> float:
    """Pick the azimuth whose line-of-sight from target to camera is longest.

    Uses the prior azimuth with a small bonus so the camera doesn't jitter
    between equally-good directions every frame.
    """
    if mesh is None:
        return prev_az_deg if prev_az_deg is not None else 135.0

    elev_rad = math.radians(elevation_deg)
    horiz = distance * math.cos(elev_rad)
    vert = distance * math.sin(elev_rad)

    # Lift the ray origin slightly above the tracked body so we do not start
    # inside the robot's own collision capsule.
    origin = target + np.array([0.0, 0.0, 0.15])

    best_az = prev_az_deg if prev_az_deg is not None else _CAMERA_AZIMUTH_CANDIDATES[0]
    best_score = -np.inf

    origins = []
    directions = []
    for az in _CAMERA_AZIMUTH_CANDIDATES:
        az_rad = math.radians(az)
        cam_pos = origin + np.array([
            horiz * math.cos(az_rad),
            horiz * math.sin(az_rad),
            vert,
        ])
        direction = cam_pos - origin
        length = np.linalg.norm(direction)
        if length <= 1e-6:
            continue
        origins.append(origin)
        directions.append(direction / length)

    if not origins:
        return best_az

    origins_arr = np.asarray(origins)
    directions_arr = np.asarray(directions)

    try:
        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=origins_arr,
            ray_directions=directions_arr,
            multiple_hits=False,
        )
    except Exception as exc:
        logger.debug(f"Raycast failed: {exc}")
        return best_az

    clear_distance = np.full(len(origins_arr), distance, dtype=np.float64)
    for hit_loc, ri in zip(locations, index_ray):
        d = float(np.linalg.norm(hit_loc - origins_arr[ri]))
        if d < clear_distance[ri]:
            clear_distance[ri] = d

    for i, az in enumerate(_CAMERA_AZIMUTH_CANDIDATES):
        if i >= len(clear_distance):
            continue
        score = clear_distance[i]
        if prev_az_deg is not None and az == prev_az_deg:
            score += hysteresis_margin
        if score > best_score:
            best_score = score
            best_az = az

    return best_az


class _AutoCameraController:
    """Mutable state for the occlusion-aware camera used during rendering."""

    def __init__(self, distance: float, elevation_deg: float, target_height: float,
                 smoothing: float = 0.85):
        self.distance = distance
        self.elevation_deg = elevation_deg
        self.target_height = target_height
        self.smoothing = smoothing
        self.mesh: trimesh.Trimesh | None = None
        self.current_azimuth: float | None = None
        self.smoothed_target: np.ndarray | None = None

    def set_mesh(self, mesh: trimesh.Trimesh | None) -> None:
        self.mesh = mesh
        self.smoothed_target = None  # reset smoothing per motion
        # Start from a canonical 135° azimuth unless we find something better.
        self.current_azimuth = 135.0

    def compute_parameters(self, tracked_pos: np.ndarray) -> CameraParameters:
        raw_target = tracked_pos + np.array([0.0, 0.0, self.target_height])
        if self.smoothed_target is None:
            self.smoothed_target = raw_target
        else:
            a = self.smoothing
            self.smoothed_target = a * self.smoothed_target + (1.0 - a) * raw_target

        az = _pick_unoccluded_azimuth(
            self.mesh,
            self.smoothed_target,
            self.distance,
            self.elevation_deg,
            self.current_azimuth,
        )
        self.current_azimuth = az

        elev_rad = math.radians(self.elevation_deg)
        az_rad = math.radians(az)
        horiz = self.distance * math.cos(elev_rad)
        vert = self.distance * math.sin(elev_rad)
        cam_pos = self.smoothed_target + np.array([
            horiz * math.cos(az_rad),
            horiz * math.sin(az_rad),
            vert,
        ])
        return CameraParameters(
            position=(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])),
            target=(
                float(self.smoothed_target[0]),
                float(self.smoothed_target[1]),
                float(self.smoothed_target[2]),
            ),
            distance=self.distance,
            azimuth=az,
            elevation=self.elevation_deg,
        )


def _motion_name(npz_path: str) -> str:
    """Produce a stable, filesystem-friendly name for the motion's mp4."""
    parent = Path(npz_path).parent.name
    stem = Path(npz_path).stem
    name = parent if parent and parent != "." else stem
    return name.replace(os.sep, "_")


def _move_latest_video(video_dir: Path, dst_path: Path, start_time: float) -> Path | None:
    """Find the mp4 the recorder just wrote (mtime > start_time) and rename it."""
    candidates = [
        p for p in video_dir.glob("*.mp4")
        if p.stat().st_mtime >= start_time and not p.name.startswith("temp_raw_")
    ]
    if not candidates:
        logger.warning(f"No new mp4 found under {video_dir} since {start_time}")
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()
    shutil.move(str(latest), str(dst_path))
    return dst_path


def _setup_env_for_motion(env, motion_idx: int) -> dict:
    """Point env 0 at motion_idx on its terrain tile, initialize the robot,
    clear observation history, and return a fresh obs_dict."""
    mc = env.command_manager.get_state("motion_command")
    lib = mc.motion_library

    mc.motion_indices[0] = motion_idx
    mc.time_steps[0] = 0

    tile_grid = getattr(mc, "_tile_grid", None)
    if tile_grid is not None:
        n_cols = tile_grid.shape[1]
        col = min(motion_idx, n_cols - 1)
        env.simulator.scene.env_origins[0] = tile_grid[0, col]

    mi = mc.motion_indices[:1]
    ts = mc.time_steps[:1]

    root_pos = lib.body_pos_w(mi, ts)[:, 0].clone()
    root_rot = lib.body_quat_w(mi, ts)[:, 0].clone()
    root_lin_vel = lib.body_lin_vel_w(mi, ts)[:, 0].clone()
    root_ang_vel = lib.body_ang_vel_w(mi, ts)[:, 0].clone()
    dof_pos = lib.joint_pos(mi, ts).clone()
    dof_vel = lib.joint_vel(mi, ts).clone()

    if tile_grid is not None:
        root_pos = root_pos + env.simulator.scene.env_origins[:1]

    # Apply init_root_offset (e.g. lift robot 10 cm above contact to avoid
    # interpenetration with terrain at spawn). Mirrors training config.
    init_pose_cfg = getattr(mc, "init_pose_cfg", None)
    init_root_offset = getattr(init_pose_cfg, "init_root_offset", None) if init_pose_cfg else None
    if init_root_offset is not None:
        offset = torch.tensor(list(init_root_offset), device=env.device, dtype=root_pos.dtype)
        root_pos = root_pos + offset

    env.simulator.dof_pos[0] = dof_pos[0]
    env.simulator.dof_vel[0] = dof_vel[0]
    env.simulator.robot_root_states[0, :3] = root_pos[0]
    env.simulator.robot_root_states[0, 3:7] = root_rot[0]
    env.simulator.robot_root_states[0, 7:10] = root_lin_vel[0]
    env.simulator.robot_root_states[0, 10:13] = root_ang_vel[0]

    env_ids = torch.arange(env.num_envs, device=env.device)
    env.simulator.set_actor_root_state_tensor_robots(env_ids, env.simulator.robot_root_states)
    env.simulator.set_dof_state_tensor_robots(env_ids, env.simulator.dof_state)

    env.episode_length_buf[0] = 0
    env.reset_buf[0] = 0
    env.time_out_buf[0] = 0

    # Clear observation history for env 0 so the policy doesn't see stale
    # frames from the previous motion, then recompute obs from the new state.
    reset_ids = torch.zeros(1, dtype=torch.long, device=env.device)
    env.observation_manager.reset(reset_ids)
    return env.observation_manager.compute()


def _run_motion(env, algo, motion_idx: int, obs_dict: dict, max_safety_steps: int = 2000) -> int:
    """Roll out policy for env 0 on the current motion. Return step count."""
    for key in obs_dict:
        obs_dict[key] = obs_dict[key].to(algo.device)

    eval_policy = algo.get_inference_policy(device=str(algo.device))
    mc = env.command_manager.get_state("motion_command")
    motion_total = int(mc.motion_library.time_step_totals[motion_idx].item())

    prev_ts = 0
    near_end = False
    step = 0
    while step < max_safety_steps:
        with torch.inference_mode():
            actions = eval_policy(obs_dict)

        actor_state = {"actions": actions}
        obs_dict, _rewards, dones, _infos = env.step(actor_state)
        for key in obs_dict:
            obs_dict[key] = obs_dict[key].to(algo.device)
        step += 1

        cur_ts = int(mc.time_steps[0].item())
        if cur_ts >= motion_total - 2:
            near_end = True
        if near_end and cur_ts < prev_ts - 5:
            logger.info(f"[motion {motion_idx}] completed naturally at step {step}")
            break
        if bool(dones[0].item()) and step > 1:
            logger.info(f"[motion {motion_idx}] terminated early at step {step}")
            break
        prev_ts = cur_ts
    return step


def main() -> None:
    init_eval_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion_dir", default=None)
    parser.add_argument("--output_dir", required=True, help="Where to write per-motion mp4s.")
    parser.add_argument("--max_motions", type=int, default=None,
                        help="Cap number of motions to render (debug).")
    parser.add_argument("--playback_rate", type=float, default=1.0)
    parser.add_argument("--video_width", type=int, default=960)
    parser.add_argument("--video_height", type=int, default=540)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cfg = CheckpointConfig(checkpoint=args.checkpoint)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)

    # num_envs=1 keeps PhysX memory modest; we still need the whole 130-column
    # tile grid loaded so env 0 can teleport between tiles per motion.
    # disable_logger=False keeps our overridden video config alive through
    # get_eval_config() (which otherwise swaps logger for DisabledLoggerConfig).
    saved_cfg = dataclasses.replace(
        saved_cfg,
        eval_overrides=dataclasses.replace(
            saved_cfg.eval_overrides,
            num_envs=1,
            headless=True,
            disable_logger=False,
        ),
    )

    # Enable video recording directly into our output dir.
    from holosoma.config_types.video import VideoConfig, SphericalCameraConfig
    # High-elevation camera so walls/stairs rarely occlude the robot.
    # ~55° elevation, 4 m distance => camera lands ~3.3 m above the tracked
    # body and ~2.3 m horizontally offset, giving a consistent 3/4 aerial view.
    video_cfg = VideoConfig(
        enabled=True,
        interval=1,
        width=args.video_width,
        height=args.video_height,
        playback_rate=args.playback_rate,
        output_format="mp4",
        save_dir=str(output_dir),
        upload_to_wandb=False,
        show_command_overlay=False,
        record_env_id=0,
        camera=SphericalCameraConfig(
            distance=4.0,
            azimuth=135.0,
            elevation=55.0,
            smoothing=0.9,
        ),
    )
    saved_cfg = dataclasses.replace(
        saved_cfg,
        logger=dataclasses.replace(saved_cfg.logger, video=video_cfg),
    )

    eval_cfg = saved_cfg.get_eval_config()

    # Training decimates terrain tiles to 3k faces per tile to save GPU memory.
    # For rendering we want the original mesh so stairs don't look like ramps.
    new_terrain_term = dataclasses.replace(
        eval_cfg.terrain.terrain_term, obj_max_faces_per_tile=0
    )
    eval_cfg = dataclasses.replace(
        eval_cfg,
        terrain=dataclasses.replace(eval_cfg.terrain, terrain_term=new_terrain_term),
    )
    logger.info("Overriding terrain.obj_max_faces_per_tile -> 0 (full-resolution meshes)")

    # Override motion_dir, preserving dataclass type so base_task auto-sync works.
    if args.motion_dir is not None:
        motion_config = eval_cfg.command.setup_terms["motion_command"].params["motion_config"]
        if isinstance(motion_config, dict):
            motion_config["motion_dir"] = args.motion_dir
        else:
            new_mc = dataclasses.replace(motion_config, motion_dir=args.motion_dir)
            eval_cfg.command.setup_terms["motion_command"].params["motion_config"] = new_mc
        logger.info(f"Overriding motion_dir -> {args.motion_dir}")

    env, device, simulation_app = setup_simulation_environment(eval_cfg)

    # Load algo + checkpoint.
    algo_class = get_class(eval_cfg.algo._target_)
    algo = algo_class(
        device=device,
        env=env,
        config=eval_cfg.algo.config,
        log_dir=str(output_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
    algo.load(str(args.checkpoint))
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    if hasattr(env, "set_is_evaluating"):
        env.set_is_evaluating()
    env.reset_all()

    mc = env.command_manager.get_state("motion_command")
    npz_paths = list(mc.motion_library._all_npz_files)
    num_motions = len(npz_paths)
    if args.max_motions is not None:
        num_motions = min(num_motions, args.max_motions)
    logger.info(f"Rendering {num_motions} motions -> {output_dir}")

    recorder = env.simulator.video_recorder
    assert recorder is not None, "Video recorder not initialized; is logger.video.enabled set?"

    # Install an occlusion-aware camera controller: at every frame we pick the
    # azimuth whose line-of-sight from the robot is least blocked by the tile
    # mesh, so walls/stairs rarely hide the humanoid.
    auto_cam = _AutoCameraController(
        distance=video_cfg.camera.distance,
        elevation_deg=video_cfg.camera.elevation,
        target_height=0.5,
        smoothing=0.85,
    )

    def _auto_get_camera_parameters(robot_pos=None):
        rec_env = recorder.config.record_env_id
        root = env.simulator.robot_root_states[rec_env, :3].detach().cpu().numpy()
        return auto_cam.compute_parameters(root)

    recorder._get_camera_parameters = _auto_get_camera_parameters

    for motion_idx in range(num_motions):
        motion_path = npz_paths[motion_idx]
        name = _motion_name(motion_path)
        dst = output_dir / f"{name}.mp4"
        logger.info(f"[{motion_idx + 1}/{num_motions}] rendering '{name}'")

        obs_dict = _setup_env_for_motion(env, motion_idx)

        # Load the per-tile mesh shifted to this tile's world origin so the
        # auto camera can raycast against real world-frame geometry.
        env_origin = env.simulator.scene.env_origins[0].detach().cpu().numpy()
        tile_mesh = _load_tile_mesh_world(motion_path, env_origin)
        auto_cam.set_mesh(tile_mesh)

        start_wall = time.time()
        recorder.start_recording(episode_id=motion_idx)
        try:
            _run_motion(env, algo, motion_idx, obs_dict)
        finally:
            recorder.stop_recording()

        moved = _move_latest_video(output_dir, dst, start_wall)
        if moved is not None:
            logger.info(f"  saved -> {moved}")

    if simulation_app:
        close_simulation_app(simulation_app)
    logger.info("Done.")


if __name__ == "__main__":
    main()
