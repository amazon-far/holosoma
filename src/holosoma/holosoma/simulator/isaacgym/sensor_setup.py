"""Isaac Gym mounted camera and LiDAR sensor setup.

IsaacGym has no batched camera: one sensor is created per environment and attached to its
mount body (or fixed in the world). These module-level functions take the simulator (the
shared sensor setup convention) and drive that per-env pipeline — mount
resolution, per-env sensor creation, the optical-frame change of basis, and the render
read-out. The per-camera native-handle map lives on the simulator as ``sim.camera_handles``
(the structural twin of ``sim.object_handles``), written during the env-build loop and read at
render.

Isaac Gym has no native scene-query LiDAR API, so LiDAR uses Holosoma's terrain Warp mesh.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch
from isaacgym import gymapi, gymtorch
from loguru import logger

from holosoma.config_types.sensor import CameraSensorConfig, LidarSensorConfig
from holosoma.simulator.shared.lidar_range import finalize_lidar_returns
from holosoma.simulator.shared.sensor_manager import LidarRecord, SensorManager
from holosoma.utils.rotations import quat_apply, quat_mul

if TYPE_CHECKING:
    from holosoma.config_types.sensor import SensorMountConfig
    from holosoma.managers.terrain.terms.locomotion import TerrainLocomotion
    from holosoma.simulator.isaacgym.isaacgym import IsaacGym

# Change-of-basis quaternion (xyzw) from the holosoma camera frame (-Z fwd / +Y up) to
# IsaacGym's native camera frame (+X fwd / +Z up): the rotation sending +X->-Z, +Z->+Y, +Y->-X.
# MuJoCo/IsaacSim use the -Z fwd / +Y up frame directly.
_CANONICAL_TO_ISAACGYM_XYZW = (0.5, -0.5, -0.5, -0.5)


def build_env_cameras(sim: IsaacGym, env_id: int, env_ptr: Any, robot_handle: Any) -> None:
    """Create and body-attach one camera sensor per configured camera, for this env.

    Appends each created handle to ``sim.camera_handles[cam_name]`` (parallel to ``sim.envs``).
    """
    for cam_name, cam in sim.sensor_config.items():
        if not isinstance(cam, CameraSensorConfig):
            continue
        props = gymapi.CameraProperties()
        props.width = cam.width
        props.height = cam.height
        props.enable_tensors = True  # GPU image tensors (zero-copy read)
        props.near_plane = cam.near  # agnostic-core clipping, honored on every backend
        props.far_plane = cam.far
        # IsaacGym takes a horizontal fov; derive it from the vertical fov and aspect.
        aspect = cam.width / cam.height
        props.horizontal_fov = math.degrees(2 * math.atan(math.tan(math.radians(cam.vertical_fov) / 2) * aspect))
        ig = cam.isaacgym
        if ig.supersampling_horizontal is not None:
            props.supersampling_horizontal = ig.supersampling_horizontal
        if ig.supersampling_vertical is not None:
            props.supersampling_vertical = ig.supersampling_vertical
        if ig.use_collision_geometry is not None:
            props.use_collision_geometry = ig.use_collision_geometry

        cam_handle = sim.gym.create_camera_sensor(env_ptr, props)
        if cam_handle < 0:
            raise RuntimeError(f"IsaacGym failed to create camera sensor '{cam_name}' (graphics disabled?).")

        local_tf = _mount_to_isaacgym_transform(cam.mount)
        if cam.mount.target_kind == "world":
            # Free-floating: place at the env-local pose and leave it fixed (no body to follow).
            # Unlike attach_camera_to_body, set_camera_transform takes a WORLD transform (the
            # env_ptr only selects which env's camera, it does not offset the pose), so add this
            # env's origin — otherwise every env's world camera lands near the global origin and
            # only env 0 (origin [0,0,0]) frames its scene; envs 1..N look at empty space.
            origin = sim.env_origins[env_id]
            world_tf = gymapi.Transform(
                p=gymapi.Vec3(
                    local_tf.p.x + float(origin[0]),
                    local_tf.p.y + float(origin[1]),
                    local_tf.p.z + float(origin[2]),
                ),
                r=local_tf.r,
            )
            sim.gym.set_camera_transform(cam_handle, env_ptr, world_tf)
        else:
            body_handle = _resolve_mount_body_handle(sim, env_ptr, robot_handle, cam.mount)
            sim.gym.attach_camera_to_body(cam_handle, env_ptr, body_handle, local_tf, gymapi.FOLLOW_TRANSFORM)
        sim.camera_handles[cam_name].append(cam_handle)


def _resolve_mount_body_handle(sim: IsaacGym, env_ptr: Any, robot_handle: Any, mount: SensorMountConfig) -> int:
    """Return the rigid-body handle for a sensor mount within ``env_ptr``.

    ``robot_link`` -> a named robot link (name the root link to mount on the base);
    ``actor`` -> a scene actor's (single) body. Uses IsaacGym's per-actor body lookup.
    """
    if mount.target_kind == "robot_link":
        body_name = mount.target
        handle = sim.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, body_name)
        if handle < 0:
            raise ValueError(f"Camera robot mount body '{body_name}' not found in robot bodies {sim._body_list}.")
        return cast("int", handle)
    if mount.target_kind == "actor":
        actor_handles = sim.object_handles.get(mount.target)
        if not actor_handles:
            raise ValueError(f"Camera actor mount '{mount.target}' is not a registered scene object.")
        actor_handle = actor_handles[-1]  # the handle just built for this env
        body_name = sim.gym.get_actor_rigid_body_names(env_ptr, actor_handle)[0]
        return cast("int", sim.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name))
    raise ValueError(f"Unknown camera mount target_kind '{mount.target_kind}'.")


def _mount_to_isaacgym_transform(mount: SensorMountConfig) -> gymapi.Transform:
    """Mount (pos + w-first quat, -Z fwd/+Y up) to an IsaacGym local Transform.

    Composes the mount orientation with the change-of-basis so the camera looks where the
    mount intends on IsaacGym's native (+X fwd / +Z up) optical axis.
    """
    wxyz = torch.tensor([mount.orientation], dtype=torch.float32)  # [1,4] w,x,y,z
    xyzw = wxyz[:, [1, 2, 3, 0]]
    basis = torch.tensor([_CANONICAL_TO_ISAACGYM_XYZW], dtype=torch.float32)
    native_xyzw = quat_mul(xyzw, basis, w_last=True)[0]
    tf = gymapi.Transform()
    tf.p = gymapi.Vec3(*mount.position)
    tf.r = gymapi.Quat(float(native_xyzw[0]), float(native_xyzw[1]), float(native_xyzw[2]), float(native_xyzw[3]))
    return tf


def register_cameras(sim: IsaacGym, manager: SensorManager) -> None:
    """Register each configured camera."""
    for name, config in sim.sensor_config.items():
        if isinstance(config, CameraSensorConfig):
            manager.register_camera(name, config)


def render_cameras(sim: IsaacGym) -> None:
    """Render all due camera sensors once (per control step), honoring update_decimation."""
    if sim.sensor_manager is None:
        return
    due = sim.sensor_manager.collect_due()
    if not due:
        return
    # Update the graphics scene from the latest physics state, then render all camera sensors
    # in one pass. fetch_results + step_graphics are required before render or the image is
    # blank (same sequence the video recorder uses).
    sim.gym.fetch_results(sim.sim, True)
    sim.gym.step_graphics(sim.sim)
    sim.gym.render_all_camera_sensors(sim.sim)
    sim.gym.start_access_image_tensors(sim.sim)
    try:
        for camera_record in due:
            handles = sim.camera_handles[camera_record.name]  # per-env handles (parallel to sim.envs)
            if "rgb" in camera_record.config.data_types:
                frames = []
                for e in range(sim.num_envs):
                    # IsaacGym IMAGE_COLOR is RGBA uint8 [H,W,4] on GPU (channel 0 = R); drop
                    # alpha to get R,G,B.
                    gpu_t = sim.gym.get_camera_image_gpu_tensor(sim.sim, sim.envs[e], handles[e], gymapi.IMAGE_COLOR)
                    frames.append(gymtorch.wrap_tensor(gpu_t)[..., :3].clone())  # [H,W,4] RGBA -> RGB
                camera_record.set_buffer("rgb", torch.stack(frames, dim=0).to(sim.device))  # [N,H,W,3]
            if "depth" in camera_record.config.data_types:
                frames = []
                for e in range(sim.num_envs):
                    # IsaacGym IMAGE_DEPTH is [H,W] float32, negative distance along the camera
                    # axis (looks down -Z) with -inf for no-hit. Negate to get positive
                    # meters; -inf becomes the +inf no-hit sentinel.
                    gpu_t = sim.gym.get_camera_image_gpu_tensor(sim.sim, sim.envs[e], handles[e], gymapi.IMAGE_DEPTH)
                    frames.append((-gymtorch.wrap_tensor(gpu_t)).clone())  # [H,W] +meters, +inf no-hit
                camera_record.set_buffer("depth", torch.stack(frames, dim=0).unsqueeze(-1).to(sim.device))  # [N,H,W,1]
    finally:
        sim.gym.end_access_image_tensors(sim.sim)


def register_lidars(sim: IsaacGym, manager: SensorManager) -> None:
    """Register terrain-only scanners.

    Isaac Gym Preview 4 exposes no scene-query or LiDAR API. This compatibility path reuses the
    terrain Warp mesh already maintained by Holosoma and therefore cannot return robot/actor hits.
    """
    lidar_count = 0
    for name, config in sim.sensor_config.items():
        if not isinstance(config, LidarSensorConfig):
            continue
        lidar_count += 1
        lidar_record = manager.register_lidar(name, config)
        if config.mount.target_kind == "robot_link":
            body_index = sim.find_rigid_body_indice(config.mount.target)
            if body_index < 0:
                raise ValueError(
                    f"LiDAR '{name}' robot_link '{config.mount.target}' was not found in the Isaac Gym robot."
                )
            lidar_record.backend_cache["isaacgym_body_index"] = body_index
    if lidar_count:
        logger.warning(
            "Isaac Gym LiDAR uses Holosoma's static terrain mesh only; robot and scene-object "
            "geometry will not produce returns."
        )


def _mount_pose(sim: IsaacGym, lidar_record: LidarRecord) -> tuple[torch.Tensor, torch.Tensor]:
    mount = lidar_record.config.mount
    if mount.target_kind == "robot_link":
        body_index = lidar_record.backend_cache.get("isaacgym_body_index")
        if not isinstance(body_index, int):
            raise RuntimeError(f"LiDAR '{lidar_record.name}' has no resolved Isaac Gym mount-body index.")
        parent_pos = sim.rigid_body_pos_w[:, body_index]
        parent_quat = sim.rigid_body_quat_w[:, body_index]
    elif mount.target_kind == "actor":
        env_ids = torch.arange(sim.num_envs, device=sim.device)
        states = sim.get_actor_states([mount.target], env_ids)
        parent_pos, parent_quat = states[:, :3], states[:, 3:7]
    else:
        parent_pos = sim.env_origins
        parent_quat = torch.zeros((sim.num_envs, 4), dtype=torch.float32, device=sim.device)
        parent_quat[:, 3] = 1.0

    mount_pos = torch.tensor(mount.position, dtype=parent_pos.dtype, device=sim.device).expand(sim.num_envs, -1)
    mount_quat_wxyz = torch.tensor(mount.orientation, dtype=parent_quat.dtype, device=sim.device)
    mount_quat = mount_quat_wxyz[[1, 2, 3, 0]].expand(sim.num_envs, -1)
    sensor_pos = parent_pos + quat_apply(parent_quat, mount_pos, True)
    sensor_quat = quat_mul(parent_quat, mount_quat, True)
    return sensor_pos, sensor_quat


def render_lidars(sim: IsaacGym) -> None:
    manager = sim.sensor_manager
    if manager is None:
        return
    due = manager.collect_lidars_due()
    if not due:
        return
    terrain = sim.terrain_manager.get_state("locomotion_terrain")
    if not hasattr(terrain, "warp_mesh"):
        raise RuntimeError("Isaac Gym LiDAR requires a terrain with a Warp ray-casting mesh.")
    terrain_with_mesh = cast("TerrainLocomotion", terrain)
    # Importing Warp initializes its global runtime, which a camera-only simulator should never need.
    from holosoma.utils import warp_utils

    for lidar_record in due:
        pattern = lidar_record.pattern_at(sim.time())
        origin, orientation = _mount_pose(sim, lidar_record)
        ray_count = pattern.directions.shape[0]
        starts = origin.unsqueeze(1).expand(-1, ray_count, -1)
        local = pattern.directions.unsqueeze(0).expand(sim.num_envs, -1, -1)
        directions_world = quat_apply(
            orientation.unsqueeze(1).expand(-1, ray_count, -1),
            local,
            True,
        )
        hits = warp_utils.ray_cast(starts.contiguous(), directions_world.contiguous(), terrain_with_mesh.warp_mesh)
        distances = torch.linalg.vector_norm(hits - starts, dim=-1)
        finalize_lidar_returns(lidar_record, pattern.directions, distances)


def create_sensors(sim: IsaacGym) -> None:
    """Create one shared manager and register every configured sensor."""
    if not sim.sensor_config:
        return
    cfg = sim.simulator_config.sim
    manager = SensorManager(sim.device, cfg.fps / cfg.control_decimation_steps)
    sim.sensor_manager = manager
    register_cameras(sim, manager)
    register_lidars(sim, manager)


def render_sensors(sim: IsaacGym) -> None:
    """Capture all camera and LiDAR sensors due this control step."""
    render_cameras(sim)
    render_lidars(sim)
