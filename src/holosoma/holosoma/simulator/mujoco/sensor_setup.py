# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""MuJoCo mounted camera and LiDAR sensor setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
import numpy.typing as npt
import torch

from holosoma.config_types.sensor import DEFAULT_MUJOCO_LIDAR_GEOM_GROUPS, CameraSensorConfig, LidarSensorConfig
from holosoma.simulator.mujoco.backends import WarpBackend
from holosoma.simulator.mujoco.geom_groups import (
    ROBOT_COLLISION_GEOM_GROUP,
    ROBOT_VISUAL_GEOM_GROUP,
    map_authored_geom_group_mask,
)
from holosoma.simulator.mujoco.scene_manager import LIDAR_FILTER_SITE_PREFIX, LIDAR_MOUNT_SITE_PREFIX
from holosoma.simulator.shared.lidar_range import finalize_lidar_returns
from holosoma.simulator.shared.sensor_manager import LidarRecord, SensorManager
from holosoma.utils.warp_interop import from_torch_vec3

if TYPE_CHECKING:
    from holosoma.simulator.mujoco.mujoco import MuJoCo


@dataclass
class _WarpRayWorkspace:
    """Fixed-shape Torch/Warp buffers and replay graph for one LiDAR configuration."""

    torch_stream: torch.cuda.Stream
    origins: torch.Tensor
    directions: torch.Tensor
    distances: torch.Tensor
    geom_ids: torch.Tensor
    normals: torch.Tensor
    excludes: torch.Tensor
    geom_group: Any
    origins_wp: Any
    directions_wp: Any
    distances_wp: Any
    geom_ids_wp: Any
    normals_wp: Any
    excludes_wp: Any
    graph: Any | None = None


def register_cameras(sim: MuJoCo, manager: SensorManager) -> None:
    """Register cameras and create the backend's camera renderers.

    Widens MuJoCo's global near/far clip range to cover every configured camera because MuJoCo
    clipping is global rather than per-camera.
    """
    cameras = {name: config for name, config in sim.sensor_config.items() if isinstance(config, CameraSensorConfig)}
    if cameras:
        assert sim.root_model is not None
        extent = float(sim.root_model.stat.extent)
        if extent > 0:
            sim.root_model.vis.map.znear = min(camera.near for camera in cameras.values()) / extent
            sim.root_model.vis.map.zfar = max(camera.far for camera in cameras.values()) / extent

        for name, config in cameras.items():
            manager.register_camera(name, config)
        sim.backend.create_renderers(manager.cameras)


def offset_world_cameras(sim: MuJoCo) -> None:
    """Place each environment's world-mount cameras at its own origin for MJWarp."""
    if sim.sensor_manager is None or not hasattr(sim.backend, "offset_world_cameras"):
        return
    camera_ids = getattr(sim.backend, "_cam_ids", {})
    world_camera_ids = [
        camera_ids[name]
        for name, config in sim.sensor_config.items()
        if isinstance(config, CameraSensorConfig) and config.mount.target_kind == "world" and name in camera_ids
    ]
    if world_camera_ids:
        sim.backend.offset_world_cameras(world_camera_ids, sim.env_origins)


def render_cameras(sim: MuJoCo) -> None:
    """Render due cameras into their cached output buffers."""
    if sim.sensor_manager is None:
        return
    due = sim.sensor_manager.collect_due()
    if due:
        sim.backend.render_cameras(due)


def register_lidars(sim: MuJoCo, manager: SensorManager) -> None:
    """Register configured LiDARs and resolve their compiled query ids."""
    assert sim.root_model is not None
    for name, lidar in sim.sensor_config.items():
        if not isinstance(lidar, LidarSensorConfig):
            continue
        sensor_record = manager.register_lidar(name, lidar)
        site_name = f"{LIDAR_MOUNT_SITE_PREFIX}{name}"
        site_id = mujoco.mj_name2id(sim.root_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise RuntimeError(f"LiDAR mount site '{site_name}' is missing from the compiled MuJoCo model.")
        sensor_record.backend_cache["mujoco_site_id"] = site_id
        sensor_record.backend_cache["mujoco_body_exclude_id"] = _resolve_body_exclude(sim, name, lidar, site_id)


def _resolve_body_exclude(sim: MuJoCo, name: str, lidar: LidarSensorConfig, site_id: int) -> int:
    """Resolve the configured native one-body ray filter after model compilation."""
    assert sim.root_model is not None
    body_filter = lidar.body_filter
    if body_filter.target_kind == "mount":
        site_body = int(sim.root_model.site_bodyid[site_id])
        if site_body == 0 and lidar.mount.target_kind != "world":
            raise ValueError(
                f"LiDAR '{name}' mount target '{lidar.mount.target_kind}:{lidar.mount.target}' "
                "compiled into MuJoCo's world body and cannot be selectively filtered."
            )
        return site_body if site_body != 0 else -1
    if body_filter.target_kind in ("robot", "none"):
        return -1

    filter_site_name = f"{LIDAR_FILTER_SITE_PREFIX}{name}"
    filter_site_id = mujoco.mj_name2id(sim.root_model, mujoco.mjtObj.mjOBJ_SITE, filter_site_name)
    if filter_site_id < 0:
        raise RuntimeError(
            f"LiDAR '{name}' MuJoCo body-filter marker '{filter_site_name}' is missing from the compiled scene."
        )
    filter_body = int(sim.root_model.site_bodyid[filter_site_id])
    if filter_body == 0:
        raise ValueError(
            f"LiDAR '{name}' body filter target '{body_filter.target_kind}:{body_filter.target}' "
            "compiled into MuJoCo's world body and cannot be selectively filtered."
        )
    return filter_body


def _effective_geom_groups(lidar: LidarSensorConfig) -> tuple[bool, ...]:
    """Map authored geom-group intent onto the compiled robot's reserved layers."""
    configured = tuple(DEFAULT_MUJOCO_LIDAR_GEOM_GROUPS if lidar.mujoco is None else lidar.mujoco.geom_groups)
    effective = list(map_authored_geom_group_mask(configured))
    if lidar.body_filter.target_kind == "robot":
        # Explicit masks may enable groups 3-5, so whole-robot filtering clears both compiled layers.
        effective[ROBOT_VISUAL_GEOM_GROUP] = False
        effective[ROBOT_COLLISION_GEOM_GROUP] = False
    return tuple(effective)


def _cast_cpu(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    origin: npt.NDArray[np.float64],
    directions_world: npt.NDArray[np.float64],
    *,
    geom_groups: npt.NDArray[np.uint8],
    include_static: bool,
    body_exclude: int,
    far: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Cast one origin's rays using the MuJoCo 3.10 batched ray API."""
    ray_count = directions_world.shape[0]
    distances = np.empty(ray_count, dtype=np.float64)
    geom_ids = np.empty(ray_count, dtype=np.int32)
    mujoco.mj_multiRay(
        model,
        data,
        np.ascontiguousarray(origin, dtype=np.float64),
        np.ascontiguousarray(directions_world.reshape(-1), dtype=np.float64),
        geom_groups,
        include_static,
        body_exclude,
        geom_ids,
        distances,
        None,
        ray_count,
        far,
    )
    return distances, geom_ids


def _warp_geom_groups(sensor_record: LidarRecord, wp: Any) -> Any:
    """Build the public six-component Warp vector required by ``mujoco_warp.rays``."""
    groups = _effective_geom_groups(sensor_record.config)
    if len(groups) != 6:
        raise RuntimeError(f"LiDAR '{sensor_record.name}' needs exactly six MuJoCo geom-group flags, got {groups}.")
    vec6f = wp.types.vector(length=6, dtype=wp.float32)
    return vec6f(*[float(enabled) for enabled in groups])


def _warp_workspace(
    sim: MuJoCo,
    sensor_record: LidarRecord,
    backend: WarpBackend,
    *,
    ray_count: int,
) -> _WarpRayWorkspace:
    """Allocate and retain one fixed-size Warp query workspace for a LiDAR."""
    import warp as wp

    cache_key = "mujoco_warp_rays"
    cached = sensor_record.backend_cache.get(cache_key)
    if cached is not None:
        if not isinstance(cached, _WarpRayWorkspace):
            raise RuntimeError(f"LiDAR '{sensor_record.name}' has an invalid MuJoCo Warp workspace.")
        workspace: _WarpRayWorkspace = cached
        if workspace.directions.shape != (sim.num_envs, ray_count, 3):
            raise RuntimeError(f"LiDAR '{sensor_record.name}' ray shape changed after registration.")
        return workspace

    # Keep Torch buffer writes and the captured Warp ray graph on the same CUDA stream.
    torch_stream = torch.cuda.ExternalStream(wp.get_stream(backend.mjw_device).cuda_stream)
    with torch.cuda.stream(torch_stream), wp.ScopedDevice(backend.mjw_device):
        workspace = _WarpRayWorkspace(
            torch_stream=torch_stream,
            origins=torch.empty((sim.num_envs, ray_count, 3), dtype=torch.float32, device=sim.sim_device),
            directions=torch.empty((sim.num_envs, ray_count, 3), dtype=torch.float32, device=sim.sim_device),
            distances=torch.empty((sim.num_envs, ray_count), dtype=torch.float32, device=sim.sim_device),
            geom_ids=torch.empty((sim.num_envs, ray_count), dtype=torch.int32, device=sim.sim_device),
            normals=torch.empty((sim.num_envs, ray_count, 3), dtype=torch.float32, device=sim.sim_device),
            excludes=torch.full((ray_count,), -1, dtype=torch.int32, device=sim.sim_device),
            geom_group=_warp_geom_groups(sensor_record, wp),
            origins_wp=None,
            directions_wp=None,
            distances_wp=None,
            geom_ids_wp=None,
            normals_wp=None,
            excludes_wp=None,
        )
        body_exclude = sensor_record.backend_cache.get("mujoco_body_exclude_id")
        if not isinstance(body_exclude, int):
            raise RuntimeError(f"LiDAR '{sensor_record.name}' has no resolved MuJoCo body exclusion.")
        workspace.excludes.fill_(body_exclude)
        workspace.origins_wp = from_torch_vec3(workspace.origins, backend.mjw_device)
        workspace.directions_wp = from_torch_vec3(workspace.directions, backend.mjw_device)
        workspace.distances_wp = wp.from_torch(workspace.distances)
        workspace.geom_ids_wp = wp.from_torch(workspace.geom_ids)
        workspace.normals_wp = from_torch_vec3(workspace.normals, backend.mjw_device)
        workspace.excludes_wp = wp.from_torch(workspace.excludes)
    sensor_record.backend_cache[cache_key] = workspace
    return workspace


def _render_cpu(sim: MuJoCo, sensor_record: LidarRecord) -> None:
    assert sim.root_model is not None
    pattern = sensor_record.pattern_at(sim.time())
    directions_local = pattern.directions.to("cpu")
    site_id = sensor_record.backend_cache["mujoco_site_id"]
    if not isinstance(site_id, int):
        raise RuntimeError(f"LiDAR '{sensor_record.name}' has no resolved MuJoCo mount-site id.")
    geom_groups = np.asarray(_effective_geom_groups(sensor_record.config), dtype=np.uint8)
    include_static = True
    body_exclude = sensor_record.backend_cache.get("mujoco_body_exclude_id")
    if not isinstance(body_exclude, int):
        raise RuntimeError(f"LiDAR '{sensor_record.name}' has no resolved MuJoCo body exclusion.")
    distances_by_env = []
    geom_ids_by_env = []
    local_np = directions_local.numpy().astype(np.float64, copy=False)
    for env_id in range(sim.num_envs):
        data = sim.backend.get_render_data(world_id=env_id)
        origin = data.site_xpos[site_id].copy()
        if sensor_record.config.mount.target_kind == "world":
            origin += sim.env_origins[env_id].detach().cpu().numpy()
        rotation = data.site_xmat[site_id].reshape(3, 3)
        directions_world = local_np @ rotation.T
        distances, geom_ids = _cast_cpu(
            sim.root_model,
            data,
            origin,
            directions_world,
            geom_groups=geom_groups,
            include_static=include_static,
            body_exclude=body_exclude,
            far=sensor_record.config.far,
        )
        distances_by_env.append(torch.from_numpy(distances))
        geom_ids_by_env.append(torch.from_numpy(geom_ids))
    finalize_lidar_returns(
        sensor_record,
        pattern.directions,
        torch.stack(distances_by_env).to(sim.sim_device),
        geom_ids=torch.stack(geom_ids_by_env).to(sim.sim_device),
    )


def _render_warp(
    sim: MuJoCo,
    sensor_record: LidarRecord,
    backend: WarpBackend,
    ray_context: Any,
    sim_time: float,
) -> None:
    import mujoco_warp as mjw
    import warp as wp

    assert sim.root_model is not None
    pattern = sensor_record.pattern_at(sim_time)
    directions_local = pattern.directions
    ray_count = directions_local.shape[0]
    site_id = sensor_record.backend_cache["mujoco_site_id"]
    if not isinstance(site_id, int):
        raise RuntimeError(f"LiDAR '{sensor_record.name}' has no resolved MuJoCo mount-site id.")
    workspace = _warp_workspace(sim, sensor_record, backend, ray_count=ray_count)
    consumer_stream = torch.cuda.current_stream(sim.sim_device)
    workspace.torch_stream.wait_stream(consumer_stream)
    with torch.cuda.stream(workspace.torch_stream), wp.ScopedDevice(backend.mjw_device):
        site_xpos = wp.to_torch(backend.mjw_data.site_xpos)[:, site_id]
        site_xmat = wp.to_torch(backend.mjw_data.site_xmat)[:, site_id]
        workspace.origins.copy_(site_xpos.unsqueeze(1).expand(-1, ray_count, -1))
        if sensor_record.config.mount.target_kind == "world":
            workspace.origins += sim.env_origins.to(workspace.origins.device, workspace.origins.dtype).unsqueeze(1)
        workspace.directions.copy_(torch.einsum("nij,rj->nri", site_xmat, directions_local))
        if workspace.graph is None:
            with wp.ScopedCapture() as capture:
                mjw.rays(
                    backend.mjw_model,
                    backend.mjw_data,
                    workspace.origins_wp,
                    workspace.directions_wp,
                    workspace.geom_group,
                    True,
                    workspace.excludes_wp,
                    workspace.distances_wp,
                    workspace.geom_ids_wp,
                    workspace.normals_wp,
                    rc=ray_context,
                )
            workspace.graph = capture.graph
        wp.capture_launch(workspace.graph)
    consumer_stream.wait_stream(workspace.torch_stream)
    finalize_lidar_returns(sensor_record, directions_local, workspace.distances, geom_ids=workspace.geom_ids)


def render_lidars(sim: MuJoCo) -> None:
    if sim.sensor_manager is None:
        return
    due = sim.sensor_manager.collect_lidars_due()
    if isinstance(sim.backend, WarpBackend):
        if not due:
            return
        enabled_geom_groups = sorted(
            {
                group
                for sensor_record in sim.sensor_manager.lidars
                for group, enabled in enumerate(_effective_geom_groups(sensor_record.config))
                if enabled
            }
        )
        ray_context = sim.backend.prepare_lidar_ray_context(enabled_geom_groups or [0, 1, 2])
        sim_time = sim.time()
        for sensor_record in due:
            _render_warp(sim, sensor_record, sim.backend, ray_context, sim_time)
        return
    for sensor_record in due:
        _render_cpu(sim, sensor_record)


def create_sensors(sim: MuJoCo) -> None:
    """Create one shared manager and register every configured sensor."""
    if not sim.sensor_config:
        return
    cfg = sim.simulator_config.sim
    manager = SensorManager(sim.sim_device, cfg.fps / cfg.control_decimation_steps)
    sim.sensor_manager = manager
    register_cameras(sim, manager)
    register_lidars(sim, manager)


def render_sensors(sim: MuJoCo) -> None:
    """Capture all camera and LiDAR sensors due this control step."""
    if isinstance(sim.backend, WarpBackend):
        sim.backend.begin_sensor_render()
    render_cameras(sim)
    render_lidars(sim)
