"""Isaac Sim mounted camera and LiDAR sensor setup.

Each configured camera is a ``TiledCamera`` created before ``clone_environments`` as a child prim
of its mount body, so the USD transform hierarchy makes it follow that body natively across all
cloned envs. These module-level functions take the simulator (the ``fields.py`` / MuJoCo
shared sensor setup convention) and drive that pipeline — prim-path resolution, the FOV->aperture
pinhole cfg, per-env creation, ``SensorManager`` registration, and the per-control-step render
read-out. The per-camera ``TiledCamera`` map lives on the simulator as ``sim.tiled_cameras``,
written before clone and read at render.

Native sensors are built before cloning, then registered with the shared sensor manager after
the scene is built.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, cast

import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg, patterns
from isaaclab.utils.math import quat_apply

from holosoma.config_types.frequency import resolve_decimation
from holosoma.config_types.sensor import CameraSensorConfig, IsaacSimLidarConfig, LidarSensorConfig
from holosoma.simulator.isaacsim.camera_projection import camera_spawn_cfg_for
from holosoma.simulator.isaacsim.lidar_target_discovery import (
    _is_path_at_or_below,
    assemble_default_mesh_prim_paths,
)
from holosoma.simulator.isaacsim.prim_utils import (
    get_current_stage,
    is_rigid_body_enabled,
    resolve_robot_link_prim_expression,
    resolve_robot_link_rigid_body_prim_expression,
    set_instanceable,
)
from holosoma.simulator.shared.lidar_range import finalize_lidar_returns
from holosoma.simulator.shared.sensor_manager import SensorManager

if TYPE_CHECKING:
    from holosoma.config_types.sensor import LidarRayPatternConfig, SensorMountConfig
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim


def build_cameras(sim: IsaacSim) -> None:
    """Build a TiledCamera per configured camera (before clone, so each replicates per env).

    The camera optical convention is OpenGL (-Z forward, +Y up), the same frame holosoma uses,
    so the mount offset passes through with ``convention="opengl"`` and no extra rotation.
    Mount quat is the config-layer ``[w,x,y,z]`` (IsaacLab OffsetCfg.rot is also w-first). Populates
    ``sim.tiled_cameras`` and registers each camera prim into ``sim.scene.sensors``.
    """
    sim.tiled_cameras = {}
    for cam_name, cam in sim.sensor_config.items():
        if not isinstance(cam, CameraSensorConfig):
            continue
        parent = _camera_mount_prim_path(cam.mount)
        # Map holosoma data_types -> IsaacLab annotators.
        annotators = [{"rgb": "rgb", "depth": "distance_to_image_plane"}[d] for d in cam.data_types]
        cam_cfg = TiledCameraCfg(
            prim_path=f"{parent}/{cam_name}",
            offset=TiledCameraCfg.OffsetCfg(
                pos=tuple(cam.mount.position),
                rot=tuple(cam.mount.orientation),  # (w, x, y, z)
                convention="opengl",  # -Z forward / +Y up, the frame holosoma uses
            ),
            data_types=annotators,
            spawn=camera_spawn_cfg_for(cam, sim_utils),
            width=cam.width,
            height=cam.height,
            # Keep raw +inf no-return data; CameraRecord applies Holosoma's common depth policy.
            depth_clipping_behavior="none",
        )
        sim.tiled_cameras[cam_name] = TiledCamera(cam_cfg)
        sim.scene.sensors[cam_name] = sim.tiled_cameras[cam_name]


def _camera_mount_prim_path(mount: SensorMountConfig) -> str:
    """Resolve a sensor mount to a per-env camera prim path (parent = the mount body prim).

    ``robot_link`` -> a robot link prim (name the root link to mount on the base);
    ``actor`` -> a spawned scene-object prim. The camera prim is created as a child of this
    path so the USD transform hierarchy makes it follow the body natively.
    """
    ns = "/World/envs/env_.*"
    if mount.target_kind == "robot_link":
        return resolve_robot_link_prim_expression(mount.target)
    if mount.target_kind == "actor":
        return f"{ns}/{mount.target}"
    if mount.target_kind == "world":
        # Free-floating: child of the env prim itself, so the mount offset is the pose in the
        # per-env frame (each cloned env carries its own copy at the same relative pose).
        return ns
    raise ValueError(f"Unknown camera mount target_kind '{mount.target_kind}'.")


def register_cameras(sim: IsaacSim, manager: SensorManager) -> None:
    """Register TiledCameras built before environment cloning."""
    for name, config in sim.sensor_config.items():
        if isinstance(config, CameraSensorConfig):
            manager.register_camera(name, config)


def render_cameras(sim: IsaacSim) -> None:
    """Cache each due TiledCamera's RGB/depth as an env-first frame into its buffer.

    TiledCameras are RTX sensors the sim already updates in its own render pass
    (``scene.update`` in ``simulate_at_each_physics_step``); this reads that output, drops
    alpha, and writes a ``[N,H,W,3]`` uint8 (rgb) / ``[N,H,W,1]`` float32 (depth) frame once
    per control step (honoring ``update_decimation``), mirroring the other backends' render ->
    buffer -> read flow."""
    if sim.sensor_manager is None:
        return
    for camera_record in sim.sensor_manager.collect_due():
        out = sim.tiled_cameras[camera_record.name].data.output
        if "rgb" in camera_record.config.data_types:
            rgb = out["rgb"][..., :3]  # [N,H,W,3], drop alpha
            if rgb.dtype != torch.uint8:
                rgb = rgb.clamp(0, 255).to(torch.uint8)
            camera_record.set_buffer("rgb", rgb)
        if "depth" in camera_record.config.data_types:
            # distance_to_image_plane is float32 meters, image-plane. Ensure a trailing channel
            # dim; CameraRecord applies the configured cross-backend no-return convention.
            depth = out["distance_to_image_plane"].to(torch.float32)
            camera_record.set_buffer("depth", depth if depth.ndim == 4 else depth.unsqueeze(-1))


def _pattern(
    _native_config: Any,
    device: str,
    *,
    pattern_config: LidarRayPatternConfig,
    publish_hz: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    provider = pattern_config.get_provider_cls()(pattern_config, device=device, publish_hz=publish_hz)
    directions = provider.pattern_at(0.0).directions
    return torch.zeros_like(directions), directions


@lru_cache(maxsize=1)
def _mutable_multi_mesh_ray_caster_cls() -> type[Any]:
    """Create the IsaacLab 2.3.2 LiDAR type only when a LiDAR is configured."""
    from isaaclab.sensors import MultiMeshRayCaster

    class MutableMultiMeshRayCaster(MultiMeshRayCaster):
        """Multi-mesh ray caster with fixed-shape direction updates."""

        @property
        def ray_starts_w(self) -> torch.Tensor:
            """Return IsaacLab's current world-frame ray origins."""
            return self._ray_starts_w

        def set_ray_directions(self, directions: torch.Tensor) -> None:
            expected = (self.num_rays, 3)
            if directions.shape != expected:
                raise ValueError(f"RayCaster directions must keep shape {expected}, got {directions.shape}.")
            local = directions.to(device=self.ray_directions.device, dtype=self.ray_directions.dtype)
            offset_quat = torch.tensor(
                self.cfg.offset.rot,
                device=self.ray_directions.device,
                dtype=self.ray_directions.dtype,
            ).expand(self.num_rays, -1)
            local = quat_apply(offset_quat, local)
            self.ray_directions.copy_(local.unsqueeze(0).expand_as(self.ray_directions))
            self.mark_outdated()

        def mark_outdated(self) -> None:
            """Force the next data access to cast at the current final-step pose."""
            self._is_outdated[:] = True

        def _obtain_trackable_prim_view(self, target_prim_path: str) -> Any:
            """Track the nearest enabled physics owner, skipping disabled rigid-body APIs."""
            from isaaclab.sim.views import XformPrimView
            from pxr import UsdPhysics

            mesh_prim = sim_utils.find_first_matching_prim(target_prim_path)
            current_prim = mesh_prim
            current_path_expr = target_prim_path
            prim_view = None

            while prim_view is None:
                if current_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim_view = self._physics_sim_view.create_articulation_view(current_path_expr.replace(".*", "*"))
                    break
                if is_rigid_body_enabled(current_prim):
                    prim_view = self._physics_sim_view.create_rigid_body_view(current_path_expr.replace(".*", "*"))
                    break

                current_prim = current_prim.GetParent()
                current_path_expr = current_path_expr.rsplit("/", 1)[0]
                if not current_prim.IsValid():
                    prim_view = XformPrimView(target_prim_path, device=self._device, stage=self.stage)
                    current_path_expr = target_prim_path

            mesh_prims = sim_utils.find_matching_prims(target_prim_path)
            view_prims = sim_utils.find_matching_prims(current_path_expr)
            if len(mesh_prims) != len(view_prims):
                raise RuntimeError(
                    f"LiDAR target '{target_prim_path}' resolves to {len(mesh_prims)} meshes but "
                    f"{len(view_prims)} trackable owners at '{current_path_expr}'."
                )

            positions = []
            orientations = []
            for target_prim, view_prim in zip(mesh_prims, view_prims):
                position, orientation = sim_utils.resolve_prim_pose(target_prim, view_prim)
                positions.append(torch.tensor(position, dtype=torch.float32, device=self.device))
                orientations.append(torch.tensor(orientation, dtype=torch.float32, device=self.device))
            return prim_view, (torch.stack(positions), torch.stack(orientations))

        def _update_buffers_impl(self, env_ids: Any) -> None:
            """Update rays with correct rigid-body-to-mesh transform composition."""
            from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view
            from isaaclab.utils.math import quat_apply, quat_mul
            from isaaclab.utils.warp import raycast_dynamic_meshes

            self._update_ray_infos(env_ids)
            mesh_idx = 0
            for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
                count = self._num_meshes_per_env[target_cfg.prim_expr]
                if not target_cfg.track_mesh_transforms:
                    mesh_idx += count
                    continue

                pos_w, ori_w = obtain_world_pose_from_view(view, None)
                pos_w = pos_w.squeeze(0) if pos_w.ndim == 3 else pos_w
                ori_w = ori_w.squeeze(0) if ori_w.ndim == 3 else ori_w
                if target_cfg.prim_expr in self.mesh_offsets:
                    pos_offset, ori_offset = self.mesh_offsets[target_cfg.prim_expr]
                    pos_w = pos_w + quat_apply(ori_w, pos_offset)
                    ori_w = quat_mul(ori_w, ori_offset)

                view_count = view.count
                if view_count != 1:
                    view_count //= self._num_envs
                    pos_w = pos_w.view(self._num_envs, view_count, 3)
                    ori_w = ori_w.view(self._num_envs, view_count, 4)
                self._mesh_positions_w[:, mesh_idx : mesh_idx + view_count] = pos_w
                self._mesh_orientations_w[:, mesh_idx : mesh_idx + view_count] = ori_w
                mesh_idx += view_count

            self._data.ray_hits_w[env_ids], _, _, _, mesh_ids = raycast_dynamic_meshes(
                self._ray_starts_w[env_ids],
                self._ray_directions_w[env_ids],
                mesh_ids_wp=self._mesh_ids_wp,
                max_dist=self.cfg.max_distance,
                mesh_positions_w=self._mesh_positions_w[env_ids],
                mesh_orientations_w=self._mesh_orientations_w[env_ids],
                return_mesh_id=self.cfg.update_mesh_ids,
            )
            if self.cfg.update_mesh_ids:
                self._data.ray_mesh_ids[env_ids] = mesh_ids

    return MutableMultiMeshRayCaster


def _mount_parent_path(mount: SensorMountConfig) -> str:
    namespace = "/World/envs/env_.*"
    if mount.target_kind == "robot_link":
        return resolve_robot_link_prim_expression(mount.target)
    if mount.target_kind == "actor":
        return f"{namespace}/{mount.target}"
    if mount.target_kind == "world":
        return namespace
    raise ValueError(f"Unknown LiDAR mount target_kind '{mount.target_kind}'.")


def _is_visual_raycast_prim(prim: Any) -> bool:
    """Select geometry that represents an optical surface for default LiDAR discovery.

    USD has no universal "LiDAR surface" schema. Holosoma therefore follows USD imaging
    semantics first: inherited-invisible geometry and ``proxy``/``guide`` purpose are omitted,
    while supported ``default`` and ``render`` purpose geometry is eligible. Material opacity is
    not inspected because USD materials can encode it through arbitrary shader graphs; transparent
    helper geometry must instead be hidden, assigned non-render purpose, or omitted from the asset.
    """
    from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES
    from pxr import UsdGeom

    if prim.GetTypeName() not in set(PRIMITIVE_MESH_TYPES) | {"Mesh"}:
        return False
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return False
    if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
        return False

    return imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)


def _visual_raycast_prims(prim_path: str) -> list[Any]:
    """Return supported, renderable geometry below ``prim_path``."""
    try:
        return cast(
            list[Any],
            sim_utils.get_all_matching_child_prims(
                prim_path,
                _is_visual_raycast_prim,
            ),
        )
    except ValueError:
        return []


def _visual_mesh_prim_paths(
    source_root: str,
    expression_root: str,
    *,
    excluded_body_path: str | None = None,
) -> list[str]:
    """Return non-overlapping visual-geometry expressions below one source root."""
    mesh_prims = _visual_raycast_prims(source_root)
    mesh_paths = []
    for mesh_prim in mesh_prims:
        if excluded_body_path is not None:
            owner = mesh_prim
            while owner.IsValid() and _is_path_at_or_below(str(owner.GetPath()), source_root):
                if is_rigid_body_enabled(owner):
                    break
                owner = owner.GetParent()
            if owner.IsValid() and str(owner.GetPath()) == excluded_body_path:
                continue
        mesh_path = str(mesh_prim.GetPath())
        mesh_paths.append(f"{expression_root}{mesh_path[len(source_root) :]}")
    return sorted(mesh_paths)


def _body_filter_target(lidar: LidarSensorConfig) -> tuple[str, str] | None:
    body_filter = lidar.body_filter
    if body_filter.target_kind == "none":
        return None
    if body_filter.target_kind == "mount":
        mount = lidar.mount
        return None if mount.target_kind == "world" else (mount.target_kind, mount.target)
    return body_filter.target_kind, body_filter.target


def _robot_mesh_prim_paths(excluded_link: str | None) -> list[str]:
    """Return robot visual-geometry expressions, omitting one native body owner."""
    source_root = "/World/envs/env_0/Robot"
    expression_root = "/World/envs/env_.*/Robot"
    excluded_body_path = (
        None
        if excluded_link is None
        else resolve_robot_link_rigid_body_prim_expression(excluded_link).replace("env_.*", "env_0")
    )
    return _visual_mesh_prim_paths(
        source_root,
        expression_root,
        excluded_body_path=excluded_body_path,
    )


def _default_mesh_prim_paths(sim: IsaacSim, lidar: LidarSensorConfig) -> list[str]:
    excluded_target = _body_filter_target(lidar)
    if excluded_target is not None and excluded_target[0] == "actor":
        actor_name = excluded_target[1]
        if actor_name not in sim.scene.rigid_objects:
            raise ValueError(
                f"LiDAR body filter target 'actor:{actor_name}' is not registered. Known actors: "
                f"{list(sim.scene.rigid_objects)}."
            )
    return assemble_default_mesh_prim_paths(
        excluded_target=excluded_target,
        robot_mesh_prim_paths=_robot_mesh_prim_paths,
        # InteractiveScene.rigid_objects includes standalone objects and every static or dynamic
        # body expanded from scene files, so this covers all Holosoma scene-object meshes.
        actor_names=sim.scene.rigid_objects,
        discover_mesh_prim_paths=_visual_mesh_prim_paths,
    )


def build_lidars(sim: IsaacSim) -> None:
    """Create one mutable MultiMeshRayCaster per configured LiDAR before environment cloning."""
    control_hz = sim.simulator_config.sim.fps / sim.simulator_config.sim.control_decimation_steps
    lidars = [(name, config) for name, config in sim.sensor_config.items() if isinstance(config, LidarSensorConfig)]
    if not lidars:
        return

    for name, lidar in lidars:
        from isaaclab.sensors import MultiMeshRayCasterCfg

        parent = _mount_parent_path(lidar.mount)
        if lidar.mount.target_kind == "world":
            mount_path = f"{parent}/holosoma_lidar_{name}"
            source_path = mount_path.replace("env_.*", "env_0")
            sim_utils.create_prim(
                source_path,
                prim_type="Xform",
                translation=tuple(lidar.mount.position),
                orientation=tuple(lidar.mount.orientation),
            )
            offset = MultiMeshRayCasterCfg.OffsetCfg()
        else:
            mount_path = parent
            offset = MultiMeshRayCasterCfg.OffsetCfg(
                pos=tuple(lidar.mount.position),
                rot=tuple(lidar.mount.orientation),
            )
        update_decimation = resolve_decimation(
            lidar.update_decimation,
            control_hz,
            field=f"LiDAR '{name}' update_decimation",
        )
        publish_hz = control_hz / update_decimation
        native_pattern = patterns.PatternBaseCfg(
            func=partial(
                _pattern,
                pattern_config=lidar.pattern,
                publish_hz=publish_hz,
            )
        )
        isaacsim_config = lidar.isaacsim or IsaacSimLidarConfig()
        if isaacsim_config.mesh_prim_paths is None:
            mesh_paths = _default_mesh_prim_paths(sim, lidar)
        else:
            mesh_paths = isaacsim_config.mesh_prim_paths
        mesh_targets = [
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr=path,
                # Discovery returns exact ground descendants; the whole ground subtree is static.
                track_mesh_transforms=not (path == "/World/ground" or path.startswith("/World/ground/")),
            )
            for path in mesh_paths
        ]
        cfg = MultiMeshRayCasterCfg(
            prim_path=mount_path,
            offset=offset,
            update_period=0.0,
            ray_alignment="base",
            pattern_cfg=native_pattern,
            debug_vis=sim.debug_viz_enabled,
            mesh_prim_paths=mesh_targets,
            max_distance=lidar.far,
        )
        sensor = _mutable_multi_mesh_ray_caster_cls()(cfg)
        sim.scene.sensors[name] = sensor


def register_lidars(sim: IsaacSim, manager: SensorManager) -> None:
    for name, config in sim.sensor_config.items():
        if isinstance(config, LidarSensorConfig):
            manager.register_lidar(name, config)


def render_lidars(sim: IsaacSim) -> None:
    if sim.sensor_manager is None:
        return
    for lidar_record in sim.sensor_manager.collect_lidars_due():
        pattern = lidar_record.pattern_at(sim.time())
        sensor = sim.scene.sensors.get(lidar_record.name)
        if not isinstance(sensor, _mutable_multi_mesh_ray_caster_cls()):
            raise TypeError(f"LiDAR '{lidar_record.name}' has no mutable MultiMeshRayCaster.")
        sensor.set_ray_directions(pattern.directions)
        native = sensor.data
        distances = torch.linalg.vector_norm(native.ray_hits_w - sensor.ray_starts_w, dim=-1)
        finalize_lidar_returns(lidar_record, pattern.directions, distances)


def build_sensors(sim: IsaacSim) -> None:
    """Build native sensors before environment cloning."""
    needs_expanded_robot = any(
        isinstance(config, LidarSensorConfig) or config.mount.target_kind == "robot_link"
        for config in sim.sensor_config.values()
    )
    if needs_expanded_robot:
        # IsaacLab cannot author children or resolve exact targets through instance proxies.
        set_instanceable(get_current_stage(), "/World/envs/env_0/Robot", False)
    build_cameras(sim)
    build_lidars(sim)


def create_sensors(sim: IsaacSim) -> None:
    """Create one shared manager and register every configured sensor."""
    if not sim.sensor_config:
        return
    cfg = sim.simulator_config.sim
    manager = SensorManager(sim.sim_device, cfg.fps / cfg.control_decimation_steps)
    sim.sensor_manager = manager
    register_cameras(sim, manager)
    register_lidars(sim, manager)


def render_sensors(sim: IsaacSim) -> None:
    """Capture all camera and LiDAR sensors due this control step."""
    render_cameras(sim)
    render_lidars(sim)
