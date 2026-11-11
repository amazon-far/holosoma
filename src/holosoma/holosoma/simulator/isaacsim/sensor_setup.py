"""Isaac Sim mounted camera and LiDAR sensor setup.

The selected backend creates a Lab 2 ``TiledCamera`` or Lab 3 ``Camera`` before
environment cloning. Camera prims follow their mount bodies through the USD
transform hierarchy. ``sim.tiled_cameras`` retains the native camera map for both
versions; the shared sensor manager owns cadence, held observations and depth
clipping. Native LiDAR sensors follow the same build-before-clone lifecycle.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import patterns

from holosoma.config_types.frequency import resolve_decimation
from holosoma.config_types.sensor import CameraSensorConfig, IsaacSimLidarConfig, LidarSensorConfig
from holosoma.simulator.isaacsim.camera_projection import camera_spawn_cfg_for
from holosoma.simulator.isaacsim.lidar_target_discovery import (
    _is_path_at_or_below,
    assemble_default_mesh_prim_paths,
)
from holosoma.simulator.isaacsim.prim_utils import (
    _DISABLED_RIGID_BODY_LINK_KEY,
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
    from holosoma.simulator.shared.sensor_manager import CameraRecord
    from holosoma.simulator.isaacsim.backends.base import IsaacSimBackendAdapter


def build_cameras(sim: IsaacSim) -> None:
    """Build each backend's native camera before cloning it across environments.

    The camera optical convention is OpenGL (-Z forward, +Y up), the same frame holosoma uses,
    so the mount offset passes through with ``convention="opengl"`` and no extra rotation.
    Mount quat is the config-layer ``[w,x,y,z]``; the backend converts it to the native order. Populates
    ``sim.tiled_cameras`` and registers each camera prim into ``sim.scene.sensors``.
    """
    sim.tiled_cameras = {}
    backend = sim.backend
    for cam_name, cam in sim.sensor_config.items():
        if not isinstance(cam, CameraSensorConfig):
            continue
        parent = _camera_mount_prim_path(cam.mount, backend=backend)
        # Map holosoma data_types -> IsaacLab annotators.
        annotators = [{"rgb": "rgb", "depth": "distance_to_image_plane"}[d] for d in cam.data_types]
        cam_cfg = backend.create_camera_cfg(
            prim_path=f"{parent}/{cam_name}",
            offset_position=tuple(cam.mount.position),
            offset_orientation=tuple(cam.mount.orientation),
            data_types=annotators,
            spawn=camera_spawn_cfg_for(cam, sim_utils),
            width=cam.width,
            height=cam.height,
            # Keep raw +inf no-return data; CameraRecord applies Holosoma's common depth policy.
            depth_clipping_behavior="none",
        )
        camera = backend.create_camera(cam_cfg)
        sim.tiled_cameras[cam_name] = camera
        sim.scene.sensors[cam_name] = camera


def _camera_mount_prim_path(mount: SensorMountConfig, *, backend: IsaacSimBackendAdapter) -> str:
    """Resolve a sensor mount to a per-env camera prim path (parent = the mount body prim).

    ``robot_link`` -> a robot link prim (name the root link to mount on the base);
    ``actor`` -> a spawned scene-object prim. The camera prim is created as a child of this
    path so the USD transform hierarchy makes it follow the body natively.
    """
    ns = backend.env_regex_ns
    if mount.target_kind == "robot_link":
        return resolve_robot_link_prim_expression(mount.target, backend=backend)
    if mount.target_kind == "actor":
        return f"{ns}/{mount.target}"
    if mount.target_kind == "world":
        # Free-floating: child of the env prim itself, so the mount offset is the pose in the
        # per-env frame (each cloned env carries its own copy at the same relative pose).
        return ns
    raise ValueError(f"Unknown camera mount target_kind '{mount.target_kind}'.")


def register_cameras(sim: IsaacSim, manager: SensorManager) -> None:
    """Register the cameras built before environment cloning with the shared manager."""
    for name, config in sim.sensor_config.items():
        if isinstance(config, CameraSensorConfig):
            manager.register_camera(name, config)


def sync_camera_fabric_transforms(sim: IsaacSim) -> None:
    """Repair Lab 2 camera Fabric transforms to match their authored mount offsets.

    Call once after ``sim.reset()``. The renderer places cameras from Fabric. Isaac Lab initializes
    each TiledCamera during the reset; its first pose read syncs the camera into Fabric through an
    app update that also steps physics. The camera's world pose comes from USD, where the mount body
    has not moved, so a camera synced after its mount body moved gets a local offset derived against
    the moved body, and renders off its mount for the whole run by an amount that depends on the
    physics dt. This was measured on one of four cameras (the last created; 4-8 cm and 5-8 deg on a
    wrist camera); which cameras are hit is not explained. The USD offset stays correct, so it is
    copied back into Fabric for every camera.
    """
    # Lab 3's native FrameView initializes transforms for every backend.
    # The Kit-less path must not import Fabric's Kit runtime.
    if sim.backend.is_isaacsim6:
        return
    import carb
    import usdrt
    from pxr import Usd, UsdGeom
    from usdrt import Rt

    # Isaac Lab's views keep camera poses in USD without Fabric or on a CPU device; the USD offset is
    # then what renders, and there is no Fabric transform to correct.
    if (
        not sim.tiled_cameras
        or sim.sim_device == "cpu"
        or not carb.settings.get_settings().get("/physics/fabricEnabled")
    ):
        return
    fabric_stage = usdrt.Usd.Stage.Attach(sim_utils.get_current_stage_id())
    for camera in sim.tiled_cameras.values():
        for prim in sim_utils.find_matching_prims(camera.cfg.prim_path):
            local = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
            fabric_prim = Rt.Xformable(fabric_stage.GetPrimAtPath(str(prim.GetPath())))
            fabric_prim.GetFabricHierarchyLocalMatrixAttr().Set(
                usdrt.Gf.Matrix4d(*(local[row][col] for row in range(4) for col in range(4)))
            )
    usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
        fabric_stage.GetFabricId(), fabric_stage.GetStageIdAsStageId()
    ).update_world_xforms()


def render_cameras(sim: IsaacSim, due: list[CameraRecord]) -> None:
    """Publish due observations, owning their buffers across later shared renders.

    Drop alpha and cache ``[N,H,W,3]`` uint8 (rgb) /
    ``[N,H,W,1]`` float32 (depth) outputs at each camera's configured cadence."""
    for camera_record in due:
        out = sim.tiled_cameras[camera_record.name].data.output
        if "rgb" in camera_record.config.data_types:
            rgb = sim.backend.tensor(out["rgb"])[..., :3]  # [N,H,W,3], drop alpha
            if rgb.dtype != torch.uint8:
                rgb = rgb.clamp(0, 255).to(torch.uint8)
            camera_record.set_buffer("rgb", rgb.clone())
        if "depth" in camera_record.config.data_types:
            # distance_to_image_plane is float32 meters, image-plane. Ensure a trailing channel
            # dim; CameraRecord applies the configured cross-backend no-return convention.
            depth = sim.backend.camera_depth(out["distance_to_image_plane"]).to(torch.float32)
            camera_record.set_buffer("depth", (depth if depth.ndim == 4 else depth.unsqueeze(-1)).clone())


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


def _mount_parent_path(mount: SensorMountConfig, *, backend: IsaacSimBackendAdapter) -> str:
    namespace = backend.env_regex_ns
    if mount.target_kind == "robot_link":
        return resolve_robot_link_prim_expression(mount.target, backend=backend)
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


def _robot_mesh_prim_paths(excluded_link: str | None, *, backend: IsaacSimBackendAdapter) -> list[str]:
    """Return robot visual-geometry expressions, omitting one native body owner."""
    source_root = "/World/envs/env_0/Robot"
    expression_root = f"{backend.env_regex_ns}/Robot"
    excluded_body_path = (
        None
        if excluded_link is None
        else resolve_robot_link_rigid_body_prim_expression(excluded_link, backend=backend).replace(
            backend.env_regex_ns, "/World/envs/env_0", 1
        )
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
        robot_mesh_prim_paths=partial(_robot_mesh_prim_paths, backend=sim.backend),
        # InteractiveScene.rigid_objects includes standalone objects and every static or dynamic
        # body expanded from scene files, so this covers all Holosoma scene-object meshes.
        actor_names=sim.scene.rigid_objects,
        discover_mesh_prim_paths=_visual_mesh_prim_paths,
        env_regex_ns=sim.backend.env_regex_ns,
    )


def build_lidars(sim: IsaacSim) -> None:
    """Create one mutable MultiMeshRayCaster per configured LiDAR before environment cloning."""
    control_hz = sim.simulator_config.sim.fps / sim.simulator_config.sim.control_decimation_steps
    lidars = [(name, config) for name, config in sim.sensor_config.items() if isinstance(config, LidarSensorConfig)]
    if not lidars:
        return

    for name, lidar in lidars:
        from isaaclab.sensors import MultiMeshRayCasterCfg

        parent = _mount_parent_path(lidar.mount, backend=sim.backend)
        if lidar.mount.target_kind == "world":
            mount_path, offset = sim.backend.world_lidar_mount(name, lidar.mount.position, lidar.mount.orientation)
        else:
            mount_path = parent
            offset = MultiMeshRayCasterCfg.OffsetCfg(
                pos=tuple(lidar.mount.position),
                rot=sim.backend.config_quat_to_native(lidar.mount.orientation),
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
        sensor = sim.backend.mutable_ray_caster_class()(cfg)
        sim.scene.sensors[name] = sensor


def register_lidars(sim: IsaacSim, manager: SensorManager) -> None:
    for name, config in sim.sensor_config.items():
        if isinstance(config, LidarSensorConfig):
            manager.register_lidar(name, config)


@contextmanager
def _app_updates_without_physics(*, uses_kit: bool = True) -> Iterator[None]:
    """Keep Kit app updates from stepping physics, as ``SimulationContext.play``/``pause`` do.

    Isaac Lab's ``XformPrimView`` syncs Fabric from USD on its first read through an app update
    (``isaaclab.sim.utils.update_stage``), which would otherwise step physics while the timeline plays.
    """
    if not uses_kit:
        yield
        return
    import carb

    settings = carb.settings.get_settings()
    previous = settings.get_as_bool("/app/player/playSimulations")
    settings.set_bool("/app/player/playSimulations", False)
    try:
        yield
    finally:
        settings.set_bool("/app/player/playSimulations", previous)


def render_lidars(sim: IsaacSim) -> None:
    if sim.sensor_manager is None:
        return
    for lidar_record in sim.sensor_manager.collect_lidars_due():
        pattern = lidar_record.pattern_at(sim.time())
        sensor = sim.scene.sensors.get(lidar_record.name)
        if not isinstance(sensor, sim.backend.mutable_ray_caster_class()):
            raise TypeError(f"LiDAR '{lidar_record.name}' has no mutable MultiMeshRayCaster.")
        sensor.set_ray_directions(pattern.directions)
        with _app_updates_without_physics(uses_kit=sim.backend.uses_kit):
            native = sensor.data
        distances = torch.linalg.vector_norm(sim.backend.tensor(native.ray_hits_w) - sensor.ray_starts_w, dim=-1)
        finalize_lidar_returns(lidar_record, pattern.directions, distances)


def build_sensors(sim: IsaacSim) -> None:
    """Build native sensors before environment cloning."""
    needs_expanded_robot = any(
        isinstance(config, LidarSensorConfig) or config.mount.target_kind == "robot_link"
        for config in sim.sensor_config.values()
    )
    if needs_expanded_robot:
        # IsaacLab cannot author children or resolve exact targets through instance proxies.
        set_instanceable(get_current_stage(sim.backend), "/World/envs/env_0/Robot", False)
    if sim.backend.is_isaacsim6 and any(isinstance(config, LidarSensorConfig) for config in sim.sensor_config.values()):
        _remove_disabled_rigid_body_markers(get_current_stage(sim.backend))
    build_cameras(sim)
    build_lidars(sim)


def _remove_disabled_rigid_body_markers(stage: Any) -> None:
    """Let Lab 3 native LiDAR track the enabled owner of collapsed fixed-link geometry.

    Some importers leave an explicitly disabled RigidBodyAPI on a fixed-link prim. PhysX ignores
    that API, while Lab 3's native ray caster resolves owners by API presence. Removing the disabled
    marker on the authored source makes both resolve the same owner without replacing the native
    Warp ray/mesh pipeline or changing the prim's transform.
    """
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath("/World/envs/env_0")
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not is_rigid_body_enabled(prim):
            # Preserve the imported link's identity when visual descendants share
            # its name, while native LiDAR still sees only the enabled physics owner.
            prim.SetCustomDataByKey(_DISABLED_RIGID_BODY_LINK_KEY, True)
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)


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
    """Capture when needed, then publish only observations due on the control clock."""
    due = sim.sensor_manager.collect_due() if sim.sensor_manager is not None else []
    recording = sim.video_recorder is not None and sim.video_recorder.is_recording
    native_recorder = None
    if due or recording:
        cameras = (
            list(sim.tiled_cameras.values())
            if sim.backend.shared_camera_rendering
            else [sim.tiled_cameras[record.name] for record in due]
        )
        if recording and sim.backend.is_isaacsim6:
            from .native_video_recorder import NativeVideoRecorder

            native_recorder = cast(NativeVideoRecorder, sim.video_recorder)
            native_recorder.prepare_frame()
            cameras.append(native_recorder.camera)
        sim.render_for_sensors()
        sim.backend.refresh_cameras(cameras)
        if native_recorder is not None:
            native_recorder.mark_frame_ready()
    render_cameras(sim, due)
    render_lidars(sim)
