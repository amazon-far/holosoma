"""Isaac Sim mounted camera and LiDAR sensor setup.

RTX cameras are ``TiledCamera`` sensors created before ``clone_environments`` as children of their
mount bodies, so the USD transform hierarchy makes them follow those bodies across cloned envs.
Ray-cast cameras register native scene mesh views and render through the bespoke Warp kernel.
These module-level functions take the simulator (the ``fields.py`` / MuJoCo
shared sensor setup convention) and drive that pipeline — prim-path resolution, the FOV->aperture
pinhole cfg, per-env creation, ``SensorManager`` registration, and the per-control-step render
read-out. The per-camera ``TiledCamera`` map lives on the simulator as ``sim.tiled_cameras``,
written before clone and read at render.

Native sensors are built before cloning, then registered with the shared sensor manager after
the scene is built.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import carb
import torch
import isaaclab.sim as sim_utils
import warp as wp
from isaaclab.sensors import MultiMeshRayCaster, TiledCamera, TiledCameraCfg, patterns
from isaaclab.utils.math import quat_apply

from holosoma.config_types.frequency import resolve_decimation
from holosoma.config_types.sensor import (
    CameraSensorConfig,
    IsaacSimLidarConfig,
    LidarBodyFilterConfig,
    LidarSensorConfig,
    is_isaacsim_raycast_camera,
)
from holosoma.simulator.isaacsim.camera_projection import camera_spawn_cfg_for
from holosoma.simulator.isaacsim.raycast_targets import (
    TERRAIN_PRIM_PATH,
    RobotMeshPoses,
    _is_path_at_or_below,
    assemble_default_mesh_prim_paths,
    select_mesh_prim_paths,
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
from holosoma.simulator.shared.raycast_depth_camera import RayCastDepthCamera, RaycastScene, RaycastSceneMeshes

if TYPE_CHECKING:
    from holosoma.config_types.sensor import LidarRayPatternConfig, SensorMountConfig
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim
    from holosoma.simulator.shared.sensor_manager import CameraRecord


def build_cameras(sim: IsaacSim) -> None:
    """Build an RTX camera or a native ray-cast geometry view before environment cloning.

    The camera optical convention is OpenGL (-Z forward, +Y up), the same frame holosoma uses,
    so the mount offset passes through with ``convention="opengl"`` and no extra rotation.
    Mount quat is the config-layer ``[w,x,y,z]`` (IsaacLab OffsetCfg.rot is also w-first). Populates
    ``sim.tiled_cameras`` and registers each camera prim into ``sim.scene.sensors``.
    """
    sim.tiled_cameras = {}
    for cam_name, cam in sim.sensor_config.items():
        if is_isaacsim_raycast_camera(cam):
            from isaaclab.sensors import MultiMeshRayCasterCfg

            # The native sensor participates in the scene's initialization/reset lifecycle, sharing
            # USD extraction, mesh caching and transform tracking with LiDAR. It never casts rays.
            cfg = MultiMeshRayCasterCfg(
                prim_path=_camera_mount_prim_path(cam.mount),
                pattern_cfg=patterns.PatternBaseCfg(func=_mesh_view_pattern),
                mesh_prim_paths=[],
                debug_vis=False,
            )
            sim.scene.sensors[cam_name] = SceneMeshRayCaster(cfg, partial(_camera_mesh_prim_paths, sim, cam), sim=sim)
        elif isinstance(cam, CameraSensorConfig):
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
    """Register every camera; ray-cast cameras are created here and can re-render a subset of envs."""
    from pxr import Sdf

    sim.raycast_cameras = {}
    for name, config in sim.sensor_config.items():
        if not isinstance(config, CameraSensorConfig):
            continue
        record = manager.register_camera(name, config)
        if is_isaacsim_raycast_camera(config):
            scene = cast(RaycastScene, sim.scene.sensors[name])
            sim.raycast_cameras[name] = RayCastDepthCamera(sim, config, scene)
            record.rerender = partial(sim.raycast_cameras[name].render, record)
            record.write_mount_pose = sim.raycast_cameras[name].write_mount_pose
            continue
        record.write_mount_pose = partial(_write_camera_mount_pose, sim, name)
        # clone_environments(copy_from_source=False) inherits env_0's USD opinions. Author
        # only each camera's two local-pose opinions so a later env_0 write cannot move
        # untouched clones. This is startup-only, before Camera._view exists; copying entire
        # environments would duplicate their assets. Batch USD notices in one ChangeBlock.
        camera = sim.tiled_cameras[name]
        with Sdf.ChangeBlock():
            for prim in sim_utils.find_matching_prims(camera.cfg.prim_path):
                for attr_name in ("xformOp:translate", "xformOp:orient"):
                    attribute = prim.GetAttribute(attr_name)
                    attribute.Set(attribute.Get())


def _write_camera_mount_pose(sim: IsaacSim, name: str, env_ids: torch.Tensor, poses: torch.Tensor) -> None:
    camera = sim.tiled_cameras[name]
    camera._view.set_local_poses(poses[:, :3], poses[:, [6, 3, 4, 5]], env_ids)
    # Isaac Lab 2.3.2's supported local-pose API writes USD even in Fabric mode. A world-pose
    # write would lose the mounting offset when the parent moves. Sync the local hierarchy
    # matrix too, so Fabric rendering follows that parent at the newly authored offset.
    sync_camera_fabric_transforms(sim, camera_name=name, env_ids=env_ids)
    camera._is_outdated[env_ids] = True


def sync_camera_fabric_transforms(
    sim: IsaacSim, *, camera_name: str | None = None, env_ids: torch.Tensor | None = None
) -> None:
    """Reset each camera's Fabric local transform to its authored mount offset.

    Call once after ``sim.reset()``. The renderer places cameras from Fabric. Isaac Lab initializes
    each TiledCamera during the reset; its first pose read syncs the camera into Fabric through an
    app update that also steps physics. The camera's world pose comes from USD, where the mount body
    has not moved, so a camera synced after its mount body moved gets a local offset derived against
    the moved body, and renders off its mount for the whole run by an amount that depends on the
    physics dt. This was measured on one of four cameras (the last created; 4-8 cm and 5-8 deg on a
    wrist camera); which cameras are hit is not explained. The USD offset stays correct, so it is
    copied back into Fabric for every camera.
    """
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
    cameras = sim.tiled_cameras.values() if camera_name is None else [sim.tiled_cameras[camera_name]]
    for camera in cameras:
        # Use the initialized view's cached ordering, matching set_local_poses' env indices.
        prims = camera._view._prims
        if env_ids is not None:
            prims = [prims[i] for i in env_ids.tolist()]
        for prim in prims:
            local = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
            fabric_prim = Rt.Xformable(fabric_stage.GetPrimAtPath(str(prim.GetPath())))
            fabric_prim.GetFabricHierarchyLocalMatrixAttr().Set(
                usdrt.Gf.Matrix4d(*(local[row][col] for row in range(4) for col in range(4)))
            )
    usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
        fabric_stage.GetFabricId(), fabric_stage.GetStageIdAsStageId()
    ).update_world_xforms()


def render_cameras(sim: IsaacSim, due: list[CameraRecord]) -> None:
    """Cache each due camera's frame into its buffer: ray-cast cameras cast here, TiledCameras are read out.

    TiledCamera output comes from the render ``render_sensors`` just ran; drops alpha, and writes a
    ``[N,H,W,3]`` uint8 (rgb) / ``[N,H,W,1]`` float32 (depth) frame."""
    for camera_record in due:
        if camera_record.name in sim.raycast_cameras:
            sim.raycast_cameras[camera_record.name].render(camera_record)
            continue
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


def _mesh_view_pattern(_native_config: Any, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal native bookkeeping pattern; the scene mesh provider does no ray casting."""
    return torch.zeros(1, 3, device=device), torch.tensor([[0.0, 0.0, -1.0]], device=device)


def _mesh_targets(paths: list[str]) -> list[Any]:
    """Configure scene geometry once for both LiDAR and depth views."""
    from isaaclab.sensors import MultiMeshRayCasterCfg

    return [
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr=path,
            # Holosoma clones geometry from env_0, so every env shares env_0's cached mesh.
            is_shared=True,
            track_mesh_transforms=not _is_path_at_or_below(path, TERRAIN_PRIM_PATH),
        )
        for path in paths
    ]


class MutableMultiMeshRayCaster(MultiMeshRayCaster):
    """Multi-mesh ray caster with fixed-shape directions and per-environment mount offsets."""

    def __init__(self, cfg: Any) -> None:
        self._mesh_owner_paths: dict[str, str] = {}
        super().__init__(cfg)

    def _initialize_rays_impl(self) -> None:
        # A native context reset rebuilds rays from cfg; retain runtime offsets across it.
        previous_starts = self.ray_starts.clone() if hasattr(self, "_mount_orientations") else None
        super()._initialize_rays_impl()
        if previous_starts is not None:
            self.ray_starts.copy_(previous_starts)
        else:
            self._mount_orientations = torch.tensor(
                self.cfg.offset.rot, device=self.device, dtype=self.ray_directions.dtype
            ).repeat(self.ray_directions.shape[0], 1)

    def set_mount_pose(self, env_ids: torch.Tensor, poses: torch.Tensor) -> None:
        self.ray_starts[env_ids] = poses[:, None, :3]
        self._mount_orientations[env_ids] = poses[:, [6, 3, 4, 5]]
        self.mark_outdated()

    @property
    def ray_starts_w(self) -> torch.Tensor:
        """Return IsaacLab's current world-frame ray origins."""
        return self._ray_starts_w

    def set_ray_directions(self, directions: torch.Tensor) -> None:
        expected = (self.num_rays, 3)
        if directions.shape != expected:
            raise ValueError(f"RayCaster directions must keep shape {expected}, got {directions.shape}.")
        local = directions.to(device=self.ray_directions.device, dtype=self.ray_directions.dtype)
        offset_quat = self._mount_orientations[:, None, :].expand(-1, self.num_rays, -1)
        self.ray_directions.copy_(quat_apply(offset_quat, local.unsqueeze(0).expand_as(self.ray_directions)))
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
                f"Raycast target '{target_prim_path}' resolves to {len(mesh_prims)} meshes but "
                f"{len(view_prims)} trackable owners at '{current_path_expr}'."
            )

        positions = []
        orientations = []
        for target_prim, view_prim in zip(mesh_prims, view_prims):
            position, orientation = sim_utils.resolve_prim_pose(target_prim, view_prim)
            positions.append(torch.tensor(position, dtype=torch.float32, device=self.device))
            orientations.append(torch.tensor(orientation, dtype=torch.float32, device=self.device))
        self._mesh_owner_paths[target_prim_path] = current_path_expr
        return prim_view, (torch.stack(positions), torch.stack(orientations))

    def _update_mesh_transforms(self, *, skip_paths: frozenset[str] = frozenset()) -> None:
        """Refresh cached mesh poses with correct body-to-mesh transform composition."""
        from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view
        from isaaclab.utils.math import quat_apply, quat_mul

        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            count = self._num_meshes_per_env[target_cfg.prim_expr]
            if not target_cfg.track_mesh_transforms or target_cfg.prim_expr in skip_paths:
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

    def _update_buffers_impl(self, env_ids: Any) -> None:
        """Update LiDAR rays against the shared cached scene geometry."""
        from isaaclab.utils.warp import raycast_dynamic_meshes

        self._update_ray_infos(env_ids)
        self._update_mesh_transforms()
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


class SceneMeshRayCaster(MutableMultiMeshRayCaster):
    """Geometry-only view of the native mesh cache for the bespoke depth renderer."""

    def __init__(self, cfg: Any, resolve_paths: Callable[[], list[str]], *, sim: IsaacSim) -> None:
        self._resolve_paths = resolve_paths
        self._sim = sim
        self._depth_meshes: RaycastSceneMeshes | None = None
        self._robot_mesh_poses: RobotMeshPoses | None = None
        super().__init__(cfg)

    def _initialize_warp_meshes(self) -> None:
        # Physics starts after all Holosoma assets have loaded and environments have cloned,
        # so selection resolves against the complete scene here.
        targets = _mesh_targets(self._resolve_paths())
        self.cfg.mesh_prim_paths = targets
        self._raycast_targets_cfg = targets
        self._depth_meshes = None
        self._robot_mesh_poses = None
        self._mesh_owner_paths.clear()
        super()._initialize_warp_meshes()

    def _bind_robot_mesh_poses(self) -> RobotMeshPoses | None:
        """Reuse native physics-owner resolution and body-to-mesh offsets once."""
        robot_root = "/World/envs/env_.*/Robot"
        indices, slots, positions, orientations, paths = [], [], [], [], []
        cursor = 0
        for target in self._raycast_targets_cfg:
            count = self._num_meshes_per_env[target.prim_expr]
            if target.track_mesh_transforms and _is_path_at_or_below(target.prim_expr, robot_root):
                if count != 1:
                    raise ValueError(f"Depth discovery must resolve one mesh per target: {target.prim_expr}.")
                owner = sim_utils.find_first_matching_prim(self._mesh_owner_paths[target.prim_expr])
                if (
                    not owner.IsValid()
                    or not _is_path_at_or_below(str(owner.GetPath()), "/World/envs/env_0/Robot")
                    or not is_rigid_body_enabled(owner)
                ):
                    raise ValueError(f"Robot mesh '{target.prim_expr}' has no enabled robot physics owner.")
                if owner.GetName() not in self._sim.body_names:
                    # Bodies outside the robot config (e.g. kept fixed-joint links) stay on their native view.
                    cursor += count
                    continue
                indices.append(self._sim.find_rigid_body_indice(owner.GetName()))
                slots.append(cursor)
                position, orientation = self.mesh_offsets[target.prim_expr]
                positions.append(position.reshape(self._num_envs, 3))
                orientations.append(orientation.reshape(self._num_envs, 4))
                paths.append(target.prim_expr)
            cursor += count
        if not paths:
            # Actor/ground-only selections have no robot bindings.
            return None
        return RobotMeshPoses(
            target_paths=frozenset(paths),
            body_ids=torch.tensor(indices, dtype=torch.long, device=self.device),
            mesh_slots=torch.tensor(slots, dtype=torch.long, device=self.device),
            position_offsets=torch.stack(positions, dim=1),
            orientation_offsets=torch.stack(orientations, dim=1)[..., [1, 2, 3, 0]].contiguous(),
        )

    def prepare_meshes(self) -> RaycastSceneMeshes:
        if not self.is_initialized:
            raise RuntimeError("Raycast scene geometry requires initialized IsaacLab scene sensors.")
        if self._depth_meshes is None:
            self._robot_mesh_poses = self._bind_robot_mesh_poses()
            self._mesh_quaternions_xyzw = torch.empty_like(self._mesh_orientations_w)
            self._depth_meshes = RaycastSceneMeshes(
                mesh_ids=self._mesh_ids_wp,
                positions=wp.from_torch(self._mesh_positions_w, dtype=wp.vec3),
                orientations=wp.from_torch(self._mesh_quaternions_xyzw, dtype=wp.quat),
                num_meshes=self._mesh_positions_w.shape[1],
            )
            # prepare_sim() refreshes Holosoma body tensors after sensor initialization.
            # Seed poses through initialized native views; captures use the batched bindings.
            self._update_mesh_transforms()
            self._copy_mesh_quaternions()
        return self._depth_meshes

    def update_mesh_poses(self) -> None:
        poses = self._robot_mesh_poses
        self._update_mesh_transforms(skip_paths=poses.target_paths if poses is not None else frozenset())
        if poses is not None:
            poses.update(self._sim, self._mesh_positions_w, self._mesh_orientations_w)
        self._copy_mesh_quaternions()

    def _copy_mesh_quaternions(self) -> None:
        self._mesh_quaternions_xyzw[..., :3].copy_(self._mesh_orientations_w[..., 1:])
        self._mesh_quaternions_xyzw[..., 3].copy_(self._mesh_orientations_w[..., 0])

    def _update_buffers_impl(self, env_ids: Any) -> None:
        # InteractiveScene can eagerly update sensors; this provider must never run a second
        # ray-casting pass. Depth explicitly refreshes mesh poses at capture/rerender time.
        pass


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
    """Select geometry that represents an optical surface for ray-cast sensor discovery.

    USD has no universal optical ray-cast surface schema. Holosoma therefore follows USD imaging
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
    # A hidden ground (terrain ``hide_visual``) still collides, so ray sensors keep hitting it.
    hidden = imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
    if hidden and not _is_path_at_or_below(str(prim.GetPath()), TERRAIN_PRIM_PATH):
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


def _body_filter_target(body_filter: LidarBodyFilterConfig, mount: SensorMountConfig) -> tuple[str, str] | None:
    if body_filter.target_kind == "none":
        return None
    if body_filter.target_kind == "mount":
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


def _default_mesh_prim_paths(sim: IsaacSim, body_filter: LidarBodyFilterConfig, mount: SensorMountConfig) -> list[str]:
    excluded_target = _body_filter_target(body_filter, mount)
    if excluded_target is not None and excluded_target[0] == "actor":
        actor_name = excluded_target[1]
        if actor_name not in sim.scene.rigid_objects:
            raise ValueError(
                f"Ray-cast body filter target 'actor:{actor_name}' is not registered. Known actors: "
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


def _camera_mesh_prim_paths(sim: IsaacSim, camera: CameraSensorConfig) -> list[str]:
    """Query the loaded Holosoma scene, then apply the camera's object/mesh path rules.

    ``isaacsim.body_filter`` applies to the default selection only; explicit includes bypass it.
    """
    isaacsim = camera.isaacsim
    no_filter = LidarBodyFilterConfig(target_kind="none")
    body_filter = isaacsim.body_filter if isaacsim.raycast_meshes.include_paths is None else no_filter
    paths = _default_mesh_prim_paths(sim, body_filter, camera.mount)
    return select_mesh_prim_paths(
        paths,
        camera.isaacsim.raycast_meshes,
        resolve_matching_paths=lambda expression: (
            str(prim.GetPath()) for prim in sim_utils.find_matching_prims(expression)
        ),
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
            )
        else:
            mount_path = parent
        # The world mount prim is an identity anchor in the env frame. Every mount kind uses
        # the mutable ray offsets, so world offsets can change independently in each env too.
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
            mesh_paths = _default_mesh_prim_paths(sim, lidar.body_filter, lidar.mount)
        else:
            mesh_paths = isaacsim_config.mesh_prim_paths
        cfg = MultiMeshRayCasterCfg(
            prim_path=mount_path,
            offset=offset,
            update_period=0.0,
            ray_alignment="base",
            pattern_cfg=native_pattern,
            debug_vis=sim.debug_viz_enabled,
            mesh_prim_paths=_mesh_targets(mesh_paths),
            max_distance=lidar.far,
        )
        sensor = MutableMultiMeshRayCaster(cfg)
        sim.scene.sensors[name] = sensor


def register_lidars(sim: IsaacSim, manager: SensorManager) -> None:
    for name, config in sim.sensor_config.items():
        if isinstance(config, LidarSensorConfig):
            record = manager.register_lidar(name, config)
            record.write_mount_pose = partial(_write_lidar_mount_pose, sim, name)


def _write_lidar_mount_pose(sim: IsaacSim, name: str, env_ids: torch.Tensor, poses: torch.Tensor) -> None:
    sensor = sim.scene.sensors[name]
    if not isinstance(sensor, MutableMultiMeshRayCaster):
        raise TypeError(f"LiDAR '{name}' has no mutable MultiMeshRayCaster.")
    sensor.set_mount_pose(env_ids, poses)


@contextmanager
def _app_updates_without_physics() -> Iterator[None]:
    """Keep Kit app updates from stepping physics, as ``SimulationContext.play``/``pause`` do.

    Isaac Lab's ``XformPrimView`` syncs Fabric from USD on its first read through an app update
    (``isaaclab.sim.utils.update_stage``), which would otherwise step physics while the timeline plays.
    """
    settings = carb.settings.get_settings()
    previous = settings.get_as_bool("/app/player/playSimulations")
    settings.set_bool("/app/player/playSimulations", False)
    try:
        yield
    finally:
        settings.set_bool("/app/player/playSimulations", previous)


def render_lidars(sim: IsaacSim) -> None:
    for lidar_record in sim.sensor_manager.collect_lidars_due():
        pattern = lidar_record.pattern_at(sim.time())
        sensor = sim.scene.sensors.get(lidar_record.name)
        if not isinstance(sensor, MutableMultiMeshRayCaster):
            raise TypeError(f"LiDAR '{lidar_record.name}' has no mutable MultiMeshRayCaster.")
        sensor.set_ray_directions(pattern.directions)
        with _app_updates_without_physics():
            native = sensor.data
        distances = torch.linalg.vector_norm(native.ray_hits_w - sensor.ray_starts_w, dim=-1)
        finalize_lidar_returns(lidar_record, pattern.directions, distances)


def build_sensors(sim: IsaacSim) -> None:
    """Build native sensors before environment cloning."""
    has_lidar = any(isinstance(config, LidarSensorConfig) for config in sim.sensor_config.values())
    has_raycast_camera = any(is_isaacsim_raycast_camera(config) for config in sim.sensor_config.values())
    has_rtx_camera_on_robot = any(
        isinstance(config, CameraSensorConfig)
        and not is_isaacsim_raycast_camera(config)
        and config.mount.target_kind == "robot_link"
        for config in sim.sensor_config.values()
    )
    if has_lidar or has_raycast_camera or has_rtx_camera_on_robot:
        # IsaacLab cannot author children or resolve exact targets through instance proxies.
        set_instanceable(get_current_stage(), "/World/envs/env_0/Robot", False)
    build_cameras(sim)
    build_lidars(sim)


def create_sensors(sim: IsaacSim) -> None:
    """Register native sensors with the simulator's shared manager."""
    if not sim.sensor_config:
        return
    manager = sim.sensor_manager
    register_cameras(sim, manager)
    register_lidars(sim, manager)


def initialize_sensors(sim: IsaacSim) -> None:
    """Bind depth renderers once scene assets, body indices and native sensors are initialized."""
    for camera in sim.raycast_cameras.values():
        camera.initialize()


def render_sensors(sim: IsaacSim) -> None:
    """Render if a camera is due or video is recording, then capture the sensors due this control step."""
    due = sim.sensor_manager.collect_due()
    recording = sim.video_recorder is not None and sim.video_recorder.is_recording
    if any(record.name in sim.tiled_cameras for record in due) or recording:
        sim.render_for_sensors()
    render_cameras(sim, due)
    render_lidars(sim)
