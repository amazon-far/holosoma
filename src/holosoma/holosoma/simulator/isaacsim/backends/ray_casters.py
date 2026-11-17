"""Mutable LiDAR types selected explicitly by the Isaac Lab backend."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import isaaclab.sim as sim_utils
import torch
from isaaclab.utils.math import quat_apply

from holosoma.simulator.isaacsim.prim_utils import is_rigid_body_enabled
from holosoma.simulator.isaacsim.raycast_targets import RobotMeshPoses, _is_path_at_or_below

if TYPE_CHECKING:
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim


@lru_cache(maxsize=1)
def legacy_mutable_ray_caster_class() -> type[Any]:
    """Create Lab 2's mutable LiDAR type without importing it in standalone Lab 3 runs."""
    from isaaclab.sensors import MultiMeshRayCaster

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

    return MutableMultiMeshRayCaster


@lru_cache(maxsize=1)
def physx_mutable_ray_caster_class() -> type[Any]:
    """Keep Lab 3's persistent Warp ray/mesh buffers and native pose tracking.

    A concrete PhysX class is used because subclassing the backend-dispatch factory does not
    retain our overrides. Only the fixed-shape scan-pattern update is added; native kernels
    continue to handle tracked mesh offsets and final-pose ray casting.
    """
    from isaaclab_physx.sensors import MultiMeshRayCaster

    class MutableWarpMultiMeshRayCaster(_MutableWarpRayDirections, MultiMeshRayCaster):
        pass

    return MutableWarpMultiMeshRayCaster


@lru_cache(maxsize=1)
def ovphysx_mutable_ray_caster_class() -> type[Any]:
    """Use the OVPhysX caster's native views without loading Kit physics."""
    from isaaclab_ov.sensors.ray_caster import MultiMeshRayCaster

    class MutableOvPhysxMultiMeshRayCaster(_MutableWarpRayDirections, MultiMeshRayCaster):
        pass

    return MutableOvPhysxMultiMeshRayCaster


@lru_cache(maxsize=1)
def newton_mutable_ray_caster_class() -> type[Any]:
    """Retain configured visual targets through Newton's concrete mesh caster."""
    from isaaclab_newton.sensors import LegacyMultiMeshRayCaster

    class MutableNewtonMultiMeshRayCaster(_MutableWarpRayDirections, LegacyMultiMeshRayCaster):
        pass

    return MutableNewtonMultiMeshRayCaster


class _MutableWarpRayDirections:
    """Update either native caster's fixed-size Warp pattern in place."""

    @property
    def ray_starts_w(self: Any) -> torch.Tensor:
        import warp as wp

        return wp.to_torch(self._ray_starts_w)

    def set_ray_directions(self: Any, directions: torch.Tensor) -> None:
        expected = (self.num_rays, 3)
        if directions.shape != expected:
            raise ValueError(f"RayCaster directions must keep shape {expected}, got {directions.shape}.")
        native = self.ray_directions.torch
        local = directions.to(device=native.device, dtype=native.dtype)
        offset = torch.tensor(self.cfg.offset.rot, device=native.device, dtype=native.dtype)
        native.copy_(quat_apply(offset.expand(self.num_rays, -1), local).unsqueeze(0).expand_as(native))
        self.mark_outdated()

    def mark_outdated(self: Any) -> None:
        self._is_outdated.fill_(True)


@lru_cache(maxsize=1)
def legacy_scene_mesh_ray_caster_class() -> type[Any]:
    """Expose Lab 2's native mesh cache to the existing ray-cast depth renderer."""
    import warp as wp

    from holosoma.simulator.isaacsim.sensor_setup import _mesh_targets
    from holosoma.simulator.shared.raycast_depth_camera import RaycastSceneMeshes

    # The cached SDK subclass is created lazily, outside mypy's static class graph.
    class SceneMeshRayCaster(legacy_mutable_ray_caster_class()):  # type: ignore[misc]
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

    return SceneMeshRayCaster
