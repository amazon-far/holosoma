"""Mutable LiDAR types selected explicitly by the Isaac Lab backend."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import isaaclab.sim as sim_utils
import torch
from isaaclab.utils.math import quat_apply

from holosoma.simulator.isaacsim.prim_utils import is_rigid_body_enabled


@lru_cache(maxsize=1)
def legacy_mutable_ray_caster_class() -> type[Any]:
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
