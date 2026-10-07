from pathlib import Path
from typing import Any

import torch
import trimesh
import warp as wp

from holosoma.sensors.warp.camera_config.base_depth_camera_config import BaseDepthCameraConfig
from holosoma.sensors.warp.sensor_utils import convert_to_warp_mesh


class BaseSensor:
    """Base class and shared utilities for Warp-based sensors."""

    def __init__(
        self,
        num_envs: int,
        config: BaseDepthCameraConfig,
        terrain: trimesh.Trimesh,
        device: str = "cuda:0",
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.ray_cast_bodies = list(config.ray_cast_bodies.keys())

        # Offpath obstacle bodies
        self.add_offpath_obstacle = config.add_offpath_obstacle
        if self.add_offpath_obstacle:
            self.ray_cast_bodies.extend(list(config.offpath_obstacle_bodies.keys()))

        self.is_dyna_mesh = config.dynamic_meshes
        self.init_warp(device)
        self.build_warp_meshes(
            terrain,
            config,
        )
        self.init_ray_cast_body_poses_and_quats()

    def init_ray_cast_body_poses_and_quats(self) -> None:
        self.ray_cast_body_poses_tensor = torch.zeros(self.num_envs, len(self.ray_cast_bodies), 3, device=self.device)
        self.ray_cast_body_quats_tensor = torch.zeros(self.num_envs, len(self.ray_cast_bodies), 4, device=self.device)
        self.ray_cast_body_quats_tensor[..., 3] = 1.0
        self.ray_cast_body_poses = wp.from_torch(
            self.ray_cast_body_poses_tensor.view(self.num_envs, len(self.ray_cast_bodies), 3), dtype=wp.vec3
        )
        self.ray_cast_body_quats = wp.from_torch(
            self.ray_cast_body_quats_tensor.view(self.num_envs, len(self.ray_cast_bodies), 4), dtype=wp.quat
        )

    def init_warp(self, device: str) -> None:
        """Initialize Warp and select the configured device."""
        wp.init()
        try:
            wp.set_device(device)
        except Exception as error:
            raise RuntimeError(f"Unable to select Warp device {device!r}.") from error

    def _load_warp_mesh(self, mesh_path: Path) -> wp.Mesh:
        try:
            mesh = trimesh.load_mesh(mesh_path, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                raise TypeError(f"Expected a triangle mesh, got {type(mesh).__name__}.")
            return convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
        except Exception as error:
            raise RuntimeError(f"Failed to load mesh {mesh_path}.") from error

    def build_warp_meshes(self, terrain: trimesh.Trimesh, config: BaseDepthCameraConfig) -> None:
        """Build the per-body and terrain meshes used by Warp ray casting."""
        robot_mesh_paths = [Path(config.asset_meshes_root) / name for name in config.ray_cast_bodies.values()]
        if self.add_offpath_obstacle:
            if config.offpath_obstacle_meshes_root is None:
                raise ValueError("add_offpath_obstacle is set but offpath_obstacle_meshes_root is None.")
            obstacle_root = Path(config.offpath_obstacle_meshes_root)
            robot_mesh_paths.extend(obstacle_root / name for name in config.offpath_obstacle_bodies.values())

        self.robot_body_meshes = [self._load_warp_mesh(path) for path in robot_mesh_paths]
        self.num_robot_bodies = len(self.robot_body_meshes)
        self.robot_mesh_ids: wp.array[Any] = wp.array(
            [mesh.id for mesh in self.robot_body_meshes],
            dtype=wp.uint64,
            device=self.device,
        )

        try:
            self.terrain_mesh = convert_to_warp_mesh(terrain.vertices, terrain.faces, device=self.device)
        except Exception as error:
            raise RuntimeError("Failed to create the Warp terrain mesh.") from error
        self.terrain_mesh_id = self.terrain_mesh.id
