from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import trimesh
import warp as wp

from holosoma.sensors.warp.base_sensor import BaseSensor
from holosoma.sensors.warp.camera_config.base_depth_camera_config import BaseDepthCameraConfig
from holosoma.sensors.warp.camera_kernels_warp import DepthCameraWarpKernels
from holosoma.sensors.warp.sensor_utils import quat_from_euler_xyz_tensor, torch_rand_float_tensor

if TYPE_CHECKING:
    # warp exposes Graph only through a runtime module __getattr__, so ``wp.Graph`` is unknown to
    # type checkers. Same type-check-time import as simulator/mujoco/backends/warp_backend.py.
    from warp.context import Graph


class CameraSensor(BaseSensor):
    """Warp ray-cast camera used by the depth observation pipeline."""

    def __init__(
        self,
        num_envs: int,
        config: BaseDepthCameraConfig,
        terrain: trimesh.Trimesh,
        device: str = "cuda:0",
    ) -> None:
        if config.segmentation_camera:
            raise NotImplementedError("Warp segmentation rendering is not implemented.")
        if config.return_pointcloud and config.dynamic_meshes:
            raise NotImplementedError("Dynamic robot meshes are not implemented for point-cloud rendering.")

        self.cfg = config
        self.num_sensors = int(config.num_sensors)
        self.camera_names = list(config.base_link_frame)
        if len(self.camera_names) != self.num_sensors:
            raise ValueError(
                f"num_sensors={self.num_sensors}, but base_link_frame defines {len(self.camera_names)} cameras."
            )
        missing_offsets = set(self.camera_names) - set(config.offset)
        if missing_offsets:
            raise ValueError(f"Missing camera offsets for: {sorted(missing_offsets)}.")

        super().__init__(num_envs, config, terrain, device)
        self.width = int(config.width)
        self.height = int(config.height)
        self.horizontal_fov = math.radians(config.horizontal_fov_deg)
        self.far_plane = float(config.max_range)
        self.calculate_depth = bool(config.calculate_depth)
        self.graph: Graph | None = None

        self._initialize_camera_matrix()
        self._create_camera_tensors()

    def _initialize_camera_matrix(self) -> None:
        center_x = self.width / 2
        center_y = self.height / 2
        focal_length = center_x / math.tan(self.horizontal_fov / 2)
        vertical_fov = 2 * math.atan(self.height / (2 * focal_length))
        focal_x = center_x / math.tan(self.horizontal_fov / 2)
        focal_y = center_y / math.tan(vertical_fov / 2)

        intrinsics = wp.mat44(
            focal_x,
            0.0,
            center_x,
            0.0,
            0.0,
            focal_y,
            center_y,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        self.intrinsics_inverse = wp.inverse(intrinsics)
        self.center_x = int(center_x)
        self.center_y = int(center_y)

    def _create_camera_tensors(self) -> None:
        image_shape: tuple[int, ...] = (self.num_envs, self.num_sensors, self.height, self.width)
        if self.cfg.return_pointcloud:
            image_shape += (3,)
        self.image_tensors = torch.zeros(image_shape, device=self.device, requires_grad=False)

        data_frame_rotation = torch.deg2rad(
            torch.as_tensor(self.cfg.offset_rot_base, device=self.device, dtype=torch.float32)
        )
        data_frame_quat = quat_from_euler_xyz_tensor(data_frame_rotation)
        self.camera_sensor_data_frame_quat = data_frame_quat.expand(self.num_envs, self.num_sensors, -1)

        vector_shape = (self.num_envs, self.num_sensors, 3)
        self.camera_sensor_translation = torch.zeros(vector_shape, device=self.device)
        self.camera_sensor_rotation = torch.zeros(vector_shape, device=self.device)
        for camera_index, camera_name in enumerate(self.camera_names):
            self.camera_sensor_translation[:, camera_index] = torch.as_tensor(
                self.cfg.offset[camera_name]["offset_pos"],
                device=self.device,
            )
            self.camera_sensor_rotation[:, camera_index] = torch.as_tensor(
                self.cfg.offset[camera_name]["offset_rot"],
                device=self.device,
            )

        self.camera_sensor_local_position = torch.empty_like(self.camera_sensor_translation)
        self.camera_sensor_local_orientation = torch.empty(
            (self.num_envs, self.num_sensors, 4),
            device=self.device,
        )
        self._initialize_local_poses()

        self.camera_sensor_position = torch.zeros_like(self.camera_sensor_local_position)
        self.camera_sensor_orientation = torch.zeros_like(self.camera_sensor_local_orientation)
        self.camera_sensor_orientation[..., 3] = 1.0
        self.set_pose_tensor(self.camera_sensor_position, self.camera_sensor_orientation)
        self.set_image_tensor(self.image_tensors)

    def _initialize_local_poses(self) -> None:
        if self.cfg.randomize_placement:
            min_translation = torch.empty_like(self.camera_sensor_translation)
            max_translation = torch.empty_like(self.camera_sensor_translation)
            min_rotation = torch.empty_like(self.camera_sensor_rotation)
            max_rotation = torch.empty_like(self.camera_sensor_rotation)
            for camera_index, camera_name in enumerate(self.camera_names):
                min_translation[:, camera_index] = torch.as_tensor(
                    self.cfg.min_translation[camera_name],
                    device=self.device,
                )
                max_translation[:, camera_index] = torch.as_tensor(
                    self.cfg.max_translation[camera_name],
                    device=self.device,
                )
                min_rotation[:, camera_index] = torch.as_tensor(
                    self.cfg.min_euler_rotation_deg[camera_name],
                    device=self.device,
                )
                max_rotation[:, camera_index] = torch.as_tensor(
                    self.cfg.max_euler_rotation_deg[camera_name],
                    device=self.device,
                )

            self.camera_sensor_local_position[:] = torch_rand_float_tensor(
                min_translation + self.camera_sensor_translation,
                max_translation + self.camera_sensor_translation,
            )
            local_euler_rotation = torch.deg2rad(
                torch_rand_float_tensor(
                    min_rotation + self.camera_sensor_rotation,
                    max_rotation + self.camera_sensor_rotation,
                )
            )
        else:
            self.camera_sensor_local_position[:] = self.camera_sensor_translation
            local_euler_rotation = torch.deg2rad(self.camera_sensor_rotation)

        self.camera_sensor_local_orientation[:] = quat_from_euler_xyz_tensor(local_euler_rotation)

    def create_render_graph_pointcloud(self, debug: bool = False) -> None:
        if not debug:
            wp.capture_begin(device=self.device)
        wp.launch(
            kernel=DepthCameraWarpKernels.draw_optimized_kernel_pointcloud,
            dim=(self.num_envs, self.num_sensors, self.width, self.height),
            inputs=[
                self.terrain_mesh_id,
                self.camera_position_array,
                self.camera_orientation_array,
                self.intrinsics_inverse,
                self.far_plane,
                self.pixels,
                self.cfg.pointcloud_in_world_frame,
            ],
            device=self.device,
        )
        if not debug:
            self.graph = wp.capture_end(device=self.device)

    def create_render_graph_depth_range(self, debug: bool = False) -> None:
        if not debug:
            wp.capture_begin(device=self.device)
        if self.is_dyna_mesh:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_depth_range_dynamic,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.terrain_mesh_id,
                    self.robot_mesh_ids,
                    self.ray_cast_body_poses,
                    self.ray_cast_body_quats,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.intrinsics_inverse,
                    self.far_plane,
                    self.pixels,
                    self.center_x,
                    self.center_y,
                    self.calculate_depth,
                    self.num_robot_bodies,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_depth_range,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.terrain_mesh_id,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.intrinsics_inverse,
                    self.far_plane,
                    self.pixels,
                    self.center_x,
                    self.center_y,
                    self.calculate_depth,
                ],
                device=self.device,
            )
        if not debug:
            self.graph = wp.capture_end(device=self.device)

    def set_image_tensor(self, pixels: torch.Tensor) -> None:
        warp_dtype = wp.vec3 if self.cfg.return_pointcloud else wp.float32
        self.pixels = wp.from_torch(pixels, dtype=warp_dtype)

    def set_pose_tensor(self, positions: torch.Tensor, orientations: torch.Tensor) -> None:
        self.camera_position_array = wp.from_torch(positions, dtype=wp.vec3)
        self.camera_orientation_array = wp.from_torch(orientations, dtype=wp.quat)

    def capture(self, debug: bool = False) -> torch.Tensor:
        if debug:
            if self.cfg.return_pointcloud:
                self.create_render_graph_pointcloud(debug=True)
            else:
                self.create_render_graph_depth_range(debug=True)
            return self.image_tensors

        if self.graph is None:
            if self.cfg.return_pointcloud:
                self.create_render_graph_pointcloud()
            else:
                self.create_render_graph_depth_range()
        assert self.graph is not None
        wp.capture_launch(self.graph)
        return self.image_tensors
