"""Stateful depth observation term for distillation training.

Port of far-tracking's ``image_features`` / ``warp_image_features`` classes,
adapted to holosoma's :class:`ObservationTermBase` interface.

The term owns a :class:`CameraSensor` (Warp ray-caster), a rolling depth buffer
for per-env latency sampling, and a Perlin-noise generator for hole masks.
Normal calls read the robot body poses from the simulator, capture and process
depth, and advance the buffer. Non-mutating calls return the last committed
output so final-observation and shape queries cannot advance sensor time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from holosoma.managers.observation.base import ObservationTermBase

from holosoma.utils.perlin_noise import InfiniteFractalPerlin3D
from holosoma.sensors.warp.camera_sensor import CameraSensor
from holosoma.sensors.warp.sensor_utils import (
    quat_mul_xyzw,
    tf_apply_xyzw,
)


class WarpDepthImageObsTerm(ObservationTermBase):
    """Stateful depth obs term rendering via Warp ray-casts.

    Parameters (via ``cfg.params``)
    -------------------------------
    camera_sensor_cfg : BaseDepthCameraConfig
        E.g. ``G1FlatRsD435iConfig``.
    command_name : str
        Key of the :class:`MotionCommand` registered in the command manager.
    latency_frame : tuple[int, int] | int
        Per-env latency sampled uniformly from the (inclusive) range, or a
        single fixed int.
    buffer_len : int
        Size of the rolling depth buffer (must exceed max latency).
    resize : tuple[int, int]
        (H, W) target after crop/resize. Must match the CNN backbone input.
    enable_holes : bool
        Whether to punch Perlin-noise holes into the depth image.
    terrain_mesh_vertices, terrain_mesh_faces : numpy array
        The ground mesh the camera rays cast against. Plumbed in by the terrain
        preprocess hook at config-build time; both must be non-None when the
        term instantiates or we raise.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params

        self.num_envs = env.num_envs
        self.device = env.device

        latency_frame_param = params["latency_frame"]
        if isinstance(latency_frame_param, (tuple, list)) and len(latency_frame_param) == 2:
            self.latency_frame_range = latency_frame_param
            self.latency_frame = None
        else:
            self.latency_frame_range = None
            self.latency_frame = latency_frame_param

        resized = params["resize"]
        self.buffer_len = params["buffer_len"]
        self.enable_holes = params.get("enable_holes", False)

        # Bicubic resize via functional interpolate rather than
        # ``torchvision.transforms.Resize`` so core keeps no torchvision
        # dependency. ``antialias=True`` is required for parity: it is
        # torchvision's default and produces bit-identical output, whereas the
        # interpolate default (False) diverges by ~0.12 on a normalized depth
        # image.
        self._resize_hw = (int(resized[0]), int(resized[1]))
        self.depth_buffer = torch.zeros(
            self.num_envs, self.buffer_len, resized[0], resized[1], device=self.device
        )
        self._cached_output = torch.zeros(self.num_envs, resized[0], resized[1], device=self.device)
        self.reset_episodes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if self.latency_frame_range is not None:
            assert self.latency_frame_range[1] < self.buffer_len, (
                "Max latency frame must be less than buffer length"
            )
        else:
            assert self.latency_frame < self.buffer_len, (
                "Latency frame must be less than buffer length"
            )

        # NOTE: command_manager is constructed *after* observation_manager in
        # BaseTask.__init__, so we cannot resolve the motion command here. Defer
        # the lookup and the prev-timestep buffer to the first __call__.
        self._command_name: str = params["command_name"]
        self.command = None
        self._prev_time_step: torch.Tensor | None = None

        self._init_camera(cfg, env)

        # Perlin hole generator — generate at 64x96 then interp to (H, W) to
        # keep perlin periodicity independent of the CNN resolution.
        height, width = 64, 96
        g_cuda = torch.Generator(device=self.device)
        resolutions_2d = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
        periods = [32, 16, 8, 4, 2]
        clouds_factors = [0.3**i for i in range(1)]
        self.generator = InfiniteFractalPerlin3D(
            (height, width),
            resolutions_2d,
            periods,
            clouds_factors,
            generator=g_cuda,
            batch_size=self.num_envs,
        )

        # Per-env constant offset simulating ~3 cm calibration bias.
        self.depth_offset = torch.randn(self.num_envs, device=self.device) * 0.03

    def _init_camera(self, cfg: Any, env: Any) -> None:
        params = cfg.params
        camera_sensor_cfg = params["camera_sensor_cfg"]
        if isinstance(camera_sensor_cfg, str):
            # dotted "module:ClassName" path; instantiate lazily so ObsTermCfg
            # params stay JSON-serializable for config YAML snapshots.
            from holosoma.utils.helpers import get_class

            camera_sensor_cfg = get_class(camera_sensor_cfg)()

        self.camera_names = list(camera_sensor_cfg.base_link_frame.keys())
        self.base_link_names = list(camera_sensor_cfg.base_link_frame.values())
        self.num_cameras = len(self.camera_names)

        robot = env.simulator._robot
        self.camera_sensors_ray_cast_body_indices = torch.tensor(
            robot.find_bodies(list(camera_sensor_cfg.ray_cast_bodies.keys()), preserve_order=True)[0],
            dtype=torch.long,
            device=env.device,
        )
        self.camera_sensors_base_link_indices = robot.find_bodies(
            self.base_link_names, preserve_order=True
        )[0]

        terrain_mesh_vertices = params.get("terrain_mesh_vertices")
        terrain_mesh_faces = params.get("terrain_mesh_faces")
        if terrain_mesh_vertices is None or terrain_mesh_faces is None:
            raise ValueError(
                "WarpDepthImageObsTerm requires terrain_mesh_vertices/faces in params. "
                "These are populated by apply_terrain_preprocess; check that the "
                "preset uses a terrain preprocess hook and that the depth_camera "
                "obs group is named so the hook can find it."
            )
        # apply_terrain_preprocess may hand us either raw arrays (legacy) or a
        # single .npz path stashed in both slots (current); handle both.
        if isinstance(terrain_mesh_vertices, str) and isinstance(terrain_mesh_faces, str) and terrain_mesh_vertices == terrain_mesh_faces:
            loaded = np.load(terrain_mesh_vertices)
            terrain_mesh_vertices = loaded["vertices"]
            terrain_mesh_faces = loaded["faces"]
        terrain_mesh = trimesh.Trimesh(
            vertices=terrain_mesh_vertices, faces=terrain_mesh_faces
        )

        self.camera_sensor = CameraSensor(
            self.num_envs,
            camera_sensor_cfg,
            terrain_mesh,
            device=env.device,
        )
        self.clipping_range = (camera_sensor_cfg.min_range, camera_sensor_cfg.max_range)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(0, self.num_envs, device=self.device)
        self.reset_episodes[:] = False
        self.reset_episodes[env_ids] = True

    def _ensure_command_resolved(self, env: Any) -> None:
        if self.command is not None:
            return
        self.command = env.command_manager.get_state(self._command_name)
        if self.command is None:
            raise ValueError(
                f"WarpDepthImageObsTerm requires a stateful command "
                f"'{self._command_name}'; got None from command_manager.get_state()."
            )
        self._prev_time_step = self.command.time_steps.clone()

    def __call__(self, env: Any, *, modify_history: bool = True, **kwargs) -> torch.Tensor:
        if not modify_history:
            return self._cached_output.to(env.device)

        self._ensure_command_resolved(env)
        depth_images = self._get_depth_images(env)
        processed_images = self._process_depth_images(depth_images, env_ids=slice(None))
        self.depth_buffer[:, :-1] = self.depth_buffer[:, 1:].clone()
        self.depth_buffer[:, -1] = processed_images.to(self.device)

        # Detect motion-end resets (timestep went backwards) and hard-reset episodes.
        time_step_now = self.command.time_steps
        motion_end_reset = time_step_now < self._prev_time_step
        self._prev_time_step = time_step_now.clone()
        reset_idxs = motion_end_reset | self.reset_episodes
        if reset_idxs.any():
            # Fill every buffer slot of a resetting env with the current frame.
            # The source is expanded to the destination shape explicitly rather
            # than relying on broadcast: under
            # ``torch.use_deterministic_algorithms(True)`` boolean-mask
            # assignment requires the source to already match the masked
            # destination, and a (n, 1, H, W) source against an (n, L, H, W)
            # destination raises instead of broadcasting along L.
            frame = processed_images.to(self.device)[reset_idxs].unsqueeze(1)
            self.depth_buffer[reset_idxs] = frame.expand(-1, self.buffer_len, -1, -1)
        self.reset_episodes[:] = False

        if self.latency_frame_range is not None:
            current_latency = torch.randint(
                self.latency_frame_range[0],
                self.latency_frame_range[1] + 1,
                (self.num_envs,),
                device=self.device,
            )
            env_indices = torch.arange(self.num_envs, device=self.device)
            buffer_indices = self.buffer_len - 1 - current_latency
            output = self.depth_buffer[env_indices, buffer_indices]
        else:
            output = self.depth_buffer[:, -1 - self.latency_frame]
        self._cached_output = output.clone()
        return self._cached_output.to(env.device)

    def _get_depth_images(self, env: Any) -> torch.Tensor:
        robot = env.simulator._robot
        if self.camera_sensor.is_dyna_mesh:
            ray_cast_body_poses = [
                robot.data.body_pos_w[:, self.camera_sensors_ray_cast_body_indices]
            ]
            # holosoma stores quats as wxyz; warp sensor wants xyzw.
            ray_cast_body_quats = [
                robot.data.body_quat_w[:, self.camera_sensors_ray_cast_body_indices][..., [1, 2, 3, 0]]
            ]
            ray_cast_body_poses_tensor = torch.cat(ray_cast_body_poses, dim=1)
            ray_cast_body_quats_tensor = torch.cat(ray_cast_body_quats, dim=1)
            self.camera_sensor.ray_cast_body_poses_tensor[:, :] = ray_cast_body_poses_tensor
            self.camera_sensor.ray_cast_body_quats_tensor[:, :] = ray_cast_body_quats_tensor

        camera_base_link_pos = robot.data.body_pos_w[:, self.camera_sensors_base_link_indices]
        camera_base_link_quat = robot.data.body_quat_w[:, self.camera_sensors_base_link_indices][..., [1, 2, 3, 0]]
        self.camera_sensor.camera_sensor_position[:] = tf_apply_xyzw(
            camera_base_link_quat,
            camera_base_link_pos,
            self.camera_sensor.camera_sensor_local_position,
        )
        self.camera_sensor.camera_sensor_orientation[:] = quat_mul_xyzw(
            camera_base_link_quat,
            quat_mul_xyzw(
                self.camera_sensor.camera_sensor_local_orientation,
                self.camera_sensor.camera_sensor_data_frame_quat,
            ),
        )

        depth_images = self.camera_sensor.capture()  # (num_cameras, num_envs, H, W)
        depth_images = torch.clamp(
            depth_images, min=self.clipping_range[0], max=self.clipping_range[1]
        )
        return depth_images[:, 0]  # first (only) camera

    def _process_depth_images(self, depth_images: torch.Tensor, env_ids=None) -> torch.Tensor:
        depth_images = self._crop_depth_images(depth_images)
        depth_images = F.interpolate(
            depth_images.unsqueeze(1),
            size=self._resize_hw,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(1)

        device = depth_images.device
        dtype = depth_images.dtype

        def make_tensor(data):
            return torch.tensor(data, device=device, dtype=dtype)

        max_depth = self.clipping_range[1]
        N, H, W = depth_images.shape
        depth_images = torch.clamp(depth_images, max=max_depth)
        depth_images[depth_images < 0.15] = max_depth

        # Sobel edge detection → stochastic shuffle / empty on edge pixels.
        img = depth_images.unsqueeze(1)
        sobel_x = make_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).unsqueeze(0).unsqueeze(0)
        sobel_y = make_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).unsqueeze(0).unsqueeze(0)

        gx = F.conv2d(img, sobel_x, padding=1)
        gy = F.conv2d(img, sobel_y, padding=1)
        grad_mag = torch.sqrt(gx * gx + gy * gy).squeeze(1)

        edge_thr = 1.0
        edge_mask = grad_mag > edge_thr
        edge_mask[:, :, :3] = False
        edge_mask[:, :, -3:] = False
        edge_mask[:, :3, :] = False
        edge_mask[:, -3:, :] = False

        r = torch.rand_like(edge_mask.float())
        shuffle_mask = edge_mask & (r < 0.9)
        shuffle_mask = F.max_pool2d(
            shuffle_mask.unsqueeze(1).float(), kernel_size=3, stride=1, padding=1
        ).squeeze(1).bool()

        for ii in range(1):
            x_pad = F.pad(depth_images.unsqueeze(1), (1, 1, 1, 1), mode="circular")
            patches = F.unfold(x_pad, kernel_size=3, padding=0, stride=1)
            patches = patches.view(N, 9, H, W)
            if ii == 0:
                idx = torch.randint(0, 9, (N, 1, H, W), device=device)
                gathered = torch.gather(patches, 1, idx).squeeze(1)
            else:
                gathered = torch.max(patches, dim=1).values
            depth_images = torch.where(shuffle_mask, gathered, depth_images)

        edge_thr = 0.6
        edge_mask = (grad_mag > edge_thr) & (
            F.max_pool2d(depth_images.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1) > 2.5
        )
        edge_mask[:, :, :3] = False
        edge_mask[:, :, -3:] = False
        edge_mask[:, :3, :] = False
        edge_mask[:, -3:, :] = False
        set_empty_mask = edge_mask & (r < 0.7)
        depth_images[set_empty_mask] = max_depth

        if self.enable_holes:
            frame = self.generator.generate_frame()
            if env_ids is not None and not isinstance(env_ids, slice):
                frame = frame[env_ids]
            frame = F.interpolate(frame.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)
            frame = F.max_pool2d(frame, kernel_size=3, stride=1, padding=1)
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
            frame = (frame.squeeze(1) < 0.2).bool() & (depth_images < 2.0).bool() & (depth_images > 0.2).bool()
            depth_images[frame] = max_depth

        depth_images += torch.randn_like(depth_images) * 0.03
        depth_images += self.depth_offset[:, None, None]

        depth_images = self._normalize_depth_images(depth_images)
        return depth_images

    def _crop_depth_images(self, depth_images: torch.Tensor) -> torch.Tensor:
        # 2 rows from top, 4 cols from each side (matches far-tracking convention).
        return depth_images[:, 2:, 4:-4]

    def _normalize_depth_images(self, depth_images: torch.Tensor) -> torch.Tensor:
        lo, hi = self.clipping_range
        return (depth_images - lo) / (hi - lo) - 0.5
