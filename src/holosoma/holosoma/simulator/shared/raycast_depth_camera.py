"""Ray-cast depth camera, rendered with Warp for ``CameraSensorConfig.isaacsim.renderer="raycast"``.

The camera casts one ray per pixel against the terrain mesh and against the meshes of selected
robot links at their current poses (self-occlusion). It reads only backend-neutral simulator
state -- ``rigid_body_pos_w``, ``rigid_body_quat_w`` (xyzw), ``find_rigid_body_indice`` and the
terrain term's ``warp_mesh``.

The kernel is adapted from Aerial Gym Simulator's ``DepthCameraWarpKernels`` (BSD-3-Clause; see
``THIRD_PARTY_LICENSES``), as used by the PHP release depth camera: rays pass through integer
pixel coordinates of a pinhole camera, and each ray keeps the nearest robot-mesh hit unless the
terrain is closer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh
import warp as wp

from holosoma.config_types.sensor import CameraSensorConfig
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import quat_apply, quat_from_euler_xyz, quat_mul
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.warp_utils import convert_to_wp_mesh

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.shared.sensor_manager import CameraRecord

# Rotates the shared optical frame (-Z forward, +Y up) into the pinhole frame the kernel
# casts in (+Z forward, +Y down): a half turn about X, as an xyzw quaternion.
_OPTICAL_TO_PINHOLE_XYZW = (1.0, 0.0, 0.0, 0.0)

_NO_HIT = wp.constant(math.inf)


@wp.kernel
def _render_depth_kernel(
    env_ids: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    terrain_mesh: wp.uint64,
    body_meshes: wp.array(dtype=wp.uint64),  # type: ignore[valid-type]
    body_pos: wp.array(dtype=wp.vec3, ndim=2),  # type: ignore[valid-type]
    body_quat: wp.array(dtype=wp.quat, ndim=2),  # type: ignore[valid-type]
    num_bodies: int,
    cam_pos: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    cam_quat: wp.array(dtype=wp.quat),  # type: ignore[valid-type]
    k_inv: wp.mat44,
    far: float,
    depth: wp.array(dtype=wp.float32, ndim=3),  # type: ignore[valid-type]
) -> None:
    # wp.tid() returns a tuple for a 3D launch; Warp's stubs type it as an int.
    i, y, x = wp.tid()  # type: ignore[misc]
    env = env_ids[i]
    origin = cam_pos[env]
    rotation = cam_quat[env]
    pixel = wp.vec3(float(x), float(y), 1.0)
    ray = wp.normalize(wp.quat_rotate(rotation, wp.transform_vector(k_inv, pixel)))
    # K^-1 maps the principal point (width / 2, height / 2) to +Z, so the principal ray is the
    # camera's +Z axis for any image size.
    principal = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
    # Distance along the ray times this cosine is the distance to the image plane.
    cosine = wp.min(wp.max(wp.dot(ray, principal), wp.float32(1.0e-6)), wp.float32(1.0))

    best = wp.float32(_NO_HIT)
    bound = wp.float32(far)
    # mesh_query_ray output arguments, typed as the Warp builtin expects.
    distance = wp.float32(0.0)
    u = wp.float32(0.0)
    v = wp.float32(0.0)
    sign = wp.float32(0.0)
    normal = wp.vec3()
    face = wp.int32(0)
    for b in range(num_bodies):
        q = body_quat[env, b]
        local_origin = wp.quat_rotate_inv(q, origin - body_pos[env, b])
        local_ray = wp.quat_rotate_inv(q, ray)
        if wp.mesh_query_ray(  # type: ignore[call-arg]
            body_meshes[b],
            wp.vec3(local_origin),
            wp.vec3(local_ray),
            bound / cosine,
            distance,
            u,
            v,
            sign,
            normal,
            face,
        ):
            hit = cosine * distance
            if best == _NO_HIT or hit < best:
                best = hit
                bound = hit
    if wp.mesh_query_ray(  # type: ignore[call-arg]
        terrain_mesh,
        wp.vec3(origin),
        wp.vec3(ray),
        bound / cosine,
        distance,
        u,
        v,
        sign,
        normal,
        face,
    ):
        best = cosine * distance
    depth[env, y, x] = best


class RayCastDepthCamera:
    """Renders one ray-cast :class:`CameraSensorConfig` camera into its sensor record.

    Simulator state it needs (body indices, terrain mesh) is resolved on the first render, when
    every backend has finished building its scene. Per-env mount offsets come from a
    randomization setup term through :meth:`set_mount_offset`.
    """

    def __init__(self, sim: BaseSimulator, config: CameraSensorConfig) -> None:
        self.sim = sim
        self.config = config
        self.device = str(sim.sim_device)
        self._ready = False
        self._position_offset: torch.Tensor | None = None
        self._euler_offset: torch.Tensor | None = None

    def set_mount_offset(self, position: torch.Tensor, euler: torch.Tensor) -> None:
        """Perturb each env's mount by ``position`` ``[N, 3]`` (meters) and XYZ Euler ``euler``
        ``[N, 3]`` (radians), in the mount-link frame: the position offset is added to
        ``mount.position`` and the rotation is applied about the link axes before
        ``mount.orientation``."""
        expected = (self.sim.num_envs, 3)
        if tuple(position.shape) != expected or tuple(euler.shape) != expected:
            raise ValueError(
                f"Mount offsets must be {list(expected)}, got {list(position.shape)} and {list(euler.shape)}."
            )
        self._position_offset = position.to(device=self.device, dtype=torch.float32)
        self._euler_offset = euler.to(device=self.device, dtype=torch.float32)
        if self._ready:
            self._set_mount_pose()

    # ------------------------------------------------------------------ setup

    def _setup(self) -> None:
        sim, cfg = self.sim, self.config
        num_envs = sim.num_envs
        wp.init()
        self._wp_device = wp.get_device(self.device)

        self._mount_body = sim.find_rigid_body_indice(cfg.mount.target)
        # One entry per mesh; a link with several meshes appears once for each.
        occluders = [
            (link, path)
            for link, paths in cfg.isaacsim.raycast_link_meshes.items()
            for path in ([paths] if isinstance(paths, str) else paths)
            if path
        ]
        self._body_ids = [sim.find_rigid_body_indice(link) for link, _ in occluders]
        self._body_meshes = [self._load_mesh(path) for _, path in occluders]
        # Warp kernels need a non-empty array even when no link occludes the camera.
        self._body_mesh_ids: wp.array[Any] = wp.array(
            [mesh.id for mesh in self._body_meshes] or [0], dtype=wp.uint64, device=self._wp_device
        )
        terrain = sim.terrain_manager.get_state("locomotion_terrain")
        terrain_mesh = getattr(terrain, "warp_mesh", None)
        if terrain_mesh is None:
            raise RuntimeError(
                f"RayCastDepthCamera needs a terrain term with a 'warp_mesh'; got {type(terrain).__name__}."
            )
        self._terrain_mesh = terrain_mesh

        # Square pixels; the principal point is the image center, as in the PHP release camera.
        focal = (cfg.height / 2) / math.tan(math.radians(cfg.vertical_fov) / 2)
        intrinsics = np.array(
            [[focal, 0.0, cfg.width / 2, 0.0], [0.0, focal, cfg.height / 2, 0.0], [0.0, 0.0, 1.0, 0.0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self._k_inv = wp.mat44(*np.linalg.inv(intrinsics).astype(np.float32).flatten().tolist())

        self._set_mount_pose()

        # Persistent buffers the kernel reads and writes.
        dev = self.device
        num_bodies = len(self._body_ids)
        self._cam_pos = torch.zeros(num_envs, 3, device=dev)
        self._cam_quat = torch.zeros(num_envs, 4, device=dev)
        self._body_pos = torch.zeros(num_envs, max(num_bodies, 1), 3, device=dev)
        self._body_quat = torch.zeros(num_envs, max(num_bodies, 1), 4, device=dev)
        self._body_quat[..., 3] = 1.0
        self._image = torch.zeros(num_envs, cfg.height, cfg.width, device=dev)
        self._all_envs = torch.arange(num_envs, dtype=torch.int32, device=dev)
        self._ready = True

    def _set_mount_pose(self) -> None:
        """Each env's camera pose in its mount link's frame, with any offsets applied."""
        cfg, dev, num_envs = self.config, self.device, self.sim.num_envs
        position = torch.tensor(cfg.mount.position, dtype=torch.float32, device=dev).expand(num_envs, 3)
        w, x, y, z = cfg.mount.orientation
        mount_quat = torch.tensor([x, y, z, w], dtype=torch.float32, device=dev).expand(num_envs, 4).contiguous()
        if self._position_offset is not None and self._euler_offset is not None:
            position = position + self._position_offset
            euler = self._euler_offset
            mount_quat = quat_mul(quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]), mount_quat, w_last=True)
        to_pinhole = torch.tensor(_OPTICAL_TO_PINHOLE_XYZW, dtype=torch.float32, device=dev).expand(num_envs, 4)
        self._local_position = position.clone()
        self._local_quat = quat_mul(mount_quat, to_pinhole.contiguous(), w_last=True)

    def _load_mesh(self, path: str) -> Any:
        mesh = trimesh.load_mesh(resolve_data_file_path(path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Link mesh {path!r} is not a triangle mesh ({type(mesh).__name__}).")
        return convert_to_wp_mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), self.device)

    # ----------------------------------------------------------------- render

    def _update_poses(self) -> None:
        link_pos = self.sim.rigid_body_pos_w[:, self._mount_body]
        link_quat = self.sim.rigid_body_quat_w[:, self._mount_body]
        self._cam_pos[:] = link_pos + quat_apply(link_quat, self._local_position, w_last=True)
        self._cam_quat[:] = quat_mul(link_quat.contiguous(), self._local_quat, w_last=True)
        if self._body_ids:
            self._body_pos[:] = self.sim.rigid_body_pos_w[:, self._body_ids]
            self._body_quat[:] = self.sim.rigid_body_quat_w[:, self._body_ids]

    def render(self, record: CameraRecord, env_ids: torch.Tensor | None = None) -> None:
        """Render all envs, or only ``env_ids``, into ``record``'s ``"depth"`` buffer."""
        if not self._ready:
            self._setup()
        self._update_poses()
        ids = self._all_envs if env_ids is None else env_ids.to(device=self.device, dtype=torch.int32)
        if ids.numel() == 0:
            return
        cfg = self.config
        # On torch's current stream, so the launch is ordered after the pose updates above and
        # before the reads below without a host sync, whichever stream torch is using.
        stream = wp.stream_from_torch(self.device) if self._wp_device.is_cuda else None
        wp.launch(
            _render_depth_kernel,
            dim=(ids.numel(), cfg.height, cfg.width),
            inputs=[
                wp.from_torch(ids.contiguous(), dtype=wp.int32),
                self._terrain_mesh.id,
                self._body_mesh_ids,
                wp.from_torch(self._body_pos, dtype=wp.vec3),
                wp.from_torch(self._body_quat, dtype=wp.quat),
                len(self._body_ids),
                wp.from_torch(self._cam_pos, dtype=wp.vec3),
                wp.from_torch(self._cam_quat, dtype=wp.quat),
                self._k_inv,
                float(cfg.far),
                wp.from_torch(self._image, dtype=wp.float32),
            ],
            device=self._wp_device,
            stream=stream,
        )
        if env_ids is None or "depth" not in record.buffers:
            record.set_buffer("depth", self._image.unsqueeze(-1))
        else:
            long_ids = ids.long()
            record.set_buffer_rows("depth", long_ids, self._image[long_ids].unsqueeze(-1))
