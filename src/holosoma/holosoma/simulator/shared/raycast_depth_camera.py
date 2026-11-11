"""Ray-cast depth camera, rendered with Warp for ``CameraSensorConfig.isaacsim.renderer="raycast"``.

The camera casts one ray per pixel against meshes supplied by the loaded scene. Mesh selection,
conversion, caching, and transform tracking belong to the scene provider; this renderer only
consumes persistent mesh IDs and poses.

The kernel is adapted from Aerial Gym Simulator's ``DepthCameraWarpKernels`` (BSD-3-Clause; see
``THIRD_PARTY_LICENSES``): rays pass through pixel centres of a pinhole camera, as in the native
renderers, and each ray keeps the nearest scene-mesh hit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import warp as wp

from holosoma.config_types.sensor import CameraSensorConfig
from holosoma.utils.rotations import quat_apply, quat_from_euler_xyz, quat_mul
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.shared.sensor_manager import CameraRecord


@dataclass(frozen=True)
class RaycastSceneMeshes:
    """Persistent per-environment mesh IDs and world poses, owned by the scene provider.

    IDs and poses have shape ``[num_envs, num_meshes]``; positions are Warp vec3 and orientations
    are Warp quaternions (xyzw). The provider retains the meshes and updates poses in place.
    """

    mesh_ids: wp.array[Any]
    positions: wp.array[Any]
    orientations: wp.array[Any]
    num_meshes: int


class RaycastScene(Protocol):
    """Scene geometry initialized after assets load; pose updates do not perform ray queries."""

    def prepare_meshes(self) -> RaycastSceneMeshes: ...

    def update_mesh_poses(self) -> None: ...


# Rotates the shared optical frame (-Z forward, +Y up) into the pinhole frame the kernel
# casts in (+Z forward, +Y down): a half turn about X, as an xyzw quaternion.
_OPTICAL_TO_PINHOLE_XYZW = (1.0, 0.0, 0.0, 0.0)

_NO_HIT = wp.constant(math.inf)


@wp.kernel(enable_backward=False)
def _render_depth_kernel(
    env_ids: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    mesh_ids: wp.array(dtype=wp.uint64, ndim=2),  # type: ignore[valid-type]
    mesh_pos: wp.array(dtype=wp.vec3, ndim=2),  # type: ignore[valid-type]
    mesh_quat: wp.array(dtype=wp.quat, ndim=2),  # type: ignore[valid-type]
    num_meshes: int,
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
    pixel = wp.vec3(float(x) + 0.5, float(y) + 0.5, 1.0)
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
    for b in range(num_meshes):
        q = mesh_quat[env, b]
        local_origin = wp.quat_rotate_inv(q, origin - mesh_pos[env, b])
        local_ray = wp.quat_rotate_inv(q, ray)
        if wp.mesh_query_ray(  # type: ignore[call-arg]
            mesh_ids[env, b],
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
            # The query is bounded by the nearest hit so far, so any hit is nearer.
            best = cosine * distance
            bound = best
    depth[env, y, x] = best


class RayCastDepthCamera:
    """Renders one ray-cast :class:`CameraSensorConfig` camera into its sensor record.

    :meth:`initialize` binds loaded scene geometry after the simulator's load phases. Per-env mount
    offsets come from a randomization setup term through :meth:`set_mount_offset`.
    """

    def __init__(self, sim: BaseSimulator, config: CameraSensorConfig, mesh_provider: RaycastScene) -> None:
        self.sim = sim
        self.config = config
        self.device = sim.sim_device
        self.mesh_provider = mesh_provider
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

    def initialize(self) -> None:
        """Bind the provider's scene meshes and allocate persistent rendering buffers once."""
        if self._ready:
            return
        sim, cfg = self.sim, self.config
        num_envs = sim.num_envs
        wp.init()
        self._wp_device = wp.get_device(self.device)

        if cfg.mount.target not in sim.body_names:
            raise ValueError(
                f"Ray-cast camera mount link '{cfg.mount.target}' is not in the robot config's body_names "
                f"{sorted(sim.body_names)}; ray-cast cameras follow a configured robot body."
            )
        self._mount_body = sim.find_rigid_body_indice(cfg.mount.target)
        self._meshes = self.mesh_provider.prepare_meshes()

        # Square pixels; the principal point is the image center.
        focal = (cfg.height / 2) / math.tan(math.radians(cfg.vertical_fov) / 2)
        intrinsics = np.array(
            [[focal, 0.0, cfg.width / 2, 0.0], [0.0, focal, cfg.height / 2, 0.0], [0.0, 0.0, 1.0, 0.0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self._k_inv = wp.mat44(*np.linalg.inv(intrinsics).astype(np.float32).flatten().tolist())

        self._set_mount_pose()

        # Persistent buffers the kernel reads and writes.
        dev = self.device
        self._cam_pos = torch.zeros(num_envs, 3, device=dev)
        self._cam_quat = torch.zeros(num_envs, 4, device=dev)
        self._image = torch.zeros(num_envs, cfg.height, cfg.width, device=dev)
        self._all_envs = torch.arange(num_envs, dtype=torch.int32, device=dev)
        # Warp views sharing the buffers' memory; the buffers are only ever written in place.
        self._wp_cam_pos = wp.from_torch(self._cam_pos, dtype=wp.vec3)
        self._wp_cam_quat = wp.from_torch(self._cam_quat, dtype=wp.quat)
        self._wp_image = wp.from_torch(self._image, dtype=wp.float32)
        self._wp_all_envs = wp.from_torch(self._all_envs, dtype=wp.int32)
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

    # ----------------------------------------------------------------- render

    def _update_poses(self) -> None:
        link_pos = self.sim.rigid_body_pos_w[:, self._mount_body]
        link_quat = self.sim.rigid_body_quat_w[:, self._mount_body]
        self._cam_pos[:] = link_pos + quat_apply(link_quat, self._local_position, w_last=True)
        self._cam_quat[:] = quat_mul(link_quat.contiguous(), self._local_quat, w_last=True)
        self.mesh_provider.update_mesh_poses()

    def render(self, record: CameraRecord, env_ids: torch.Tensor | None = None) -> None:
        """Render all envs, or only ``env_ids``, into ``record``'s ``"depth"`` buffer. The first
        render covers every env."""
        if not self._ready:
            raise RuntimeError("RayCastDepthCamera must be initialized after scene loading and before rendering.")
        if "depth" not in record.buffers:
            env_ids = None
        ids = self._all_envs if env_ids is None else env_ids.to(device=self.device, dtype=torch.int32).contiguous()
        if ids.numel() == 0:
            return
        self._update_poses()
        wp_ids = self._wp_all_envs if env_ids is None else wp.from_torch(ids, dtype=wp.int32)
        cfg = self.config
        # On torch's current stream, so the launch is ordered after the pose updates above and
        # before the reads below without a host sync, whichever stream torch is using.
        stream = wp.stream_from_torch(self.device) if self._wp_device.is_cuda else None
        wp.launch(
            _render_depth_kernel,
            dim=(ids.numel(), cfg.height, cfg.width),
            inputs=[
                wp_ids,
                self._meshes.mesh_ids,
                self._meshes.positions,
                self._meshes.orientations,
                self._meshes.num_meshes,
                self._wp_cam_pos,
                self._wp_cam_quat,
                self._k_inv,
                float(cfg.far),
                self._wp_image,
            ],
            device=self._wp_device,
            stream=stream,
        )
        if env_ids is None:
            record.set_buffer("depth", self._image.unsqueeze(-1))
        else:
            long_ids = ids.long()
            record.set_buffer_rows("depth", long_ids, self._image[long_ids].unsqueeze(-1))
