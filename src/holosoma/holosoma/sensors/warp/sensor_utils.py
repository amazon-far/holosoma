from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import warp as wp


@torch.jit.script
def quat_apply_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by xyzw quaternions."""
    shape = vector.shape
    quaternion = quaternion.reshape(-1, 4)
    vector = vector.reshape(-1, 3)
    xyz = quaternion[:, :3]
    cross = xyz.cross(vector, dim=-1) * 2
    return (vector + quaternion[:, 3:] * cross + xyz.cross(cross, dim=-1)).view(shape)


@torch.jit.script
def tf_apply_xyzw(quaternion: torch.Tensor, translation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply an xyzw quaternion and translation to vectors."""
    return quat_apply_xyzw(quaternion, vector) + translation


def quat_mul_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply equally shaped xyzw quaternion tensors."""
    if left.shape != right.shape:
        raise ValueError(f"Quaternion shapes must match, got {left.shape} and {right.shape}.")
    shape = left.shape
    left = left.reshape(-1, 4)
    right = right.reshape(-1, 4)

    x1, y1, z1, w1 = left[:, 0], left[:, 1], left[:, 2], left[:, 3]
    x2, y2, z2, w2 = right[:, 0], right[:, 1], right[:, 2], right[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack((x, y, z, w), dim=-1).view(shape)


@torch.jit.script
def torch_rand_float_tensor(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Sample uniformly between element-wise tensor bounds."""
    return (upper - lower) * torch.rand_like(upper) + lower


@torch.jit.script
def quat_from_euler_xyz_tensor(euler_xyz: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles to xyzw quaternions."""
    roll = euler_xyz[..., 0]
    pitch = euler_xyz[..., 1]
    yaw = euler_xyz[..., 2]
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack((qx, qy, qz, qw), dim=-1)


def convert_to_warp_mesh(
    points: npt.NDArray[np.floating[Any]], indices: npt.NDArray[np.integer[Any]], device: str
) -> wp.Mesh:
    """Create a Warp triangle mesh from NumPy vertices and face indices."""
    return wp.Mesh(
        points=wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(np.asarray(indices, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device),
    )
