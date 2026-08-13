from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as torch_nn_func
from torch import Tensor

from holosoma.utils.torch_jit import torch_jit_script
from holosoma.utils.torch_utils import (
    copysign,
    normalize,
)


@torch_jit_script
def quat_unit(a):
    return normalize(a)


@torch_jit_script
def yaw_quat(quat: torch.Tensor, w_last: bool) -> torch.Tensor:
    shape = quat.shape
    quat_yaw = quat.view(-1, 4)
    if w_last:
        qx = quat_yaw[:, 0]
        qy = quat_yaw[:, 1]
        qz = quat_yaw[:, 2]
        qw = quat_yaw[:, 3]
    else:
        qw = quat_yaw[:, 0]
        qx = quat_yaw[:, 1]
        qy = quat_yaw[:, 2]
        qz = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = torch.zeros_like(quat_yaw)
    if w_last:
        quat_yaw[:, 2] = torch.sin(yaw / 2)
        quat_yaw[:, 3] = torch.cos(yaw / 2)
    else:
        quat_yaw[:, 3] = torch.sin(yaw / 2)
        quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.view(shape)


@torch_jit_script
def quat_apply(a: Tensor, b: Tensor, w_last: bool) -> Tensor:
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    if w_last:
        xyz = a[:, :3]
        w = a[:, 3:]
    else:
        xyz = a[:, 1:]
        w = a[:, :1]
    t = xyz.cross(b, dim=-1) * 2
    return (b + w * t + xyz.cross(t, dim=-1)).view(shape)


@torch_jit_script
def quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor, w_last: bool) -> torch.Tensor:
    quat_yaw = yaw_quat(quat, w_last)
    return quat_apply(quat_yaw, vec, w_last)


@torch_jit_script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


@torch_jit_script
def quat_conjugate(a: Tensor, w_last: bool) -> Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    if w_last:
        return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)
    return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


@torch_jit_script
def quat_rotate(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch_jit_script
def quat_rotate_inverse(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch_jit_script
def quat_angle_axis(x: Tensor, w_last: bool) -> tuple[Tensor, Tensor]:
    """
    The (angle, axis) representation of the rotation. The axis is normalized to unit length.
    The angle is guaranteed to be between [0, pi].
    """
    if w_last:
        w = x[..., -1]
        axis = x[..., :3]
    else:
        w = x[..., 0]
        axis = x[..., 1:]
    s = 2 * (w**2) - 1
    angle = s.clamp(-1, 1).arccos()  # just to be safe
    axis /= axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
    return angle, axis


@torch_jit_script
def quat_from_angle_axis(angle: Tensor, axis: Tensor, w_last: bool) -> Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    if w_last:
        return quat_unit(torch.cat([xyz, w], dim=-1))
    return quat_unit(torch.cat([w, xyz], dim=-1))


@torch_jit_script
def vec_to_heading(h_vec):
    return torch.atan2(h_vec[..., 1], h_vec[..., 0])


@torch_jit_script
def heading_to_quat(h_theta, w_last: bool):
    axis = torch.zeros(
        h_theta.shape
        + [
            3,
        ],
        device=h_theta.device,
    )
    axis[..., 2] = 1
    return quat_from_angle_axis(h_theta, axis, w_last=w_last)


@torch_jit_script
def quat_axis(q: Tensor, axis: int, w_last: bool) -> Tensor:
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec, w_last)


@torch_jit_script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch_jit_script
def get_basis_vector(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    return quat_rotate(q, v, w_last)


@torch_jit_script
def quat_to_angle_axis(quat: torch.Tensor, eps: float = 1.0e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (x, y, z, w). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554

        Migrated from axis_angle_from_quat in IssacLab.
    """

    # Quaternion is [q_x, q_y, q_z, q_w] = [n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2), cos(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48

    quat = quat * (1.0 - 2.0 * (quat[..., 3:4] < 0.0))
    mag = torch.linalg.norm(quat[..., :3], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 3])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return angle, quat[..., 0:3] / sin_half_angles_over_angles.unsqueeze(-1)


@torch_jit_script
def slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    return torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)


@torch_jit_script
def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    return angle_expand * axis


@torch_jit_script
def my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch_jit_script
def calc_heading(q):
    # type: (Tensor) -> Tensor
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    # this is the x axis heading
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = my_quat_rotate(q, ref_dir)

    return torch.atan2(rot_dir[..., 1], rot_dir[..., 0])


@torch_jit_script
def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    return angle_axis_to_exp_map(angle, axis)


@torch_jit_script
def calc_heading_quat(q, w_last):
    # type: (Tensor, bool) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    return quat_from_angle_axis(heading, axis, w_last=w_last)


@torch_jit_script
def calc_heading_quat_inv(q, w_last):
    # type: (Tensor, bool) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    return quat_from_angle_axis(-heading, axis, w_last=w_last)


@torch_jit_script
def quat_inverse(x, w_last):
    # type: (Tensor, bool) -> Tensor
    """
    The inverse of the rotation
    """
    return quat_conjugate(x, w_last=w_last)


@torch_jit_script
def get_euler_xyz(q: Tensor, w_last: bool) -> tuple[Tensor, Tensor, Tensor]:
    if w_last:
        qx, qy, qz, qw = 0, 1, 2, 3
    else:
        qw, qx, qy, qz = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


@torch_jit_script
def get_euler_xyz_in_tensor(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)


@torch_jit_script
def quat_pos(x):
    """
    make all the real part of the quaternion positive
    """
    q = x
    z = (q[..., 3:] < 0).float()
    return (1 - 2 * z) * q


@torch_jit_script
def is_valid_quat(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return (w * w + x * x + y * y + z * z).allclose(torch.ones_like(w))


@torch_jit_script
def quat_normalize(q):
    """
    Construct 3D rotation from quaternion (the quaternion needs not to be normalized).
    """
    return quat_unit(quat_pos(q))  # normalized to positive and unit quaternion


@torch_jit_script
def quat_mul(a, b, w_last: bool):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    else:
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    if w_last:
        quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


@torch_jit_script
def quat_mul_norm(x, y, w_last):
    # type: (Tensor, Tensor, bool) -> Tensor
    r"""
    Combine two set of 3D rotations together using \**\* operator. The shape needs to be
    broadcastable
    """
    return quat_normalize(quat_mul(x, y, w_last))


@torch_jit_script
def quat_identity(shape: list[int]):
    """
    Construct 3D identity rotation given shape
    """
    w = torch.ones(shape + [1])
    xyz = torch.zeros(shape + [3])
    q = torch.cat([xyz, w], dim=-1)
    return quat_normalize(q)


@torch_jit_script
def quat_identity_like(x):
    """
    Construct identity 3D rotation with the same shape
    """
    return quat_identity(x.shape[:-1])


@torch_jit_script
def transform_from_rotation_translation(r: torch.Tensor | None = None, t: torch.Tensor | None = None):
    """
    Construct a transform from a quaternion and 3D translation. Only one of them can be None.
    """
    assert r is not None or t is not None, "rotation and translation can't be all None"
    if r is None:
        assert t is not None
        r = quat_identity(list(t.shape))
    if t is None:
        t = torch.zeros(list(r.shape) + [3])
    return torch.cat([r, t], dim=-1)


@torch_jit_script
def transform_rotation(x):
    """Get rotation from transform"""
    return x[..., :4]


@torch_jit_script
def transform_translation(x):
    """Get translation from transform"""
    return x[..., 4:]


@torch_jit_script
def transform_mul(x, y):
    """
    Combine two transformation together
    """
    return transform_from_rotation_translation(
        r=quat_mul_norm(transform_rotation(x), transform_rotation(y), w_last=True),
        t=quat_rotate(transform_rotation(x), transform_translation(y), w_last=True) + transform_translation(x),
    )


##################################### FROM PHC rotation_conversions.py #####################################
@torch_jit_script
def quaternion_to_matrix(quaternions: torch.Tensor, w_last: bool = False) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if w_last:
        i, j, k, r = torch.unbind(quaternions, -1)
    else:
        # Original codebase assumes the quaternion is in (w, x, y, z) format when calling this function.
        r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


@torch_jit_script
def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)


# Keep '_sqrt_positive_part' before functions that depend on it for JIT
@torch_jit_script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch_jit_script
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    w x y z
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch_nn_func.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))


@torch_jit_script
def quat_from_euler_xyz(roll, pitch, yaw):
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

    return torch.stack([qx, qy, qz, qw], dim=-1)


#########################################################################################################
## Math functions copied from IsaacLab for reproducing WholeBodyTracking,
# modified for quaternion (x, y, z, w) instead of (w, x, y, z)
#########################################################################################################
@torch_jit_script
def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor, w_last: bool = True) -> torch.Tensor:
    """Computes the rotation difference between two quaternions.

    Args:
        q1: The first quaternion in (x, y, z, w). Shape is (..., 4).
        q2: The second quaternion in (x, y, z, w). Shape is (..., 4).

    Returns:
        Angular error between input quaternions in radians.
    """
    quat_diff = quat_mul(q1, quat_conjugate(q2, w_last=w_last), w_last=w_last)
    return quat_to_angle_axis(quat_diff)[0]


@torch_jit_script
def subtract_frame_transforms(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor | None = None,
    q02: torch.Tensor | None = None,
    w_last: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Subtract transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{12} = T_{01}^{-1} \times T_{02}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (x, y, z, w). Shape is (N, 4).
        t02: Position of frame 2 w.r.t. frame 0. Shape is (N, 3).
            Defaults to None, in which case the position is assumed to be zero.
        q02: Quaternion orientation of frame 2 w.r.t. frame 0 in (x, y, z, w). Shape is (N, 4).
            Defaults to None, in which case the orientation is assumed to be identity.
        w_last: Whether the quaternion is in (x, y, z, w) or (w, x, y, z) format.
    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 1.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    # compute orientation
    q10 = quat_inverse(q01, w_last=w_last)
    if q02 is not None:
        q12 = quat_mul(q10, q02, w_last=w_last)
    else:
        q12 = q10
    # compute translation
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01, w_last=w_last)
    else:
        t12 = quat_apply(q10, -t01, w_last=w_last)
    return t12, q12


# ============================================================================
# Custom Batched Quaternion Operations
# These handle special [N, M, 3] vector shapes not supported by standard ops
# ============================================================================


@torch_jit_script
def quat_rotate_inverse_batched(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply inverse quaternion rotation with batched vector support.

    This is a specialized function for handling [N, M, 3] shaped vectors,
    which is common in RL applications with multiple waypoints per environment.

    Args:
        q: Quaternion of shape [N, 4] in (x, y, z, w) order.
        v: Vectors of shape [N, M, 3].

    Returns:
        Rotated vectors of shape [N, M, 3].

    Note:
        For standard quaternion operations, use holosoma.isaac_utils.rotations.
    """
    N, M = v.shape[:2]

    # Repeat each quaternion M times to match v
    q_expanded = q.unsqueeze(1).expand(-1, M, -1).reshape(N * M, 4)  # [N*M, 4]
    v_flat = v.reshape(N * M, 3)  # [N*M, 3]

    # Use rotations.py function with w_last=True for (x,y,z,w) convention
    rotated_flat = quat_rotate_inverse(q_expanded, v_flat, w_last=True)  # [N*M, 3]

    # Reshape back to [N, M, 3]
    return rotated_flat.view(N, M, 3)


@torch_jit_script
def quat_rotate_batched(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation with batched vector support.

    This is a specialized function for handling [N, M, 3] shaped vectors,
    which is common in RL applications with multiple waypoints per environment.

    Args:
        q: Quaternion of shape [N, 4] in (x, y, z, w) order.
        v: Vectors of shape [N, M, 3].

    Returns:
        Rotated vectors of shape [N, M, 3].

    Note:
        For standard quaternion operations, use holosoma.isaac_utils.rotations.
    """
    N, M = v.shape[:2]

    q_vec = q[:, :3].unsqueeze(1).expand(-1, M, -1)  # [N, M, 3]
    q_w = q[:, 3].unsqueeze(1).expand(-1, M)  # [N, M]

    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    dot = (q_vec * v).sum(dim=-1, keepdim=True)
    c = q_vec * dot * 2.0

    return a + b + c
