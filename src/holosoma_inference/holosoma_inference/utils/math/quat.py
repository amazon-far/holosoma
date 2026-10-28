# Quaternion operations

import numpy as np


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0
    return a - b + c


def rpy_to_quat(rpy):
    """
    Convert roll, pitch, yaw (in radians) to quaternion [w, x, y, z]
    Follows ZYX rotation order (yaw → pitch → roll)
    """
    roll, pitch, yaw = rpy
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quat_to_rpy(q):
    """
    Convert quaternion to roll, pitch, yaw (ZYX order).
    Input: q = [w, x, y, z]
    Output: roll, pitch, yaw (in radians)
    """
    w, x, y, z = q

    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quat_inverse(q01):
    "q01: (1, 4), wxyz"
    # return np.concatenate((-q01[:, :3], q01[:, -1:]), axis=1)
    return np.concatenate((q01[:, 0:1], -q01[:, 1:]), axis=1)


def quat_mul(a, b):
    "a: (1, 4), b: (1, 4), wxyz"
    assert a.shape == b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

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

    return np.stack([w, x, y, z]).reshape(a.shape)


def quat_apply(a, b):  # a: (1, 4), b: (1, 3)
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:]
    w = a[:, :1]
    t = np.cross(xyz, b) * 2
    return b + w * t + np.cross(xyz, t)


def subtract_frame_transforms(q01, q02):
    q10 = quat_inverse(q01)
    return quat_mul(q10, q02)


def matrix_from_quat(quaternions):
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = np.stack(
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


def xyzw_to_wxyz(xyzw):
    return np.concatenate([xyzw[:, -1:], xyzw[:, :3]], axis=1)


def wxyz_to_xyzw(wxyz):
    return np.concatenate([wxyz[:, 1:], wxyz[:, :1]], axis=1)
