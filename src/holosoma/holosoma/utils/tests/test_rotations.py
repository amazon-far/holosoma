import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from holosoma.utils.rotations import (
    normalize_angle,
    quat_apply,
    quat_apply_yaw,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate,
    quat_unit,
    slerp,
    yaw_quat,
)


@pytest.fixture
def test_angles():
    return [
        [0.5, 0.3, 1.0],  # random angles
        [0.0, 0.0, 1.0],  # pure yaw
        [0.5, 0.5, 0.0],  # no yaw
        [0.0, 0.0, 0.0],  # zero rotation
    ]


@pytest.fixture
def test_vecs():
    return [
        [1.0, 0.0, 0.0],
        [0.2, 0.3, 0.4],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]


def test_quat_yaw(test_angles):
    for euler in test_angles:
        rot = Rotation.from_euler("xyz", euler, degrees=False)
        quat_scipy = rot.as_quat()
        quat_torch = torch.tensor(quat_scipy).unsqueeze(0).float()
        yaw_quat_torch = yaw_quat(quat_torch, w_last=True)

        # Convert back to numpy and scipy format (xyzw)
        yaw_quat_np = yaw_quat_torch.squeeze().numpy()

        # Convert back to euler angles
        rot_yaw = Rotation.from_quat(yaw_quat_np)
        euler_yaw = rot_yaw.as_euler("xyz", degrees=False)

        assert np.allclose(euler_yaw[2], euler[2], atol=1e-6), f"Test failed: {euler_yaw} != {euler}"
        assert np.allclose(euler_yaw[0], 0.0, atol=1e-6), f"Test failed: {euler_yaw} != 0.0"
        assert np.allclose(euler_yaw[1], 0.0, atol=1e-6), f"Test failed: {euler_yaw} != 0.0"


def test_quat_apply_yaw(test_angles, test_vecs):
    for euler, vec in zip(test_angles, test_vecs):
        quat = Rotation.from_euler("xyz", euler, degrees=False).as_quat()
        quat_torch = torch.tensor(quat).unsqueeze(0).float()
        vec_torch = torch.tensor(vec).unsqueeze(0).float()
        vec_yaw = quat_apply_yaw(quat_torch, vec_torch, w_last=True)
        vec_yaw_np = vec_yaw.squeeze().numpy()
        euler_yaw = euler.copy()
        euler_yaw[0] = 0.0
        euler_yaw[1] = 0.0
        vec_yaw_gt = Rotation.from_euler("xyz", euler_yaw, degrees=False).apply(vec)
        assert np.allclose(vec_yaw_np, vec_yaw_gt, atol=1e-6), (
            f"Test failed: {vec_yaw_np} != {vec_yaw_gt} for Euler: {euler} and Vec: {vec}"
        )


# ==============================================================================
# Additional rotation tests (migrated from test_torch_utils.py)
# ==============================================================================


@pytest.fixture
def quaternion_data():
    # Create some test quaternions
    q1 = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)  # Unit quaternion
    q2 = torch.tensor([0.3, 0.4, 0.5, 0.6], dtype=torch.float32)  # Non-unit quaternion
    v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)  # Unit vector along x-axis
    return q1, q2, v


def test_quat_mul_properties(quaternion_data):
    q1, q2, _ = quaternion_data

    # Test associativity
    q3 = torch.tensor([0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    result1 = quat_mul(quat_mul(q1, q2, w_last=True), q3, w_last=True)
    result2 = quat_mul(q1, quat_mul(q2, q3, w_last=True), w_last=True)
    assert torch.allclose(result1, result2, atol=1e-6)

    # Test identity quaternion
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    assert torch.allclose(quat_mul(q1, identity, w_last=True), q1)
    assert torch.allclose(quat_mul(identity, q1, w_last=True), q1)


def test_quat_apply_vs_rotate(quaternion_data):
    q, _, v = quaternion_data

    # Reshape quaternion to match expected dimensions [1, 4]
    q = q.unsqueeze(0)
    v = v.unsqueeze(0)

    # Test that quat_apply and quat_rotate give same results
    result1 = quat_apply(q, v, w_last=True)
    result2 = quat_rotate(q, v, w_last=True)
    assert torch.allclose(result1, result2, atol=1e-6)

    # Test rotation of unit vector
    rotated = quat_apply(q, v, w_last=True)
    assert torch.allclose(torch.norm(rotated, dim=-1), torch.tensor(1.0), atol=1e-6)


def test_quat_conjugate_properties(quaternion_data):
    q, _, _ = quaternion_data

    # Test conjugate properties
    conj = quat_conjugate(q, w_last=True)
    assert torch.allclose(conj[:3], -q[:3])  # Vector part should be negated
    assert torch.allclose(conj[3], q[3])  # Scalar part should be same

    # Test that conjugate of conjugate is original
    assert torch.allclose(quat_conjugate(conj, w_last=True), q)


def test_quat_unit_normalization(quaternion_data):
    _, q2, _ = quaternion_data

    # Test that output is unit quaternion
    unit_q = quat_unit(q2)
    assert torch.allclose(torch.norm(unit_q), torch.tensor(1.0), atol=1e-6)

    # Test that already unit quaternion is unchanged
    q1, _, _ = quaternion_data
    assert torch.allclose(quat_unit(q1), q1)


def test_quat_from_angle_axis_rotation():
    # Test rotation around x-axis
    angle = torch.tensor(np.pi / 2, dtype=torch.float32)
    axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    q = quat_from_angle_axis(angle, axis, w_last=True)

    # Should be unit quaternion
    assert torch.allclose(torch.norm(q), torch.tensor(1.0), atol=1e-6)

    # Test rotation of vector
    v = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    rotated = quat_apply(q, v, w_last=True)
    expected = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    assert torch.allclose(rotated, expected, atol=1e-6)


def test_normalize_angle_range():
    # Test normalization to [-pi, pi]
    angles = torch.tensor([3 * np.pi, -3 * np.pi, np.pi / 2, -np.pi / 2], dtype=torch.float32)
    normalized = normalize_angle(angles)
    assert torch.all(normalized >= -np.pi)
    assert torch.all(normalized <= np.pi)

    # Test that sin and cos are preserved
    original_sin = torch.sin(angles)
    original_cos = torch.cos(angles)
    normalized_sin = torch.sin(normalized)
    normalized_cos = torch.cos(normalized)
    assert torch.allclose(original_sin, normalized_sin, atol=1e-6)
    assert torch.allclose(original_cos, normalized_cos, atol=1e-6)


def test_slerp_clamps_cosine_roundoff() -> None:
    slightly_over_unit = torch.tensor([[0.0, 0.0, 0.0, 1.0000001]])

    interpolated = slerp(slightly_over_unit, slightly_over_unit, torch.tensor([[0.5]]))

    assert torch.isfinite(interpolated).all()
