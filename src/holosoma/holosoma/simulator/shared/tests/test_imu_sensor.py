# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Unit tests for the shared IMU output finalizer (pure, no simulator).

The quaternion reorder is the risk here: ``wxyz`` and ``xyzw`` are both four floats, so a wrong
order still looks like a unit quaternion. These tests check the buffered value by *rotating a
vector with it* through the repo's own ``w_last=True`` helpers rather than by asserting the
component order, which would only restate the implementation.
"""

from __future__ import annotations

import pytest

from holosoma.config_types.sensor import ImuSensorConfig, SensorMountConfig
from holosoma.simulator.shared.imu_sensor import finalize_imu_outputs
from holosoma.simulator.shared.sensor_manager import ImuRecord
from holosoma.utils.rotations import quat_rotate_inverse
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

_MOUNT = SensorMountConfig(target_kind="robot_link", target="pelvis")
_HALF_SQRT2 = 0.7071067811865476


def _record(name: str = "imu") -> ImuRecord:
    return ImuRecord(name=name, config=ImuSensorConfig(mount=_MOUNT))


def _finalize(
    record: ImuRecord,
    *,
    accel: torch.Tensor | None = None,
    gyro: torch.Tensor | None = None,
    quat_wxyz: torch.Tensor | None = None,
) -> None:
    finalize_imu_outputs(
        record,
        linear_acceleration=accel if accel is not None else torch.zeros(1, 3),
        angular_velocity=gyro if gyro is not None else torch.zeros(1, 3),
        orientation_wxyz=quat_wxyz if quat_wxyz is not None else torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )


def test_writes_the_three_documented_outputs() -> None:
    record = _record()
    _finalize(record, accel=torch.tensor([[0.0, 0.0, 9.81]]), gyro=torch.tensor([[0.1, 0.2, 0.3]]))

    assert sorted(record.buffers) == ["angular_velocity", "linear_acceleration", "orientation"]
    assert record.buffers["linear_acceleration"].shape == (1, 3)
    assert record.buffers["orientation"].shape == (1, 4)


def test_outputs_are_cast_to_float32() -> None:
    record = _record()
    _finalize(
        record,
        accel=torch.zeros(1, 3, dtype=torch.float64),
        gyro=torch.zeros(1, 3, dtype=torch.float64),
        quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
    )

    assert all(buffer.dtype == torch.float32 for buffer in record.buffers.values())


@pytest.mark.parametrize(
    ("quat_wxyz", "world_vector", "expected_sensor_vector"),
    [
        # Identity: the sensor frame is the world frame.
        ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        # +90 deg about Z: world +X arrives along the sensor's -Y.
        ([_HALF_SQRT2, 0.0, 0.0, _HALF_SQRT2], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        # +90 deg about Y: world +Z arrives along the sensor's -X.
        ([_HALF_SQRT2, 0.0, _HALF_SQRT2, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),
        # +90 deg about X: world +Y arrives along the sensor's -Z.
        ([_HALF_SQRT2, _HALF_SQRT2, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
    ],
)
def test_buffered_orientation_rotates_vectors_as_xyzw(
    quat_wxyz: list[float],
    world_vector: list[float],
    expected_sensor_vector: list[float],
) -> None:
    """A native ``wxyz`` attitude must buffer as ``xyzw``, usable directly with ``w_last=True``.

    Each case rotates a world axis into the sensor frame. Passing the buffer to a ``w_last=True``
    helper is how every consumer will use it, so an order mistake fails here even though the
    reordered components would still look like a valid quaternion.
    """
    record = _record()
    _finalize(record, quat_wxyz=torch.tensor([quat_wxyz]))

    rotated = quat_rotate_inverse(record.buffers["orientation"], torch.tensor([world_vector]), w_last=True)
    assert torch.allclose(rotated, torch.tensor([expected_sensor_vector]), atol=1e-6)


def test_rates_pass_through_unrotated() -> None:
    """The natives already report in the sensor frame; the finalizer must not re-rotate them."""
    record = _record()
    accel, gyro = torch.tensor([[1.0, -2.0, 9.81]]), torch.tensor([[0.3, 0.0, -1.5]])
    # A non-identity attitude would change these if the finalizer wrongly applied it.
    _finalize(record, accel=accel, gyro=gyro, quat_wxyz=torch.tensor([[_HALF_SQRT2, 0.0, 0.0, _HALF_SQRT2]]))

    assert torch.equal(record.buffers["linear_acceleration"], accel)
    assert torch.equal(record.buffers["angular_velocity"], gyro)


def test_multiple_envs_keep_their_own_rows() -> None:
    record = _record()
    accel = torch.tensor([[0.0, 0.0, 9.81], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
    _finalize(record, accel=accel, gyro=torch.zeros(3, 3), quat_wxyz=quat)

    assert torch.equal(record.buffers["linear_acceleration"], accel)
    assert record.buffers["orientation"].shape == (3, 4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"accel": torch.zeros(1, 4)}, r"linear_acceleration must be \[num_envs, 3\]"),
        ({"gyro": torch.zeros(1, 2)}, r"angular_velocity must be \[num_envs, 3\]"),
        ({"quat_wxyz": torch.zeros(1, 3)}, r"orientation must be \[num_envs, 4\]"),
        ({"accel": torch.zeros(3)}, r"linear_acceleration must be \[num_envs, 3\]"),
        ({"quat_wxyz": torch.zeros(2, 1, 4)}, r"orientation must be \[num_envs, 4\]"),
    ],
)
def test_wrong_shape_is_rejected_with_the_sensor_name(kwargs: dict[str, torch.Tensor], match: str) -> None:
    record = _record(name="torso_imu")
    with pytest.raises(ValueError, match=match) as excinfo:
        _finalize(record, **kwargs)
    assert "torso_imu" in str(excinfo.value)
    assert record.buffers == {}  # nothing partially written


def test_buffers_never_alias_the_input() -> None:
    """Backends pass slices of a live native buffer (float32 on MJWarp), so a cast alone would hand
    consumers a view that mutates under them on the next physics step."""
    record = _record()
    accel = torch.tensor([[0.0, 0.0, 9.81]], dtype=torch.float32)
    gyro = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    _finalize(record, accel=accel, gyro=gyro)

    assert record.buffers["linear_acceleration"] is not accel
    assert record.buffers["angular_velocity"] is not gyro
    accel.zero_()  # as the backend's live buffer would be overwritten in place
    gyro.zero_()
    assert torch.equal(record.buffers["linear_acceleration"], torch.tensor([[0.0, 0.0, 9.81]]))
    assert torch.equal(record.buffers["angular_velocity"], torch.tensor([[0.0, 0.0, 1.0]]))


def test_set_buffer_invalidates_the_cross_device_cache() -> None:
    """Consecutive samples must not serve a stale host copy to a plugin reading device='cpu'."""
    record = _record()
    _finalize(record, accel=torch.tensor([[0.0, 0.0, 9.81]]))
    assert torch.equal(record.buffer_on("linear_acceleration", "cpu"), torch.tensor([[0.0, 0.0, 9.81]]))

    _finalize(record, accel=torch.tensor([[0.0, 0.0, 0.0]]))
    assert torch.equal(record.buffer_on("linear_acceleration", "cpu"), torch.tensor([[0.0, 0.0, 0.0]]))
