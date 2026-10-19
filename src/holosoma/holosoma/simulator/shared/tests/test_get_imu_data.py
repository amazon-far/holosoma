# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Unit tests for the shared ``BaseSimulator.get_imu_data`` accessor."""

from __future__ import annotations

from typing import Any

import pytest

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.shared.sensor_manager import ImuRecord
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


class _Manager:
    """Minimal IMU-manager surface with prefilled output records."""

    def __init__(self, runtimes: dict[str, ImuRecord]) -> None:
        self._runtimes = runtimes

    def has_imu(self, name: str) -> bool:
        return name in self._runtimes

    def get_imu(self, name: str) -> ImuRecord:
        return self._runtimes[name]

    @property
    def imu_names(self) -> list[str]:
        return list(self._runtimes)


def _sim(manager: Any) -> BaseSimulator:
    sim = BaseSimulator.__new__(BaseSimulator)
    sim.sensor_manager = manager
    return sim


def test_missing_imu_or_manager_raises_not_implemented() -> None:
    for manager in (None, _Manager({})):
        with pytest.raises(NotImplementedError, match="no IMU 'imu'"):
            _sim(manager).get_imu_data("imu")


def test_missing_imu_output_raises_runtime_error() -> None:
    record = ImuRecord(name="imu", config=None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no 'angular_velocity' output"):
        _sim(_Manager({"imu": record})).get_imu_data("imu")


def test_default_data_type_is_angular_velocity() -> None:
    gyro = torch.tensor([[0.1, 0.2, 0.3]])
    record = ImuRecord(name="imu", config=None, buffers={"angular_velocity": gyro})  # type: ignore[arg-type]

    assert _sim(_Manager({"imu": record})).get_imu_data("imu") is gyro


def test_returns_latest_output_and_env_subset() -> None:
    accel = torch.arange(3 * 3, dtype=torch.float32).reshape(3, 3)
    orientation = torch.zeros(3, 4)
    record = ImuRecord(
        name="imu",
        config=None,  # type: ignore[arg-type]
        buffers={"linear_acceleration": accel, "orientation": orientation},
    )
    sim = _sim(_Manager({"imu": record}))

    assert sim.get_imu_data("imu", "linear_acceleration") is accel
    assert torch.equal(sim.get_imu_data("imu", "linear_acceleration", env_ids=[0, 2]), accel[[0, 2]])
    assert sim.get_imu_data("imu", "orientation").shape == (3, 4)


def test_same_device_is_a_direct_buffer_read() -> None:
    gyro = torch.zeros(1, 3)
    record = ImuRecord(name="imu", config=None, buffers={"angular_velocity": gyro})  # type: ignore[arg-type]

    assert _sim(_Manager({"imu": record})).get_imu_data("imu", device="cpu") is gyro
    assert record._device_cache == {}


def test_error_names_the_created_imus() -> None:
    """A typo'd sensor key should say what is available, not just that the lookup failed."""
    record = ImuRecord(name="pelvis_imu", config=None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="pelvis_imu"):
        _sim(_Manager({"pelvis_imu": record})).get_imu_data("torso_imu")
