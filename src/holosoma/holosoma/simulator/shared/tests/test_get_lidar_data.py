"""Unit tests for the shared ``BaseSimulator.get_lidar_data`` accessor."""

from __future__ import annotations

from typing import Any

import pytest

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.shared.sensor_manager import LidarRecord
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


class _Manager:
    """Minimal LiDAR-manager surface with prefilled output records."""

    def __init__(self, runtimes: dict[str, LidarRecord]) -> None:
        self._runtimes = runtimes

    def has_lidar(self, name: str) -> bool:
        return name in self._runtimes

    def get_lidar(self, name: str) -> LidarRecord:
        return self._runtimes[name]

    @property
    def lidar_names(self) -> list[str]:
        return list(self._runtimes)


def _sim(manager: Any) -> BaseSimulator:
    sim = BaseSimulator.__new__(BaseSimulator)
    sim.sensor_manager = manager
    return sim


def test_missing_lidar_or_manager_raises_not_implemented() -> None:
    for manager in (None, _Manager({})):
        with pytest.raises(NotImplementedError, match="no LiDAR 'scan'"):
            _sim(manager).get_lidar_data("scan")


def test_missing_lidar_output_raises_runtime_error() -> None:
    record = LidarRecord(name="scan", config=None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no 'points' output"):
        _sim(_Manager({"scan": record})).get_lidar_data("scan")


def test_returns_latest_output_and_env_subset() -> None:
    points = torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3)
    ranges = torch.arange(3 * 2, dtype=torch.float32).reshape(3, 2)
    record = LidarRecord(name="scan", config=None, buffers={"points": points, "ranges": ranges})  # type: ignore[arg-type]
    sim = _sim(_Manager({"scan": record}))

    assert sim.get_lidar_data("scan") is points
    assert torch.equal(sim.get_lidar_data("scan", "ranges", env_ids=[0, 2]), ranges[[0, 2]])


def test_same_device_is_a_direct_buffer_read() -> None:
    points = torch.zeros(1, 2, 3)
    record = LidarRecord(name="scan", config=None, buffers={"points": points})  # type: ignore[arg-type]

    assert _sim(_Manager({"scan": record})).get_lidar_data("scan", device="cpu") is points
    assert record._device_cache == {}
