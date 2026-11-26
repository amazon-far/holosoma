"""Partial sensor re-renders write their rows into fresh buffers."""

from __future__ import annotations

from typing import Any

import pytest

from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.simulator.shared.sensor_manager import SensorManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

NUM_ENVS = 5


def _depth_camera(manager: SensorManager, name: str) -> Any:
    config = CameraSensorConfig(
        mount=SensorMountConfig(target_kind="robot_link", target="pelvis"),
        near=0.1,
        far=3.0,
        depth_clipping_behavior="max",
        data_types=["depth"],
    )
    record = manager.register_camera(name, config)
    record.set_buffer("depth", torch.ones(NUM_ENVS, 2, 2, 1))
    return record


def test_camera_buffer_rows_apply_depth_clipping_and_drop_device_copies() -> None:
    manager = SensorManager(device="cpu", control_hz=50.0, physics_hz=50.0)
    record = _depth_camera(manager, "cam")
    stale_copy = record.buffer_on("depth", "meta")
    rows = torch.tensor([float("inf"), 1.0, 0.05, 2.0]).view(1, 2, 2, 1)

    record.set_buffer_rows("depth", torch.tensor([3]), rows)

    assert record.buffers["depth"][3].flatten().tolist() == [3.0, 1.0, 3.0, 2.0]
    assert torch.all(record.buffers["depth"][[0, 1, 2, 4]] == 1.0)
    assert record.buffer_on("depth", "meta") is not stale_copy


def test_buffer_rows_leave_previously_published_buffers_untouched() -> None:
    manager = SensorManager(device="cpu", control_hz=50.0, physics_hz=50.0)
    record = _depth_camera(manager, "cam")
    published = record.buffers["depth"]

    record.set_buffer_rows("depth", torch.tensor([0]), torch.full((1, 2, 2, 1), 2.0))

    assert torch.all(published == 1.0)
    assert record.buffers["depth"] is not published
