"""Partial sensor re-renders write only their rows, in place."""

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
        width=2,
        height=2,
    )
    record = manager.register_camera(name, config)
    record.set_buffer("depth", torch.ones(NUM_ENVS, 2, 2, 1))
    return record


def test_camera_buffer_rows_apply_depth_clipping_and_drop_device_copies() -> None:
    manager = SensorManager(device="cpu", control_hz=50.0, num_envs=NUM_ENVS)
    record = _depth_camera(manager, "cam")
    stale_copy = record.buffer_on("depth", "cpu")
    rows = torch.tensor([float("inf"), 1.0, 0.05, 2.0]).view(1, 2, 2, 1)

    ids = torch.tensor([3])
    with record.writing("depth", ids) as depth:  # a partial re-render writes only these rows
        depth[ids] = rows

    assert record.buffers["depth"][3].flatten().tolist() == [3.0, 1.0, 3.0, 2.0]
    assert torch.all(record.buffers["depth"][[0, 1, 2, 4]] == 1.0)
    assert record.buffer_on("depth", "cpu") is not stale_copy
