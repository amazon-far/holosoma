"""Camera outputs: one fixed buffer per data type, and per-frame snapshots on request (CPU-only)."""

from __future__ import annotations

import pytest

from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.simulator.shared.sensor_manager import CameraRecord, SensorManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def _camera() -> tuple[SensorManager, CameraRecord]:
    manager = SensorManager("cpu", 50.0, num_envs=2)
    config = CameraSensorConfig(
        mount=SensorMountConfig(target_kind="world", target=""), width=2, height=2, data_types=["rgb", "depth"]
    )
    return manager, manager.register_camera("c", config)


def test_frames_are_written_into_the_served_output() -> None:
    manager, record = _camera()
    served = record.buffer_on("rgb")
    assert manager.state.get("camera/c/rgb") is served and served.dtype == torch.uint8
    record.set_buffer("rgb", torch.full((2, 2, 2, 3), 9, dtype=torch.uint8))
    assert record.buffer_on("rgb") is served and int(served[0, 0, 0, 0]) == 9


def test_a_snapshot_is_shared_within_a_frame_and_kept_after_the_next() -> None:
    _, record = _camera()
    record.set_buffer("rgb", torch.full((2, 2, 2, 3), 1, dtype=torch.uint8))
    first = record.buffer_on("rgb", "cpu")
    assert record.buffer_on("rgb", "cpu") is first and first.data_ptr() != record.buffers["rgb"].data_ptr()

    record.set_buffer("depth", torch.zeros(2, 2, 2, 1))  # another output keeps its snapshot
    assert record.buffer_on("rgb", "cpu") is first
    record.set_buffer("rgb", torch.full((2, 2, 2, 3), 2, dtype=torch.uint8))
    assert int(first[0, 0, 0, 0]) == 1 and int(record.buffer_on("rgb", "cpu")[0, 0, 0, 0]) == 2


def test_a_consumer_write_into_an_output_is_rejected() -> None:
    manager, record = _camera()
    record.buffer_on("rgb")[0] = 5
    with pytest.raises(RuntimeError, match="read-only"):
        record.set_buffer("rgb", torch.zeros(2, 2, 2, 3, dtype=torch.uint8))
    record.buffer_on("rgb")[0] = 6
    with pytest.raises(RuntimeError, match="read-only"):
        manager.state.flush_writes()
