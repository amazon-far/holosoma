# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Shared mount-pose API contract; backend effects are covered by live sensor tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from holosoma.config_types.sensor import CameraSensorConfig, LidarSensorConfig, SensorMountConfig
from holosoma.simulator.shared.sensor_manager import SensorManager, SensorRecord
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_empty_manager_is_usable_before_native_device_initialization(device: str) -> None:
    manager = SensorManager(device, 50.0, num_envs=3)
    assert manager.names == []
    assert manager.collect_due() == []
    assert manager.collect_lidars_due() == []
    with pytest.raises(KeyError, match="No sensor named 'missing'"):
        manager.get_mount_pose("missing")


@pytest.fixture(params=["camera", "lidar"])
def manager(request: pytest.FixtureRequest) -> SensorManager:
    mount = SensorMountConfig(
        target_kind="actor", target="fixture", position=[0.1, 0.2, 0.3], orientation=[0.0, 1.0, 0.0, 0.0]
    )
    manager = SensorManager("cpu", 50.0, num_envs=3)
    record: SensorRecord
    if request.param == "camera":
        record = manager.register_camera("sensor", CameraSensorConfig(mount=mount, update_decimation=3))
    else:
        record = manager.register_lidar("sensor", LidarSensorConfig(mount=mount, update_decimation=3))
    record.write_mount_pose = Mock()
    return manager


def test_initial_pose_uses_runtime_xyzw_and_read_returns_a_copy(manager: SensorManager) -> None:
    poses = manager.get_mount_pose("sensor")
    torch.testing.assert_close(poses, torch.tensor([[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]]).repeat(3, 1))
    poses.zero_()
    assert manager.get_mount_pose("sensor")[0, 0] == pytest.approx(0.1)


def test_subset_order_broadcast_and_config_independence(manager: SensorManager) -> None:
    original = manager.get_mount_pose("sensor")
    updates = torch.tensor([[0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], [0.7, 0.8, 0.9, 0.0, 0.0, 1.0, 0.0]])
    ids = torch.tensor([2, 0])
    manager.set_mount_pose("sensor", updates, ids)
    torch.testing.assert_close(manager.get_mount_pose("sensor", ids), updates)
    torch.testing.assert_close(manager.get_mount_pose("sensor")[1], original[1])
    record = manager.get_sensor("sensor")
    writer = record.write_mount_pose
    assert isinstance(writer, Mock)
    writer.assert_called_once()
    written_ids, written_poses = writer.call_args.args
    torch.testing.assert_close(written_ids, ids)
    torch.testing.assert_close(written_poses, updates)
    assert record.config.mount.position == [0.1, 0.2, 0.3]
    assert record.config.mount.orientation == [0.0, 1.0, 0.0, 0.0]
    updates.zero_()
    assert manager.get_mount_pose("sensor")[2, 0] == pytest.approx(0.4)
    manager.set_mount_pose("sensor", original[0])
    torch.testing.assert_close(manager.get_mount_pose("sensor"), original)
    written_ids, written_poses = writer.call_args.args
    torch.testing.assert_close(written_ids, torch.arange(3))
    torch.testing.assert_close(written_poses, original)


def test_pose_update_preserves_last_frame_and_cadence(manager: SensorManager) -> None:
    record = manager.get_sensor("sensor")
    frame = torch.ones(3, 1, 1)
    record.set_buffer("output", frame)
    record.step_counter = 7
    manager.set_mount_pose("sensor", torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.00005]))
    assert record.buffers["output"] is frame
    assert record.step_counter == 7
    assert record.effective_decimation == 3
    torch.testing.assert_close(manager.get_mount_pose("sensor")[:, 6], torch.ones(3))
    writer = record.write_mount_pose
    assert isinstance(writer, Mock)
    torch.testing.assert_close(
        writer.call_args.args[1], torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)
    )


@pytest.mark.parametrize(
    "poses",
    [
        torch.zeros(7),
        torch.ones(7),
        torch.tensor([float("nan"), 0, 0, 0, 0, 0, 1]),
        torch.tensor([0, 0, 0, float("inf"), 0, 0, 1]),
        torch.zeros(2, 7),
        torch.zeros(3, 6),
    ],
)
def test_invalid_pose_is_rejected_without_writing(manager: SensorManager, poses: torch.Tensor) -> None:
    original = manager.get_mount_pose("sensor")
    with pytest.raises(ValueError, match="Sensor mount"):
        manager.set_mount_pose("sensor", poses)
    torch.testing.assert_close(manager.get_mount_pose("sensor"), original)
    writer = manager.get_sensor("sensor").write_mount_pose
    assert isinstance(writer, Mock)
    writer.assert_not_called()


@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([-1]),
        torch.tensor([3]),
        torch.tensor([0.5]),
        torch.tensor([False]),
        torch.tensor([[0]]),
        torch.tensor([1, 1]),
    ],
)
def test_invalid_environment_ids_are_rejected(manager: SensorManager, ids: torch.Tensor) -> None:
    original = manager.get_mount_pose("sensor")
    with pytest.raises(ValueError, match="env_ids"):
        manager.set_mount_pose("sensor", original[0], ids)
    torch.testing.assert_close(manager.get_mount_pose("sensor"), original)
    writer = manager.get_sensor("sensor").write_mount_pose
    assert isinstance(writer, Mock)
    writer.assert_not_called()


def test_empty_subset_and_failed_writer_do_not_change_state(manager: SensorManager) -> None:
    original = manager.get_mount_pose("sensor")
    manager.set_mount_pose("sensor", original[:0], torch.empty(0, dtype=torch.long))
    writer = manager.get_sensor("sensor").write_mount_pose
    assert isinstance(writer, Mock)
    writer.assert_not_called()
    writer.side_effect = RuntimeError("native write failed")
    with pytest.raises(RuntimeError, match="native write failed"):
        manager.set_mount_pose("sensor", torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(manager.get_mount_pose("sensor"), original)


def test_unknown_sensor(manager: SensorManager) -> None:
    with pytest.raises(KeyError, match="No sensor named 'missing'"):
        manager.get_mount_pose("missing")
    with pytest.raises(KeyError, match="No sensor named 'missing'"):
        manager.set_mount_pose("missing", torch.zeros(7))
