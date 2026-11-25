"""BaseSimulator's sensor state: the remote-camera barrier and the ``mount_pose/<sensor>`` fields."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any, cast

import pytest

from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.state import SimState
from holosoma.simulator.shared.sensor_manager import SensorManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def _cam() -> CameraSensorConfig:
    return CameraSensorConfig(mount=SensorMountConfig(target_kind="robot_link", target="pelvis"))


def test_a_due_remote_camera_without_a_renderer_fails_at_the_sensor_render_hook() -> None:
    manager = SensorManager("cpu", 50.0, num_envs=1)
    manager.register_camera("cam", dataclasses.replace(_cam(), remote="cosim"))
    manager.collect_due()
    with pytest.raises(RuntimeError, match=r"did not fill them: \{'cam': 'cosim'\}"):
        BaseSimulator._sensors_rendered(cast("Any", SimpleNamespace(sensor_manager=manager)))


def test_mount_pose_state_follows_set_mount_pose_and_writes_only_changed_rows() -> None:
    manager = SensorManager("cpu", 50.0, num_envs=2)
    local = manager.register_camera("local", _cam())
    manager.register_camera("remote", dataclasses.replace(_cam(), remote="cosim"))
    writes: list[list[int]] = []
    local.write_mount_pose = lambda ids, _poses: writes.append(ids.tolist())
    state = SimState()
    BaseSimulator._register_mount_poses(cast("Any", SimpleNamespace(state=state, sensor_manager=manager)))

    state.get("mount_pose/remote")
    moved = torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    manager.set_mount_pose("remote", moved)  # stored without a writer; the cached field is re-read
    torch.testing.assert_close(state.get("mount_pose/remote"), moved.expand(2, 7))

    poses = state.get("mount_pose/local")
    poses.copy_(poses.clone())
    state.flush_writes()
    assert writes == []
    poses[1, 0] += 0.5
    state.flush_writes()
    assert writes == [[1]]
    torch.testing.assert_close(manager.get_mount_pose("local"), poses)


def test_writing_the_mount_pose_field_keeps_it_valid_and_matching_the_sensor() -> None:
    manager = SensorManager("cpu", 50.0, num_envs=1)
    manager.register_camera("cam", _cam()).write_mount_pose = lambda _ids, _poses: None
    state = SimState()
    BaseSimulator._register_mount_poses(cast("Any", SimpleNamespace(state=state, sensor_manager=manager)))

    held = state.get("mount_pose/cam")
    for step in (0.1, 0.2):  # a held reference can be written and flushed repeatedly
        held[0, 0] += step
        state.flush_writes()
    torch.testing.assert_close(manager.get_mount_pose("cam"), held)

    near_unit = held.clone()
    near_unit[0, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.00005])
    held.copy_(near_unit)
    state.flush_writes()
    torch.testing.assert_close(state.get("mount_pose/cam"), manager.get_mount_pose("cam"))  # normalised

    stored = manager.get_mount_pose("cam")
    held = state.get("mount_pose/cam")
    held[0, 3:] = torch.tensor([0.0, 0.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="unit quaternion"):
        state.flush_writes()
    state.flush_writes()  # the rejected write is discarded, not retried
    torch.testing.assert_close(state.get("mount_pose/cam"), stored)
