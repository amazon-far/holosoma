"""``setup_camera_mount``: per-env camera mount poses drawn through the term sampler."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.managers.randomization.exceptions import RandomizerNotSupportedError
from holosoma.managers.randomization.terms.sensors import setup_camera_mount
from holosoma.simulator.shared.sensor_manager import SensorManager
from holosoma.utils.rotations import get_euler_xyz, quat_from_euler_xyz, quat_mul
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sampler import STAGE_SETUP, TermSampler

pytestmark = pytest.mark.no_sim

NUM_ENVS = 64
POSITION = [[-0.025, 0.025], [-0.01, 0.01], [0.0, 0.05]]
EULER_DEG = [[-2.5, 2.5], [-3.0, 3.0], [-1.0, 1.0]]
MOUNT_POSITION = [0.1, 0.0, 0.4]


def _env(orientation: list[float] | None = None, *, writer: bool = True) -> tuple[Any, SensorManager]:
    manager = SensorManager(device="cpu", control_hz=50.0, num_envs=NUM_ENVS)
    mount = SensorMountConfig(
        target_kind="robot_link", target="torso_link", position=MOUNT_POSITION, orientation=orientation or [1, 0, 0, 0]
    )
    record = manager.register_camera("head_depth", CameraSensorConfig(mount=mount, data_types=["depth"]))
    if writer:
        record.write_mount_pose = lambda _env_ids, _poses: None
    env = SimpleNamespace(num_envs=NUM_ENVS, device="cpu", simulator=SimpleNamespace(sensor_manager=manager))
    return env, manager


def _apply(env: Any, seed: int = 7, **overrides: Any) -> None:
    params: dict[str, Any] = {
        "sensor": "head_depth",
        "position_range": POSITION,
        "euler_deg_range": EULER_DEG,
        "enabled": True,
    }
    params.update(overrides)
    setup_camera_mount(env, TermSampler.bind(seed, "setup_camera_mount", STAGE_SETUP), **params)


def test_poses_are_per_env_within_each_axis_range() -> None:
    env, manager = _env()
    _apply(env)

    poses = manager.get_mount_pose("head_depth")
    offset = poses[:, :3] - torch.tensor(MOUNT_POSITION)
    euler = torch.stack(get_euler_xyz(poses[:, 3:], w_last=True), dim=-1)
    euler = torch.atan2(torch.sin(euler), torch.cos(euler))  # wrap to (-pi, pi]
    for axis, (lo, hi) in enumerate(POSITION):
        assert torch.all((offset[:, axis] >= lo - 1e-6) & (offset[:, axis] <= hi + 1e-6))
    for axis, (lo, hi) in enumerate(EULER_DEG):
        assert torch.all((euler[:, axis] >= math.radians(lo) - 1e-5) & (euler[:, axis] <= math.radians(hi) + 1e-5))
    assert offset[:, 0].unique().numel() == NUM_ENVS
    assert not torch.allclose(offset[:, 0] / 0.025, euler[:, 0] / math.radians(2.5))


def test_rotation_offset_is_applied_before_the_mount_orientation() -> None:
    half = math.radians(45.0) / 2  # mount rotated 45 deg about z
    env, manager = _env([math.cos(half), 0.0, 0.0, math.sin(half)])
    _apply(env, position_range=[[0.0, 0.0]] * 3, euler_deg_range=[[10.0, 10.0], [0.0, 0.0], [0.0, 0.0]])

    roll = torch.full((NUM_ENVS,), math.radians(10.0))
    zero = torch.zeros(NUM_ENVS)
    mount = torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)]).expand(NUM_ENVS, 4).contiguous()
    expected = quat_mul(quat_from_euler_xyz(roll, zero, zero), mount, w_last=True)
    torch.testing.assert_close(manager.get_mount_pose("head_depth")[:, 3:], expected, atol=1e-6, rtol=0.0)


def test_poses_are_reproducible_for_a_seed() -> None:
    (first_env, first), (second_env, second), (other_env, other) = _env(), _env(), _env()
    _apply(first_env)
    _apply(second_env)
    _apply(other_env, seed=8)

    assert torch.equal(first.get_mount_pose("head_depth"), second.get_mount_pose("head_depth"))
    assert not torch.equal(first.get_mount_pose("head_depth"), other.get_mount_pose("head_depth"))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"sensor": "missing"}, "no sensor named 'missing'"),
        ({"position_range": POSITION[:2]}, "per axis"),
    ],
)
def test_bad_term_configs_are_rejected(overrides: dict[str, Any], match: str) -> None:
    env, _ = _env()
    with pytest.raises(ValueError, match=match):
        _apply(env, **overrides)


def test_sensors_without_a_mount_pose_writer_are_unsupported() -> None:
    env, _ = _env(writer=False)
    with pytest.raises(RandomizerNotSupportedError):
        _apply(env)
