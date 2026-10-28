"""Tests for the bridge's root-state reuse — pure, no simulator, no DDS.

``robot_root_states`` is a backend-owned proxy whose every index rebuilds the whole 13-vector, and
one transport update reads it three times (twice for the IMU, once for odometry) at the same
simulator state. The update scope collapses that to one read; outside it, each reader still reads
the proxy directly, and the values must be identical either way.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from holosoma.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge
from holosoma.config_types.robot import RobotBridgeConfig
from holosoma.config_types.simulator import BridgeConfig
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.robot import RobotConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

pytestmark = pytest.mark.no_sim

_NUM_DOF = 2


class _CountingRootStates:
    """Stands in for a backend root-state proxy, counting how often it is indexed."""

    def __init__(self, root: torch.Tensor) -> None:
        self._root = root
        self.reads = 0

    def __getitem__(self, key: Any) -> torch.Tensor:
        self.reads += 1
        return self._root[key]


class _FakeSim:
    """Exposes exactly what the base bridge reads for a transport update."""

    def __init__(self) -> None:
        self.dof_names = [f"j{i}" for i in range(_NUM_DOF)]
        self.num_dof = _NUM_DOF
        self.device = "cpu"
        self.dof_pos = torch.zeros(1, _NUM_DOF)
        self.dof_vel = torch.zeros(1, _NUM_DOF)
        self.dof_acc = torch.zeros(1, _NUM_DOF)
        root = torch.zeros(1, 13)
        root[0, 0:3] = torch.tensor([1.0, 2.0, 3.0])  # position
        root[0, 3:7] = torch.tensor([0.0, 0.0, 0.3826834, 0.9238795])  # 45 deg yaw, xyzw
        root[0, 7:10] = torch.tensor([0.5, -0.25, 0.1])  # world linear velocity
        root[0, 10:13] = torch.tensor([0.05, -0.1, 0.3])  # world angular velocity
        self.robot_root_states = _CountingRootStates(root)
        self.base_linear_acc = torch.zeros(1, 3)

    def time(self) -> float:
        return 1.5


class _StubBridge(BasicSdk2Bridge):
    """Minimal concrete bridge: no SDK, so the abstract hooks are trivial no-ops."""

    def _init_sdk_components(self) -> None:
        pass

    def low_cmd_handler(self, msg: Any = None) -> None:
        pass

    def publish_low_state(self) -> None:
        pass

    def compute_torques(self) -> npt.NDArray[np.floating[Any]]:
        return self.torques


def _bridge(sim: _FakeSim) -> _StubBridge:
    robot_config = SimpleNamespace(
        dof_effort_limit_list=[100.0] * _NUM_DOF,
        asset=SimpleNamespace(robot_type="g1_29dof"),
        bridge=RobotBridgeConfig(),
    )
    return _StubBridge(
        cast("BaseSimulator", sim),
        cast("RobotConfig", robot_config),
        BridgeConfig(interface="lo"),
    )


def test_transport_update_reads_root_state_once_for_every_reader() -> None:
    sim = _FakeSim()
    bridge = _bridge(sim)
    sim.robot_root_states.reads = 0

    with bridge.transport_update():
        bridge._get_base_imu_data()
        bridge._get_base_odometry()

    # Two IMU reads + one odometry read collapse to a single proxy index.
    assert sim.robot_root_states.reads == 1


def test_readers_outside_a_transport_update_still_read_the_proxy_each_time() -> None:
    sim = _FakeSim()
    bridge = _bridge(sim)
    sim.robot_root_states.reads = 0

    bridge._get_base_imu_data()
    bridge._get_base_odometry()

    # Unchanged from before the update scope existed: one read per reader call.
    assert sim.robot_root_states.reads == 2


def test_transport_update_does_not_change_the_values_either_reader_returns() -> None:
    sim = _FakeSim()
    bridge = _bridge(sim)

    quat_direct, gyro_direct, acc_direct = bridge._get_base_imu_data()
    odom_direct = bridge._get_base_odometry()

    with bridge.transport_update():
        quat_scoped, gyro_scoped, acc_scoped = bridge._get_base_imu_data()
        odom_scoped = bridge._get_base_odometry()

    torch.testing.assert_close(quat_direct, quat_scoped)
    torch.testing.assert_close(gyro_direct, gyro_scoped)
    torch.testing.assert_close(acc_direct, acc_scoped)
    assert odom_direct == odom_scoped
    # The rotation is actually exercised: a 45 deg yaw makes body velocity differ from world.
    assert odom_scoped[2] != pytest.approx(sim.robot_root_states._root[0, 7:10].tolist())


def test_cached_root_state_is_dropped_when_the_update_ends() -> None:
    sim = _FakeSim()
    bridge = _bridge(sim)

    with bridge.transport_update():
        bridge._get_base_odometry()
        bridge._get_base_odometry()
    sim.robot_root_states.reads = 0

    # A later update re-reads rather than serving the first update's snapshot, so it sees state
    # written between the two.
    sim.robot_root_states._root[0, 0] = 9.0
    with bridge.transport_update():
        position, _quat, _lin, _yaw = bridge._get_base_odometry()
    assert sim.robot_root_states.reads == 1
    assert position[0] == pytest.approx(9.0)


def test_cache_is_dropped_even_when_the_update_raises() -> None:
    sim = _FakeSim()
    bridge = _bridge(sim)

    def update_that_fails_after_reading_state() -> None:
        with bridge.transport_update():
            bridge._get_base_odometry()
            raise RuntimeError("publish failed")

    with pytest.raises(RuntimeError, match="publish failed"):
        update_that_fails_after_reading_state()

    # The next read is served from the simulator, not from the failed update's snapshot.
    sim.robot_root_states.reads = 0
    sim.robot_root_states._root[0, 0] = 5.0
    position, _quat, _lin, _yaw = bridge._get_base_odometry()
    assert sim.robot_root_states.reads == 1
    assert position[0] == pytest.approx(5.0)


def test_nested_transport_updates_stay_correct() -> None:
    """A nested scope ends the cache early; readers then re-read, so values are never stale."""
    sim = _FakeSim()
    bridge = _bridge(sim)

    with bridge.transport_update():
        bridge._get_base_odometry()
        with bridge.transport_update():
            bridge._get_base_odometry()

        # The inner scope ended the shared read, so state written after it is still picked up.
        sim.robot_root_states._root[0, 0] = 7.0
        position, _quat, _lin, _yaw = bridge._get_base_odometry()
        assert position[0] == pytest.approx(7.0)

    # And the outer scope left no snapshot behind either.
    sim.robot_root_states.reads = 0
    bridge._get_base_odometry()
    assert sim.robot_root_states.reads == 1
