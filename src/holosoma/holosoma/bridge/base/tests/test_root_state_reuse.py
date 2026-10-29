"""Tests for the bridge's root-state reuse — pure, no simulator, no DDS.

``robot_root_states`` is a backend-owned proxy whose every index rebuilds the whole 13-vector, and
one exchange's publishes read it at the same simulator state. ``transport_exchange`` reads it once
for all of them; a publish called outside an exchange still reads the proxy itself, and the values
must be identical either way.
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
    """Minimal concrete bridge: no SDK, so a publish just reads what it would have sent."""

    imu: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    odom: tuple[list[float], list[float], list[float], float] | None = None
    fail_on_publish_odom = False

    def _init_sdk_components(self) -> None:
        pass

    def low_cmd_handler(self, msg: Any = None) -> None:
        pass

    def publish_low_state(self) -> None:
        self.imu = self._get_base_imu_data()

    def publish_odom(self) -> None:
        if self.fail_on_publish_odom:
            raise RuntimeError("publish failed")
        self.odom = self._get_base_odometry()

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


def test_an_exchange_reads_root_state_once_for_every_publish() -> None:
    sim = _FakeSim()
    bridge = _bridge(sim)
    sim.robot_root_states.reads = 0

    bridge.transport_exchange()

    # Two IMU reads plus one odometry read collapse to a single proxy index.
    assert sim.robot_root_states.reads == 1
    assert bridge.imu is not None
    assert bridge.odom is not None


def test_a_publish_outside_an_exchange_reads_the_proxy_itself() -> None:
    sim = _FakeSim()
    bridge = _bridge(sim)
    sim.robot_root_states.reads = 0

    bridge.publish_low_state()
    bridge.publish_odom()

    # Unchanged from before the exchange existed: one read per reader call.
    assert sim.robot_root_states.reads == 2


def test_an_exchange_does_not_change_what_either_reader_returns() -> None:
    sim = _FakeSim()
    bridge = _bridge(sim)

    quat_direct, gyro_direct, acc_direct = bridge._get_base_imu_data()
    odom_direct = bridge._get_base_odometry()

    bridge.transport_exchange()
    assert bridge.imu is not None and bridge.odom is not None
    quat_exchanged, gyro_exchanged, acc_exchanged = bridge.imu

    torch.testing.assert_close(quat_direct, quat_exchanged)
    torch.testing.assert_close(gyro_direct, gyro_exchanged)
    torch.testing.assert_close(acc_direct, acc_exchanged)
    assert odom_direct == bridge.odom
    # The rotation is actually exercised: a 45 deg yaw makes body velocity differ from world.
    assert bridge.odom[2] != pytest.approx(sim.robot_root_states._root[0, 7:10].tolist())


def test_each_exchange_re_reads_rather_than_serving_the_last_one() -> None:
    sim = _FakeSim()
    bridge = _bridge(sim)
    bridge.transport_exchange()

    sim.robot_root_states._root[0, 0] = 9.0
    sim.robot_root_states.reads = 0
    bridge.transport_exchange()

    assert sim.robot_root_states.reads == 1
    assert bridge.odom is not None
    assert bridge.odom[0][0] == pytest.approx(9.0)


def test_an_exchange_that_raises_leaves_no_stale_state_behind() -> None:
    sim = _FakeSim()
    bridge = _bridge(sim)
    bridge.fail_on_publish_odom = True

    with pytest.raises(RuntimeError, match="publish failed"):
        bridge.transport_exchange()

    # The next read is served from the simulator, not from the failed exchange's snapshot.
    sim.robot_root_states._root[0, 0] = 5.0
    sim.robot_root_states.reads = 0
    position, _quat, _lin, _yaw = bridge._get_base_odometry()
    assert sim.robot_root_states.reads == 1
    assert position[0] == pytest.approx(5.0)


def test_an_exchange_publishes_in_order_and_reads_the_command_last() -> None:
    """The order is the bridge's contract now, so pin it here rather than at the caller."""
    sim = _FakeSim()
    bridge = _bridge(sim)
    order: list[str] = []
    bridge.publish_low_state = lambda: order.append("low_state")  # type: ignore[method-assign]
    bridge.publish_odom = lambda: order.append("odom")  # type: ignore[method-assign]
    bridge.publish_wireless_controller = lambda: order.append("wireless")  # type: ignore[method-assign]
    bridge.low_cmd_handler = lambda _msg=None: order.append("command_read")  # type: ignore[method-assign]

    bridge.joystick = object()
    bridge.transport_exchange()

    assert order == ["low_state", "odom", "wireless", "command_read"]


def test_an_exchange_skips_odometry_and_the_joystick_when_not_configured() -> None:
    sim = _FakeSim()
    robot_config = SimpleNamespace(
        dof_effort_limit_list=[100.0] * _NUM_DOF,
        asset=SimpleNamespace(robot_type="g1_29dof"),
        bridge=RobotBridgeConfig(),
    )
    bridge = _StubBridge(
        cast("BaseSimulator", sim),
        cast("RobotConfig", robot_config),
        BridgeConfig(interface="lo", publish_odom=False),
    )
    published: list[str] = []
    bridge.publish_odom = lambda: published.append("odom")  # type: ignore[method-assign]
    bridge.publish_wireless_controller = lambda: published.append("wireless")  # type: ignore[method-assign]

    bridge.transport_exchange()

    assert published == []
