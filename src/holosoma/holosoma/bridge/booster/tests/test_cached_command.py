from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from holosoma.bridge.booster.booster_sdk2py_bridge import BoosterSdk2Bridge
from holosoma.config_types.simulator import BridgeConfig
from holosoma.config_values.robot import t1_29dof_waist_wrist
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

pytestmark = pytest.mark.no_sim


def test_motor_order_is_required_before_transport_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_transport(_bridge: BoosterSdk2Bridge) -> None:
        pytest.fail("Transport must not start before motor order is validated")

    monkeypatch.setattr(BoosterSdk2Bridge, "_init_sdk_components", unexpected_transport)
    robot = replace(
        t1_29dof_waist_wrist,
        bridge=replace(t1_29dof_waist_wrist.bridge, controlled_dof_names=()),
    )
    with pytest.raises(ValueError, match="requires bridge.controlled_dof_names in SDK motor order"):
        BoosterSdk2Bridge(cast("BaseSimulator", SimpleNamespace()), robot, BridgeConfig(enabled=False))


def test_subscriber_command_is_sampled_on_transport_tick() -> None:
    active_command = SimpleNamespace(motor_cmd=[])
    incoming_command = SimpleNamespace(motor_cmd=[])

    bridge = BoosterSdk2Bridge.__new__(BoosterSdk2Bridge)
    bridge.low_cmd = active_command
    bridge._pending_low_cmd = active_command

    bridge._on_low_cmd(incoming_command)
    assert bridge.low_cmd is active_command

    bridge.low_cmd_handler()
    assert bridge.low_cmd is incoming_command


def test_compute_torques_reuses_cached_command_with_fresh_simulator_state() -> None:
    simulator = SimpleNamespace(
        dof_pos=torch.zeros(1, 1),
        dof_vel=torch.zeros(1, 1),
    )
    motor_command = SimpleNamespace(tau=1.0, kp=2.0, kd=3.0, q=4.0, dq=5.0)
    cached_command = SimpleNamespace(motor_cmd=[motor_command])

    bridge = BoosterSdk2Bridge.__new__(BoosterSdk2Bridge)
    bridge.simulator = cast("BaseSimulator", simulator)
    bridge.num_motor = 1
    bridge.dof_indices = [0]
    bridge.torque_limit = np.array([100.0])
    bridge.torques = np.zeros(1)
    bridge.low_cmd = cached_command

    first = bridge.compute_torques()
    np.testing.assert_allclose(first, [24.0])

    # Recompute without ingesting a command. The cached target/gains are unchanged, but current
    # simulator feedback changes the torque.
    simulator.dof_pos[:] = 0.5
    simulator.dof_vel[:] = 0.25
    second = bridge.compute_torques()

    assert bridge.low_cmd is cached_command
    np.testing.assert_allclose(second, [22.25])
