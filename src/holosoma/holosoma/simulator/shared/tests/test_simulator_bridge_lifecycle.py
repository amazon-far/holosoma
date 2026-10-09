from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

from holosoma.config_types.simulator import BridgeConfig
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.shared import simulator_bridge
from holosoma.simulator.shared.simulator_bridge import SimulatorBridge

pytestmark = pytest.mark.no_sim


def _simulator() -> BaseSimulator:
    simulator = SimpleNamespace(
        apply_torques_at_dof=MagicMock(),
        device="cpu",
        hooks=HookRegistry(base_rates={Phase.PRE_STEP: 200.0}),
        robot_config=SimpleNamespace(bridge=SimpleNamespace(sdk_type="test")),
        time=MagicMock(return_value=0.0),
    )
    return cast("BaseSimulator", simulator)


def _robot_bridge() -> MagicMock:
    robot_bridge = MagicMock()
    robot_bridge._apply_indices = None
    robot_bridge.joystick = None
    robot_bridge.torques = np.zeros(1, dtype=np.float32)
    return robot_bridge


def test_bridge_constructor_starts_transport_and_owns_its_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    robot_bridge = _robot_bridge()
    factory = MagicMock(return_value=robot_bridge)
    clock = MagicMock()
    monkeypatch.setattr(simulator_bridge, "create_sdk2py_bridge", factory)
    monkeypatch.setattr(simulator_bridge, "ClockPub", lambda: clock)
    simulator = _simulator()

    bridge = SimulatorBridge(simulator, BridgeConfig(enabled=True, interface="lo"))
    bridge.register_hooks(simulator.hooks)
    simulator.hooks.emit(Phase.PRE_STEP)
    simulator.hooks.emit(Phase.CLOSE)

    factory.assert_called_once()
    clock.start.assert_called_once_with()
    robot_bridge.publish_low_state.assert_called_once_with()
    robot_bridge.close.assert_called_once_with()
    clock.close.assert_called_once_with()


def test_bridge_default_runs_transport_and_control_every_physics_step(monkeypatch: pytest.MonkeyPatch) -> None:
    robot_bridge = _robot_bridge()
    clock = MagicMock()
    monkeypatch.setattr(simulator_bridge, "create_sdk2py_bridge", lambda *_args: robot_bridge)
    monkeypatch.setattr(simulator_bridge, "ClockPub", lambda: clock)
    simulator = _simulator()

    SimulatorBridge(simulator, BridgeConfig(enabled=True, interface="lo"))
    for _ in range(8):
        simulator.hooks.emit(Phase.PRE_STEP)

    assert robot_bridge.publish_low_state.call_count == 8
    assert robot_bridge.publish_odom.call_count == 8
    assert robot_bridge.low_cmd_handler.call_count == 8
    assert robot_bridge.compute_torques.call_count == 8
    assert cast("MagicMock", simulator.apply_torques_at_dof).call_count == 8
    assert clock.publish.call_count == 8


def test_bridge_transport_decimation_keeps_physics_rate_control_and_cached_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot_bridge = _robot_bridge()
    robot_bridge.joystick = object()
    clock = MagicMock()
    cached_command = 0
    commands_seen_by_control: list[int] = []
    incoming_commands = iter([1, 2])

    def ingest_command() -> None:
        nonlocal cached_command
        cached_command = next(incoming_commands)

    def compute_torques() -> npt.NDArray[np.float32]:
        commands_seen_by_control.append(cached_command)
        return np.array([cached_command], dtype=np.float32)

    robot_bridge.low_cmd_handler.side_effect = ingest_command
    robot_bridge.compute_torques.side_effect = compute_torques
    monkeypatch.setattr(simulator_bridge, "create_sdk2py_bridge", lambda *_args: robot_bridge)
    monkeypatch.setattr(simulator_bridge, "ClockPub", lambda: clock)
    simulator = _simulator()

    SimulatorBridge(
        simulator,
        BridgeConfig(enabled=True, interface="lo", transport_decimation="50Hz"),
    )
    for _ in range(8):
        simulator.hooks.emit(Phase.PRE_STEP)

    # 200Hz physics / 50Hz transport: SDK I/O fires on the 4th and 8th emissions.
    assert robot_bridge.publish_low_state.call_count == 2
    assert robot_bridge.publish_odom.call_count == 2
    assert robot_bridge.publish_wireless_controller.call_count == 2
    assert robot_bridge.low_cmd_handler.call_count == 2
    assert clock.publish.call_count == 2
    # Dynamics-facing feedback remains at 200Hz and reuses each command until the next poll.
    assert robot_bridge.compute_torques.call_count == 8
    assert cast("MagicMock", simulator.apply_torques_at_dof).call_count == 8
    assert commands_seen_by_control == [0, 0, 0, 1, 1, 1, 1, 2]


def test_bridge_start_failure_closes_acquired_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    robot_bridge = MagicMock()
    robot_bridge.setup_joystick.side_effect = RuntimeError("joystick failed")
    clock = MagicMock()
    monkeypatch.setattr(simulator_bridge, "create_sdk2py_bridge", lambda *_args: robot_bridge)
    monkeypatch.setattr(simulator_bridge, "ClockPub", lambda: clock)

    with pytest.raises(RuntimeError, match="joystick failed"):
        SimulatorBridge(
            _simulator(),
            BridgeConfig(enabled=True, interface="lo", use_joystick=True),
        )

    robot_bridge.close.assert_called_once_with()
    clock.close.assert_called_once_with()


def test_bridge_close_attempts_clock_after_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    robot_bridge = MagicMock()
    robot_bridge.close.side_effect = RuntimeError("transport failed")
    clock = MagicMock()
    monkeypatch.setattr(simulator_bridge, "create_sdk2py_bridge", lambda *_args: robot_bridge)
    monkeypatch.setattr(simulator_bridge, "ClockPub", lambda: clock)
    bridge = SimulatorBridge(_simulator(), BridgeConfig(enabled=True, interface="lo"))

    with pytest.raises(RuntimeError, match="transport failed"):
        bridge.close()

    robot_bridge.close.assert_called_once_with()
    clock.close.assert_called_once_with()

    bridge.close()
    robot_bridge.close.assert_called_once_with()
    clock.close.assert_called_once_with()


def test_explicit_dds_config_survives_auto_interface_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    xml = ' \n<CycloneDDS><Domain Id="0"/></CycloneDDS>\n'
    config = BridgeConfig(enabled=True, dds_config=xml)
    factory = MagicMock()
    clock = MagicMock()
    monkeypatch.setattr(simulator_bridge, "create_sdk2py_bridge", factory)
    monkeypatch.setattr(simulator_bridge, "ClockPub", lambda: clock)
    bridge = SimulatorBridge(_simulator(), config)
    try:
        forwarded = factory.call_args.args[2]
        assert forwarded.dds_config == xml
        assert forwarded.interface in ("lo", "lo0")
        assert config.interface is None
    finally:
        bridge.close()


def test_dds_startup_failure_closes_clock_and_leaves_close_hook_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = MagicMock(side_effect=RuntimeError("DDS capability check failed"))
    clock = MagicMock()
    monkeypatch.setattr(simulator_bridge, "create_sdk2py_bridge", factory)
    monkeypatch.setattr(simulator_bridge, "ClockPub", lambda: clock)
    simulator = _simulator()
    with pytest.raises(RuntimeError, match="DDS capability check failed"):
        SimulatorBridge(simulator, BridgeConfig(enabled=True, dds_config='<CycloneDDS><Domain Id="0"/></CycloneDDS>'))
    clock.start.assert_not_called()
    clock.close.assert_called_once_with()
    simulator.hooks.emit(Phase.CLOSE)
    clock.close.assert_called_once_with()
