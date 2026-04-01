"""Unit tests for input providers.

Tests cover:
- Command enums and device mappings
- Keyboard providers: queue-based poll, key-to-command mapping, velocity tracking
- InterfaceInput: merged velocity+command device, edge detection, protocol conformance
- ROS2 providers: callback-to-command mapping, velocity clamping
- Factory methods on BasePolicy / LocomotionPolicy / WBT
- DualMode command intercept and switching
- _apply_velocity hook
"""

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from holosoma_inference.inputs import create_input
from holosoma_inference.inputs.api.base import StateCommandProvider, VelCmdProvider
from holosoma_inference.inputs.api.commands import StateCommand, VelCmd
from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS
from holosoma_inference.inputs.impl.keyboard import (
    KEYBOARD_COMMANDS,
    KEYBOARD_VELOCITY_LOCOMOTION,
)
from holosoma_inference.inputs.impl.ros2 import ROS2_COMMAND_MAP

# ---------------------------------------------------------------------------
# Fixtures: lightweight mock policy / interface objects
# ---------------------------------------------------------------------------


def _make_interface(**overrides):
    """Build a minimal mock interface with joystick methods."""
    iface = MagicMock()
    iface.get_joystick_msg.return_value = None
    iface.get_joystick_key.return_value = ""
    for k, v in overrides.items():
        setattr(iface, k, v)
    return iface


def _make_policy(**overrides):
    """Build a minimal mock policy with all attributes providers touch."""
    p = MagicMock()
    p.lin_vel_command = np.array([[0.0, 0.0]])
    p.ang_vel_command = np.array([[0.0]])
    p.stand_command = np.array([[0]])
    p.base_height_command = np.array([[0.5]])
    p.desired_base_height = 0.5
    p.active_policy_index = 0
    p.model_paths = ["a.onnx", "b.onnx"]
    p.interface = _make_interface()
    p.config = SimpleNamespace(
        task=SimpleNamespace(
            ros_cmd_vel_topic="cmd_vel",
            ros_state_input_topic="holosoma/state_input",
            ros_vel_timeout=1.0,
        )
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


@pytest.fixture
def policy():
    return _make_policy()


@pytest.fixture
def interface():
    return _make_interface()


# ============================================================================
# Command enum and mapping tests
# ============================================================================


class TestCommandMappings:
    def test_joystick_has_core_commands(self):
        assert JOYSTICK_COMMANDS["A"] == StateCommand.START
        assert JOYSTICK_COMMANDS["B"] == StateCommand.STOP
        assert JOYSTICK_COMMANDS["Y"] == StateCommand.INIT
        assert JOYSTICK_COMMANDS["L1+R1"] == StateCommand.KILL
        assert JOYSTICK_COMMANDS["select"] == StateCommand.NEXT_POLICY

    def test_joystick_has_locomotion_commands(self):
        assert JOYSTICK_COMMANDS["start"] == StateCommand.STAND_TOGGLE
        assert JOYSTICK_COMMANDS["L2"] == StateCommand.ZERO_VELOCITY

    def test_joystick_has_wbt_commands(self):
        assert JOYSTICK_COMMANDS["select+A"] == StateCommand.START_MOTION_CLIP

    def test_keyboard_has_core_commands(self):
        assert KEYBOARD_COMMANDS["]"] == StateCommand.START
        assert KEYBOARD_COMMANDS["o"] == StateCommand.STOP
        assert KEYBOARD_COMMANDS["i"] == StateCommand.INIT
        assert KEYBOARD_COMMANDS["1"] == StateCommand.SWITCH_POLICY_1
        assert KEYBOARD_COMMANDS["9"] == StateCommand.SWITCH_POLICY_9

    def test_keyboard_has_kp_commands(self):
        assert KEYBOARD_COMMANDS["v"] == StateCommand.KP_DOWN_FINE
        assert KEYBOARD_COMMANDS["b"] == StateCommand.KP_UP_FINE
        assert KEYBOARD_COMMANDS["f"] == StateCommand.KP_DOWN
        assert KEYBOARD_COMMANDS["g"] == StateCommand.KP_UP
        assert KEYBOARD_COMMANDS["r"] == StateCommand.KP_RESET

    def test_keyboard_has_locomotion_commands(self):
        assert KEYBOARD_COMMANDS["="] == StateCommand.STAND_TOGGLE
        assert KEYBOARD_COMMANDS["z"] == StateCommand.ZERO_VELOCITY

    def test_keyboard_has_wbt_commands(self):
        assert KEYBOARD_COMMANDS["m"] == StateCommand.START_MOTION_CLIP

    def test_keyboard_no_velocity_keys_in_command_mapping(self):
        """Velocity keys are in KEYBOARD_VELOCITY_LOCOMOTION, not the command mapping."""
        for key in ("w", "s", "a", "d", "q", "e"):
            assert key not in KEYBOARD_COMMANDS

    def test_keyboard_velocity_locomotion_mapping(self):
        """KEYBOARD_VELOCITY_LOCOMOTION maps WASD/QE to (array_idx, col, delta)."""
        assert KEYBOARD_VELOCITY_LOCOMOTION["w"] == (0, 0, +0.1)
        assert KEYBOARD_VELOCITY_LOCOMOTION["s"] == (0, 0, -0.1)
        assert KEYBOARD_VELOCITY_LOCOMOTION["a"] == (0, 1, +0.1)
        assert KEYBOARD_VELOCITY_LOCOMOTION["d"] == (0, 1, -0.1)
        assert KEYBOARD_VELOCITY_LOCOMOTION["q"] == (1, 0, -0.1)
        assert KEYBOARD_VELOCITY_LOCOMOTION["e"] == (1, 0, +0.1)

    def test_ros2_command_map(self):
        assert ROS2_COMMAND_MAP["start"] == StateCommand.START
        assert ROS2_COMMAND_MAP["stop"] == StateCommand.STOP
        assert ROS2_COMMAND_MAP["init"] == StateCommand.INIT
        assert ROS2_COMMAND_MAP["walk"] == StateCommand.WALK
        assert ROS2_COMMAND_MAP["stand"] == StateCommand.STAND


# ============================================================================
# Protocol conformance
# ============================================================================


class TestProtocolConformance:
    def test_interface_input_satisfies_both_protocols(self):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        device = InterfaceInput(_make_interface())
        assert isinstance(device, VelCmdProvider)
        assert isinstance(device, StateCommandProvider)

    def test_keyboard_input_satisfies_both_protocols(self):
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput

        device = KeyboardInput(deque(), KEYBOARD_VELOCITY_LOCOMOTION)
        assert isinstance(device, VelCmdProvider)
        assert isinstance(device, StateCommandProvider)

    def test_keyboard_input_no_velocity_satisfies_both(self):
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput

        device = KeyboardInput(deque())
        assert isinstance(device, VelCmdProvider)
        assert isinstance(device, StateCommandProvider)

    def test_ros2_input_satisfies_both_protocols(self):
        from holosoma_inference.inputs.impl.ros2 import Ros2Input

        prov = Ros2Input("cmd_vel", "holosoma/state_input")
        assert isinstance(prov, VelCmdProvider)
        assert isinstance(prov, StateCommandProvider)


# ============================================================================
# Keyboard providers
# ============================================================================


class TestKeyboardListener:
    def test_start_is_idempotent(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import _KeyboardListenerThread

        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        listener = _KeyboardListenerThread()
        assert listener.start() is False
        assert listener.start() is False  # second call should be a no-op

    def test_returns_false_in_non_tty(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import _KeyboardListenerThread

        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        listener = _KeyboardListenerThread()
        assert listener.start() is False

    def test_get_keyboard_listener_returns_singleton(self):
        from holosoma_inference.inputs.impl import keyboard as kb_mod
        from holosoma_inference.inputs.impl.keyboard import _KeyboardListenerThread, get_keyboard_listener

        old = kb_mod._listener
        try:
            kb_mod._listener = None
            first = get_keyboard_listener()
            assert isinstance(first, _KeyboardListenerThread)
            assert get_keyboard_listener() is first
        finally:
            kb_mod._listener = old

    def test_broadcast_to_multiple_subscribers(self):
        from holosoma_inference.inputs.impl.keyboard import _KeyboardListenerThread

        listener = _KeyboardListenerThread()
        q1 = listener.subscribe()
        q2 = listener.subscribe()

        for q in listener._subscribers:
            q.append("w")

        assert list(q1) == ["w"]
        assert list(q2) == ["w"]

        q1.popleft()
        assert len(q1) == 0
        assert len(q2) == 1


class TestKeyboardInput:
    def _make(self, velocity_keys=None):
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput

        queue = deque()
        return KeyboardInput(queue, velocity_keys)

    # --- Command tests ---

    def test_poll_returns_mapped_commands(self):
        dev = self._make()
        dev._queue.extend(["]", "o", "i"])
        dev.poll_velocity()  # drains queue, buffers commands
        assert dev.poll_commands() == [StateCommand.START, StateCommand.STOP, StateCommand.INIT]

    def test_poll_skips_unmapped_keys(self):
        dev = self._make()
        dev._queue.extend(["unknown", "nope", "]"])
        dev.poll_velocity()
        assert dev.poll_commands() == [StateCommand.START]

    def test_poll_drains_queue(self):
        dev = self._make()
        dev._queue.append("]")
        dev.poll_velocity()
        assert dev.poll_commands() == [StateCommand.START]
        dev.poll_velocity()
        assert dev.poll_commands() == []

    def test_poll_kp_commands(self):
        dev = self._make()
        dev._queue.extend(["v", "b", "f", "g", "r"])
        dev.poll_velocity()
        assert dev.poll_commands() == [
            StateCommand.KP_DOWN_FINE,
            StateCommand.KP_UP_FINE,
            StateCommand.KP_DOWN,
            StateCommand.KP_UP,
            StateCommand.KP_RESET,
        ]

    def test_poll_switch_policy(self):
        dev = self._make()
        dev._queue.append("2")
        dev.poll_velocity()
        assert dev.poll_commands() == [StateCommand.SWITCH_POLICY_2]

    def test_poll_empty_queue_returns_empty(self):
        dev = self._make()
        dev.poll_velocity()
        assert dev.poll_commands() == []

    def test_locomotion_mapping(self):
        dev = self._make()
        dev._queue.extend(["=", "]", "z"])
        dev.poll_velocity()
        assert dev.poll_commands() == [
            StateCommand.STAND_TOGGLE,
            StateCommand.START,
            StateCommand.ZERO_VELOCITY,
        ]

    def test_wbt_mapping(self):
        dev = self._make()
        dev._queue.extend(["m", "o"])
        dev.poll_velocity()
        assert dev.poll_commands() == [StateCommand.START_MOTION_CLIP, StateCommand.STOP]

    # --- Velocity tests ---

    def test_poll_returns_velocity_command(self):
        dev = self._make(velocity_keys=KEYBOARD_VELOCITY_LOCOMOTION)
        dev._queue.append("w")
        vc = dev.poll_velocity()
        assert isinstance(vc, VelCmd)
        assert pytest.approx(vc.lin_vel[0]) == 0.1
        assert pytest.approx(vc.lin_vel[1]) == 0.0
        assert pytest.approx(vc.ang_vel) == 0.0

    def test_poll_accumulates_increments(self):
        dev = self._make(velocity_keys=KEYBOARD_VELOCITY_LOCOMOTION)
        dev._queue.extend(["w", "w", "a"])
        vc = dev.poll_velocity()
        assert pytest.approx(vc.lin_vel[0]) == 0.2
        assert pytest.approx(vc.lin_vel[1]) == 0.1

    def test_poll_angular_velocity(self):
        dev = self._make(velocity_keys=KEYBOARD_VELOCITY_LOCOMOTION)
        dev._queue.extend(["q", "e", "e"])
        vc = dev.poll_velocity()
        assert pytest.approx(vc.ang_vel) == 0.1

    def test_poll_returns_snapshot(self):
        dev = self._make(velocity_keys=KEYBOARD_VELOCITY_LOCOMOTION)
        dev._queue.append("w")
        vc1 = dev.poll_velocity()
        dev._queue.append("w")
        vc2 = dev.poll_velocity()
        assert pytest.approx(vc1.lin_vel[0]) == 0.1
        assert pytest.approx(vc2.lin_vel[0]) == 0.2

    def test_zero_resets_state(self):
        dev = self._make(velocity_keys=KEYBOARD_VELOCITY_LOCOMOTION)
        dev._queue.extend(["w", "a", "e"])
        dev.poll_velocity()
        dev.zero()
        vc = dev.poll_velocity()
        assert pytest.approx(vc.lin_vel[0]) == 0.0
        assert pytest.approx(vc.lin_vel[1]) == 0.0
        assert pytest.approx(vc.ang_vel) == 0.0

    def test_no_velocity_keys_returns_none(self):
        dev = self._make()
        dev._queue.append("w")
        assert dev.poll_velocity() is None
        assert len(dev._queue) == 0  # queue drained even without velocity keys

    def test_velocity_and_commands_from_same_queue(self):
        """Velocity keys and command keys are processed from a single drain."""
        dev = self._make(KEYBOARD_VELOCITY_LOCOMOTION)
        dev._queue.extend(["w", "]", "a", "="])
        vc = dev.poll_velocity()
        assert pytest.approx(vc.lin_vel[0]) == 0.1
        assert pytest.approx(vc.lin_vel[1]) == 0.1
        assert dev.poll_commands() == [StateCommand.START, StateCommand.STAND_TOGGLE]

    def test_same_object_for_both_slots(self):
        """A single KeyboardInput assigned to both slots works correctly."""
        dev = self._make(KEYBOARD_VELOCITY_LOCOMOTION)
        dev._queue.extend(["w", "]"])

        vc = dev.poll_velocity()
        commands = dev.poll_commands()

        assert isinstance(vc, VelCmd)
        assert pytest.approx(vc.lin_vel[0]) == 0.1
        assert commands == [StateCommand.START]

    def test_broadcast_isolation(self):
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput

        q1 = deque()
        q2 = deque()
        dev = KeyboardInput(q1, KEYBOARD_VELOCITY_LOCOMOTION)

        q1.append("w")
        q2.append("w")

        vc = dev.poll_velocity()
        assert pytest.approx(vc.lin_vel[0]) == 0.1
        assert len(q2) == 1


# ============================================================================
# InterfaceInput (merged velocity + commands device)
# ============================================================================


def _joystick_msg(lx=0.0, ly=0.0, rx=0.0, keys=0):
    """Create a mock joystick message with stick and button data."""
    return SimpleNamespace(lx=lx, ly=ly, rx=rx, keys=keys)


class TestInterfaceInput:
    def test_poll_velocity_returns_none_when_no_msg(self, interface):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        interface.get_joystick_msg.return_value = None
        device = InterfaceInput(interface)
        assert device.poll_velocity() is None

    def test_poll_velocity_returns_velocity_command(self, interface):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        interface.get_joystick_msg.return_value = _joystick_msg(lx=0.0, ly=0.5, rx=-0.2)
        interface.get_joystick_key.return_value = ""

        device = InterfaceInput(interface)
        vc = device.poll_velocity()

        assert isinstance(vc, VelCmd)
        assert vc.lin_vel == pytest.approx((0.5, 0.0))
        assert vc.ang_vel == pytest.approx(0.2)

    def test_poll_velocity_applies_deadzone(self, interface):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        interface.get_joystick_msg.return_value = _joystick_msg(lx=0.05, ly=0.09, rx=0.03)
        interface.get_joystick_key.return_value = ""

        device = InterfaceInput(interface)
        vc = device.poll_velocity()

        assert vc.lin_vel == (0.0, 0.0)
        assert vc.ang_vel == 0.0

    def test_poll_velocity_returns_none_during_button_press(self, interface):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        interface.get_joystick_msg.return_value = _joystick_msg(ly=0.5, keys=256)
        interface.get_joystick_key.return_value = "A"

        device = InterfaceInput(interface)
        vc = device.poll_velocity()

        assert vc is None

    def test_poll_velocity_no_stand_command_gating(self, interface):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        interface.get_joystick_msg.return_value = _joystick_msg(ly=0.8, rx=-0.3)
        interface.get_joystick_key.return_value = ""

        device = InterfaceInput(interface)
        vc = device.poll_velocity()

        assert vc.lin_vel[0] == pytest.approx(0.8)
        assert vc.ang_vel == pytest.approx(0.3)

    def test_poll_velocity_stick_axis_mapping(self, interface):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        interface.get_joystick_msg.return_value = _joystick_msg(lx=0.5, ly=0.3, rx=0.7)
        interface.get_joystick_key.return_value = ""

        device = InterfaceInput(interface)
        vc = device.poll_velocity()

        assert vc.lin_vel[0] == pytest.approx(0.3)  # ly
        assert vc.lin_vel[1] == pytest.approx(-0.5)  # -lx
        assert vc.ang_vel == pytest.approx(-0.7)  # -rx

    def test_poll_commands_rising_edge(self, interface):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        interface.get_joystick_msg.return_value = _joystick_msg(keys=256)
        interface.get_joystick_key.return_value = "A"

        device = InterfaceInput(interface)
        commands = device.poll_commands()
        assert commands == [StateCommand.START]

    def test_poll_commands_no_dispatch_on_hold(self, interface):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        interface.get_joystick_msg.return_value = _joystick_msg(keys=256)
        interface.get_joystick_key.return_value = "A"

        device = InterfaceInput(interface)
        device.poll_commands()  # first press

        # Second poll with same button held — no new rising edge
        assert device.poll_commands() == []

    def test_poll_commands_unmapped_ignored(self, interface):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        interface.get_joystick_msg.return_value = _joystick_msg(keys=999)
        interface.get_joystick_key.return_value = "UNKNOWN"

        device = InterfaceInput(interface)
        assert device.poll_commands() == []

    def test_poll_velocity_then_commands(self, interface):
        """poll_velocity reads sticks, poll_commands reads buttons independently."""
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        interface.get_joystick_msg.return_value = _joystick_msg(ly=0.5, keys=256)
        interface.get_joystick_key.return_value = "A"

        device = InterfaceInput(interface)

        # poll_velocity returns None (buttons pressed suppress sticks)
        vc = device.poll_velocity()
        assert vc is None

        # poll_commands reads joystick independently — "A" is a rising edge
        commands = device.poll_commands()
        assert commands == [StateCommand.START]

    def test_same_object_for_both_slots(self, interface):
        """A single InterfaceInput assigned to both slots works correctly."""
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        device = InterfaceInput(interface)

        # Simulate run loop: velocity first, then commands
        interface.get_joystick_msg.return_value = _joystick_msg(ly=0.5, keys=0)
        interface.get_joystick_key.return_value = ""

        vc = device.poll_velocity()
        commands = device.poll_commands()

        assert isinstance(vc, VelCmd)
        assert commands == []  # no buttons pressed


# ============================================================================
# ROS2 providers
# ============================================================================


class TestRos2Input:
    def _make(self):
        from holosoma_inference.inputs.impl.ros2 import Ros2Input

        return Ros2Input("cmd_vel", "holosoma/state_input")

    # --- Velocity tests ---

    def test_vel_callback_stores_velocity(self):
        prov = self._make()
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.5, y=-0.3),
                angular=SimpleNamespace(z=0.8),
            )
        )
        prov._vel_callback(msg)
        vc = prov.poll_velocity()

        assert isinstance(vc, VelCmd)
        assert vc.lin_vel == pytest.approx((0.5, -0.3))
        assert vc.ang_vel == pytest.approx(0.8)

    def test_vel_callback_clamps_to_range(self):
        prov = self._make()
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=5.0, y=-5.0),
                angular=SimpleNamespace(z=99.0),
            )
        )
        prov._vel_callback(msg)
        vc = prov.poll_velocity()

        assert vc.lin_vel == (1.0, -1.0)
        assert vc.ang_vel == 1.0

    def test_frozen_values_stable(self):
        prov = self._make()
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.5, y=0.0),
                angular=SimpleNamespace(z=0.0),
            )
        )
        prov._vel_callback(msg)
        vc1 = prov.poll_velocity()
        vc2 = prov.poll_velocity()
        assert vc1.lin_vel == vc2.lin_vel
        assert vc1.ang_vel == vc2.ang_vel

    def test_vel_callback_clamps_negative_angular(self):
        prov = self._make()
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.0, y=0.0),
                angular=SimpleNamespace(z=-99.0),
            )
        )
        prov._vel_callback(msg)
        assert prov.poll_velocity().ang_vel == -1.0

    def test_vel_callback_exact_boundary_values(self):
        prov = self._make()
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=1.0, y=-1.0),
                angular=SimpleNamespace(z=1.0),
            )
        )
        prov._vel_callback(msg)
        vc = prov.poll_velocity()
        assert vc.lin_vel == (1.0, -1.0)
        assert vc.ang_vel == 1.0

    def test_vel_callback_zero_passes_through(self):
        prov = self._make()
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.0, y=0.0),
                angular=SimpleNamespace(z=0.0),
            )
        )
        prov._vel_callback(msg)
        vc = prov.poll_velocity()
        assert vc.lin_vel == (0.0, 0.0)
        assert vc.ang_vel == 0.0

    def test_zero_resets_velocity(self):
        prov = self._make()
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.5, y=0.3),
                angular=SimpleNamespace(z=0.1),
            )
        )
        prov._vel_callback(msg)
        prov.zero()
        vc = prov.poll_velocity()
        assert vc.lin_vel == (0.0, 0.0)
        assert vc.ang_vel == 0.0

    # --- Command tests ---

    def test_known_commands_queued(self):
        prov = self._make()
        prov._cmd_callback(SimpleNamespace(data="start"))
        prov._cmd_callback(SimpleNamespace(data="stop"))
        prov._cmd_callback(SimpleNamespace(data="init"))

        assert prov.poll_commands() == [StateCommand.START, StateCommand.STOP, StateCommand.INIT]

    def test_walk_stand_commands(self):
        prov = self._make()
        prov._cmd_callback(SimpleNamespace(data="walk"))
        prov._cmd_callback(SimpleNamespace(data="stand"))

        assert prov.poll_commands() == [StateCommand.WALK, StateCommand.STAND]

    def test_unknown_command_warns(self, capfd):
        from io import StringIO

        from loguru import logger

        sink = StringIO()
        handler_id = logger.add(sink, format="{message}", level="WARNING")
        try:
            prov = self._make()
            prov._cmd_callback(SimpleNamespace(data="bogus"))
            assert prov.poll_commands() == []
            assert "bogus" in sink.getvalue()
        finally:
            logger.remove(handler_id)

    def test_whitespace_and_case_normalization(self):
        prov = self._make()
        prov._cmd_callback(SimpleNamespace(data="  WALK  "))

        assert prov.poll_commands() == [StateCommand.WALK]

    def test_poll_drains_queue(self):
        prov = self._make()
        prov._cmd_callback(SimpleNamespace(data="start"))
        assert prov.poll_commands() == [StateCommand.START]
        assert prov.poll_commands() == []

    def test_empty_string_warns(self):
        from io import StringIO

        from loguru import logger

        sink = StringIO()
        handler_id = logger.add(sink, format="{message}", level="WARNING")
        try:
            prov = self._make()
            prov._cmd_callback(SimpleNamespace(data="   "))
            assert prov.poll_commands() == []
            assert "unknown command" in sink.getvalue().lower()
        finally:
            logger.remove(handler_id)

    # --- Combined tests ---

    def test_same_object_for_both_roles(self):
        """A single Ros2Input handles velocity and commands simultaneously."""
        prov = self._make()
        prov._vel_callback(
            SimpleNamespace(
                twist=SimpleNamespace(
                    linear=SimpleNamespace(x=0.5, y=0.0),
                    angular=SimpleNamespace(z=0.0),
                )
            )
        )
        prov._cmd_callback(SimpleNamespace(data="start"))

        vc = prov.poll_velocity()
        cmds = prov.poll_commands()

        assert vc.lin_vel[0] == pytest.approx(0.5)
        assert cmds == [StateCommand.START]


# ============================================================================
# Factory methods (BasePolicy / Locomotion / WBT)
# ============================================================================


def _try_import_policies():
    try:
        from holosoma_inference.policies.base import BasePolicy  # noqa: F401
        from holosoma_inference.policies.locomotion import LocomotionPolicy  # noqa: F401
        from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


_has_policies = _try_import_policies()
_skip_policies = pytest.mark.skipif(not _has_policies, reason="Policy deps not installed")


@_skip_policies
class TestCreateInputFactory:
    def _make_policy_for_factory(self, monkeypatch=None, **overrides):
        p = _make_policy()
        p.use_joystick = overrides.pop("use_joystick", True)
        del p._shared_hardware_source
        if monkeypatch is not None:
            p.logger = MagicMock()
        for k, v in overrides.items():
            setattr(p, k, v)
        return p

    def test_keyboard_returns_keyboard_input(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput

        p = self._make_policy_for_factory(monkeypatch)
        result = create_input(p, "keyboard", "velocity")
        assert isinstance(result, KeyboardInput)
        assert result._mapping == KEYBOARD_COMMANDS

    def test_interface_returns_interface_input(self):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        p = self._make_policy_for_factory()
        result = create_input(p, "interface", "velocity")
        assert isinstance(result, InterfaceInput)
        assert result._mapping == JOYSTICK_COMMANDS

    def test_joystick_maps_to_interface(self):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        p = self._make_policy_for_factory()
        result = create_input(p, "joystick", "velocity")
        assert isinstance(result, InterfaceInput)

    def test_ros2_returns_ros2_input(self):
        from holosoma_inference.inputs.impl.ros2 import Ros2Input

        p = self._make_policy_for_factory()
        result = create_input(p, "ros2", "velocity")
        assert isinstance(result, Ros2Input)

    def test_ros2_command_returns_ros2_input(self):
        from holosoma_inference.inputs.impl.ros2 import Ros2Input

        p = self._make_policy_for_factory()
        result = create_input(p, "ros2", "command")
        assert isinstance(result, Ros2Input)

    def test_unknown_source_raises(self):
        p = self._make_policy_for_factory()
        with pytest.raises(ValueError, match="Unknown input source"):
            create_input(p, "invalid", "velocity")

    def test_joystick_fallback_to_keyboard(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput

        p = self._make_policy_for_factory(monkeypatch, use_joystick=False)
        result = create_input(p, "interface", "velocity")
        assert isinstance(result, KeyboardInput)

    def test_both_same_source_shares_object(self):
        """When both channels use the same source, _create_input_providers shares one object."""
        from holosoma_inference.inputs.impl.interface import InterfaceInput
        from holosoma_inference.policies.base import BasePolicy

        p = self._make_policy_for_factory()
        p.config = SimpleNamespace(task=SimpleNamespace(velocity_input="interface", state_input="interface"))
        BasePolicy._create_input_providers(p)
        assert p._velocity_input is p._command_provider
        assert isinstance(p._velocity_input, InterfaceInput)

    def test_both_keyboard_shares_object(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput
        from holosoma_inference.policies.base import BasePolicy

        p = self._make_policy_for_factory(monkeypatch, use_joystick=False)
        p.config = SimpleNamespace(task=SimpleNamespace(velocity_input="keyboard", state_input="keyboard"))
        BasePolicy._create_input_providers(p)
        assert p._velocity_input is p._command_provider
        assert isinstance(p._velocity_input, KeyboardInput)

    def test_velocity_role_gets_velocity_keys(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput

        p = self._make_policy_for_factory(monkeypatch)
        result = create_input(p, "keyboard", "velocity")
        assert type(result) is KeyboardInput
        assert result._velocity_keys is KEYBOARD_VELOCITY_LOCOMOTION

    def test_command_role_gets_no_velocity_keys(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput

        p = self._make_policy_for_factory(monkeypatch)
        result = create_input(p, "keyboard", "command")
        assert type(result) is KeyboardInput
        assert result._velocity_keys is None


# ============================================================================
# _apply_velocity hook
# ============================================================================


@_skip_policies
class TestApplyVelocity:
    def test_base_policy_passthrough(self):
        from holosoma_inference.policies.base import BasePolicy

        bp = BasePolicy.__new__(BasePolicy)
        bp.lin_vel_command = np.array([[0.0, 0.0]])
        bp.ang_vel_command = np.array([[0.0]])

        bp._apply_velocity(VelCmd((0.5, -0.3), 0.8))

        assert bp.lin_vel_command[0, 0] == pytest.approx(0.5)
        assert bp.lin_vel_command[0, 1] == pytest.approx(-0.3)
        assert bp.ang_vel_command[0, 0] == pytest.approx(0.8)

    def test_locomotion_gates_by_stand_command(self):
        from types import SimpleNamespace

        from holosoma_inference.policies.locomotion import LocomotionPolicy

        lp = LocomotionPolicy.__new__(LocomotionPolicy)
        lp.config = SimpleNamespace(task=SimpleNamespace(auto_walk_on_vel_cmd=False))
        lp.lin_vel_command = np.array([[0.0, 0.0]])
        lp.ang_vel_command = np.array([[0.0]])

        lp.stand_command = np.array([[0]])
        lp._apply_velocity(VelCmd((0.5, -0.3), 0.8))
        assert lp.lin_vel_command[0, 0] == pytest.approx(0.0)
        assert lp.ang_vel_command[0, 0] == pytest.approx(0.0)

        lp.stand_command = np.array([[1]])
        lp._apply_velocity(VelCmd((0.5, -0.3), 0.8))
        assert lp.lin_vel_command[0, 0] == pytest.approx(0.5)
        assert lp.lin_vel_command[0, 1] == pytest.approx(-0.3)
        assert lp.ang_vel_command[0, 0] == pytest.approx(0.8)


# ============================================================================
# DualMode X/x switching
# ============================================================================


def _try_import_dual_mode():
    try:
        from holosoma_inference.policies.dual_mode import DualModePolicy  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


_has_dual_mode = _try_import_dual_mode()
_skip_dual_mode = pytest.mark.skipif(not _has_dual_mode, reason="DualMode deps not installed")


def _make_dual():
    """Build a DualModePolicy with mock policies, skipping __init__."""
    from holosoma_inference.inputs.impl.interface import InterfaceInput
    from holosoma_inference.policies.dual_mode import DualModePolicy

    dual = object.__new__(DualModePolicy)
    dual.primary = _make_policy()
    dual.secondary = _make_policy()
    dual.active = dual.primary
    dual.active_label = "primary"

    # Shared InterfaceInput — one device, both policies use it
    shared_dev = InterfaceInput(dual.primary.interface)
    dual.primary._velocity_input = shared_dev
    dual.primary._command_provider = shared_dev
    dual.secondary._velocity_input = shared_dev
    dual.secondary._command_provider = shared_dev

    dual.primary._dispatch_command = MagicMock()
    dual.secondary._dispatch_command = MagicMock()

    dual._setup_command_intercept()
    return dual


@_skip_dual_mode
class TestDualModeSwitching:
    def test_switch_mode_injected_in_mappings(self):
        dual = _make_dual()
        assert dual.primary._command_provider._mapping["X"] == StateCommand.SWITCH_MODE
        assert dual.primary._command_provider._mapping["x"] == StateCommand.SWITCH_MODE
        assert dual.secondary._command_provider._mapping["X"] == StateCommand.SWITCH_MODE

    def test_x_joystick_triggers_switch(self):
        dual = _make_dual()
        assert dual.active is dual.primary
        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)
        assert dual.active is dual.secondary

    def test_double_switch_returns_to_primary(self):
        dual = _make_dual()
        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)
        assert dual.active is dual.secondary
        dual.secondary._dispatch_command(StateCommand.SWITCH_MODE)
        assert dual.active is dual.primary
        assert dual.active_label == "primary"

    def test_non_switch_command_delegates_to_active(self):
        dual = _make_dual()
        orig_dispatch = dual._orig_dispatch[id(dual.primary)]
        dual.primary._dispatch_command(StateCommand.START)
        orig_dispatch.assert_called_once_with(StateCommand.START)

    def test_delegates_to_secondary_after_switch(self):
        dual = _make_dual()
        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)
        orig_secondary = dual._orig_dispatch[id(dual.secondary)]
        dual.secondary._dispatch_command(StateCommand.START)
        orig_secondary.assert_called_once_with(StateCommand.START)

    def test_switch_stops_old_and_starts_new(self):
        dual = _make_dual()
        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)
        dual.primary._handle_stop_policy.assert_called_once()
        dual.secondary._resolve_control_gains.assert_called_once()
        dual.secondary._init_phase_components.assert_called_once()
        dual.secondary._handle_start_policy.assert_called_once()

    def test_joystick_state_carry_over(self):
        dual = _make_dual()
        # Set key_states on primary device
        dual.primary._velocity_input.key_states = {"X": True, "A": False}

        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)

        assert dual.secondary._velocity_input.key_states == {"X": True, "A": False}
        assert dual.secondary._velocity_input.last_key_states == {"X": True, "A": False}


@_skip_dual_mode
class TestDualModeKeyboardQueueWiring:
    def test_shared_queue_between_policies(self):
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput, _KeyboardListenerThread
        from holosoma_inference.policies.dual_mode import DualModePolicy

        dual = object.__new__(DualModePolicy)
        dual.primary = _make_policy()
        dual.secondary = _make_policy()
        dual.active = dual.primary
        dual.active_label = "primary"

        listener = _KeyboardListenerThread()
        q = listener.subscribe()

        dev = KeyboardInput(q)
        dual.primary._velocity_input = dev
        dual.primary._command_provider = dev
        dual.secondary._velocity_input = dev
        dual.secondary._command_provider = dev

        dual.primary._dispatch_command = MagicMock()
        dual.secondary._dispatch_command = MagicMock()

        dual._setup_command_intercept()

        assert dual.primary._command_provider is dual.secondary._command_provider

    def test_keyboard_commands_reach_active_via_poll(self):
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput, _KeyboardListenerThread
        from holosoma_inference.policies.dual_mode import DualModePolicy

        dual = object.__new__(DualModePolicy)
        dual.primary = _make_policy()
        dual.secondary = _make_policy()
        dual.active = dual.primary
        dual.active_label = "primary"

        listener = _KeyboardListenerThread()
        q = listener.subscribe()

        dev = KeyboardInput(q)
        dual.primary._velocity_input = dev
        dual.primary._command_provider = dev
        dual.secondary._velocity_input = dev
        dual.secondary._command_provider = dev

        dual.primary._dispatch_command = MagicMock()
        dual.secondary._dispatch_command = MagicMock()

        dual._setup_command_intercept()

        for sub_q in listener._subscribers:
            sub_q.append("]")

        # Active policy drains the shared queue
        dual.active._command_provider.poll_velocity()
        commands = dual.active._command_provider.poll_commands()
        assert commands == [StateCommand.START]

        # Queue is drained — secondary sees nothing stale
        dual.secondary._command_provider.poll_velocity()
        commands2 = dual.secondary._command_provider.poll_commands()
        assert commands2 == []

    def test_shared_queue_no_stale_accumulation(self):
        """Both policies share one input queue — no stale commands after switch."""
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput, _KeyboardListenerThread
        from holosoma_inference.policies.dual_mode import DualModePolicy

        dual = object.__new__(DualModePolicy)
        dual.primary = _make_policy()
        dual.secondary = _make_policy()
        dual.active = dual.primary
        dual.active_label = "primary"

        listener = _KeyboardListenerThread()
        q = listener.subscribe()

        dev = KeyboardInput(q, KEYBOARD_VELOCITY_LOCOMOTION)
        # Both policies share the same input provider
        dual.primary._velocity_input = dev
        dual.primary._command_provider = dev
        dual.secondary._velocity_input = dev
        dual.secondary._command_provider = dev

        dual.primary._dispatch_command = MagicMock()
        dual.secondary._dispatch_command = MagicMock()

        dual._setup_command_intercept()

        # Simulate keypresses while primary is active
        for sub_q in listener._subscribers:
            sub_q.extend(["]", "w", "o"])

        # Primary drains the shared queue
        dual.active._command_provider.poll_velocity()
        cmds = dual.active._command_provider.poll_commands()
        assert cmds == [StateCommand.START, StateCommand.STOP]

        # Switch to secondary
        dual._handle_mode_switch()

        # Queue is already empty — no stale commands
        cmds_after = dual.active._command_provider.poll_commands()
        assert cmds_after == []


# ============================================================================
# Separation guarantee: wrong-channel keys are not handled
# ============================================================================


class TestChannelSeparation:
    def test_velocity_only_keys_not_in_command_mapping(self):
        for key in ("w", "a", "d", "q", "e"):
            assert key not in KEYBOARD_COMMANDS

    def test_velocity_keys_in_velocity_mapping(self):
        assert "w" in KEYBOARD_VELOCITY_LOCOMOTION
        assert "a" in KEYBOARD_VELOCITY_LOCOMOTION
        assert "d" in KEYBOARD_VELOCITY_LOCOMOTION
        assert "q" in KEYBOARD_VELOCITY_LOCOMOTION
        assert "e" in KEYBOARD_VELOCITY_LOCOMOTION

    def test_interface_ignores_unmapped_buttons(self, interface):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        device = InterfaceInput(interface)
        device.key_states = {"unknown_stick": True}
        device.last_key_states = {"unknown_stick": False}
        assert device.poll_commands() == []


# ============================================================================
# Edge cases and error paths
# ============================================================================


# ============================================================================
# VelCmd dataclass
# ============================================================================


class TestVelocityCommand:
    def test_frozen(self):
        vc = VelCmd((0.0, 0.0), 0.0)
        with pytest.raises(AttributeError):
            vc.lin_vel = (1.0, 1.0)

    def test_equality(self):
        a = VelCmd((1.0, 2.0), 3.0)
        b = VelCmd((1.0, 2.0), 3.0)
        assert a == b

    def test_fields(self):
        vc = VelCmd((0.5, -0.3), 0.8)
        assert vc.lin_vel == (0.5, -0.3)
        assert vc.ang_vel == 0.8


# ============================================================================
# InputSource enum
# ============================================================================


class TestInputSource:
    def test_use_joystick_maps_to_interface(self):
        from holosoma_inference.config.config_types.task import TaskConfig

        tc = TaskConfig(model_path="test.onnx", use_joystick=True)
        assert tc.velocity_input == "interface"
        assert tc.state_input == "interface"
