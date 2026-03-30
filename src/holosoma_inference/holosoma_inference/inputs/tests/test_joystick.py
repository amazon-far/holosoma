"""Tests for joystick/interface input providers (new impl API).

Note: Comprehensive interface input tests are in test_providers.py.
This module contains additional per-concern tests for interface-specific behaviour.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_interface(**overrides):
    iface = MagicMock()
    iface.get_joystick_msg.return_value = None
    iface.get_joystick_key.return_value = ""
    for k, v in overrides.items():
        setattr(iface, k, v)
    return iface


def _joystick_msg(lx=0.0, ly=0.0, rx=0.0, keys=0):
    return SimpleNamespace(lx=lx, ly=ly, rx=rx, keys=keys)


class TestInterfaceInputEdgeCases:
    """Additional edge-case tests for InterfaceInput."""

    def test_start_is_noop(self):
        from holosoma_inference.inputs.impl.interface import InterfaceInput
        from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS

        iface = _make_interface()
        device = InterfaceInput(iface, JOYSTICK_COMMANDS)
        device.start()  # should not raise

    def test_empty_key_states_produces_no_commands(self):
        from holosoma_inference.inputs.impl.interface import InterfaceInput
        from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS

        iface = _make_interface()
        device = InterfaceInput(iface, JOYSTICK_COMMANDS)
        assert device.key_states == {}
        assert device.poll_commands() == []

    def test_multiple_rising_edges(self):
        from holosoma_inference.inputs.api.commands import StateCommand
        from holosoma_inference.inputs.impl.interface import InterfaceInput
        from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS

        iface = _make_interface()
        device = InterfaceInput(iface, JOYSTICK_COMMANDS)
        device.key_states = {"A": True, "B": True, "Y": True}
        device.last_key_states = {}

        commands = device.poll_commands()
        assert StateCommand.START in commands
        assert StateCommand.STOP in commands
        assert StateCommand.INIT in commands

    def test_falling_edge_not_dispatched(self):
        from holosoma_inference.inputs.impl.interface import InterfaceInput
        from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS

        iface = _make_interface()
        device = InterfaceInput(iface, JOYSTICK_COMMANDS)
        device.key_states = {"A": False}
        device.last_key_states = {"A": True}

        assert device.poll_commands() == []

    def test_velocity_suppressed_when_button_pressed(self):
        from holosoma_inference.inputs.impl.interface import InterfaceInput
        from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS

        iface = _make_interface()
        iface.get_joystick_msg.return_value = _joystick_msg(ly=0.5, keys=256)
        iface.get_joystick_key.return_value = "A"

        device = InterfaceInput(iface, JOYSTICK_COMMANDS)
        vc = device.poll_velocity()

        assert vc is None

    def test_deadzone_applied_to_sticks(self):
        from holosoma_inference.inputs.impl.interface import InterfaceInput
        from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS

        iface = _make_interface()
        iface.get_joystick_msg.return_value = _joystick_msg(lx=0.05, ly=0.09, rx=0.03)
        iface.get_joystick_key.return_value = ""

        device = InterfaceInput(iface, JOYSTICK_COMMANDS)
        vc = device.poll_velocity()

        assert vc is not None
        assert vc.lin_vel == (0.0, 0.0)
        assert vc.ang_vel == 0.0


class TestJoystickCommandMapping:
    """Verify the JOYSTICK_COMMANDS mapping covers all expected buttons."""

    def test_core_buttons_mapped(self):
        from holosoma_inference.inputs.api.commands import StateCommand
        from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS

        assert JOYSTICK_COMMANDS["A"] == StateCommand.START
        assert JOYSTICK_COMMANDS["B"] == StateCommand.STOP
        assert JOYSTICK_COMMANDS["Y"] == StateCommand.INIT
        assert JOYSTICK_COMMANDS["L1+R1"] == StateCommand.KILL

    def test_locomotion_buttons_mapped(self):
        from holosoma_inference.inputs.api.commands import StateCommand
        from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS

        assert JOYSTICK_COMMANDS["back"] == StateCommand.STAND_TOGGLE
        assert JOYSTICK_COMMANDS["L2"] == StateCommand.ZERO_VELOCITY

    def test_wbt_buttons_mapped(self):
        from holosoma_inference.inputs.api.commands import StateCommand
        from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS

        assert JOYSTICK_COMMANDS["start"] == StateCommand.START_MOTION_CLIP

    def test_policy_select_mapped(self):
        from holosoma_inference.inputs.api.commands import StateCommand
        from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS

        assert JOYSTICK_COMMANDS["select"] == StateCommand.NEXT_POLICY
