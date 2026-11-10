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

    def test_falling_edge_not_dispatched(self):
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        iface = _make_interface()
        # First: press A
        iface.get_joystick_msg.return_value = _joystick_msg(keys=256)
        iface.get_joystick_key.return_value = "A"
        device = InterfaceInput(iface)
        device.poll_commands()

        # Then: release (no keys)
        iface.get_joystick_msg.return_value = _joystick_msg(keys=0)
        iface.get_joystick_key.return_value = ""
        assert device.poll_commands() == []


class TestJoystickCommandMapping:
    """Verify the JOYSTICK_COMMANDS mapping covers all expected buttons."""

    def test_core_buttons_mapped(self):
        from holosoma_inference.inputs.api.commands import StateCommand
        from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS

        assert JOYSTICK_COMMANDS["A"] == StateCommand.START
        assert JOYSTICK_COMMANDS["B"] == StateCommand.STOP
        assert JOYSTICK_COMMANDS["Y"] == StateCommand.INIT
        assert JOYSTICK_COMMANDS["L1+R1"] == StateCommand.KILL
