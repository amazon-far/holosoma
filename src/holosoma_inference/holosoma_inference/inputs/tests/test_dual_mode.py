"""Tests for DualMode switching via StateCommand.SWITCH_MODE."""

from .conftest import _make_dual, skip_dual_mode


@skip_dual_mode
class TestDualModeSwitching:
    def test_switch_mode_command_triggers_switch(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        dual = _make_dual()
        assert dual.active is dual.primary
        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)
        assert dual.active is dual.secondary
        assert dual.active_label == "secondary"

    def test_double_switch_returns_to_primary(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        dual = _make_dual()
        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)
        dual.secondary._dispatch_command(StateCommand.SWITCH_MODE)
        assert dual.active is dual.primary
        assert dual.active_label == "primary"

    def test_non_switch_command_delegates_to_active(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        dual = _make_dual()
        orig_primary_dispatch = dual._orig_dispatch[id(dual.primary)]
        dual.primary._dispatch_command(StateCommand.START)
        orig_primary_dispatch.assert_called_once_with(StateCommand.START)

    def test_delegates_to_secondary_after_switch(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        dual = _make_dual()
        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)  # switch to secondary
        orig_secondary_dispatch = dual._orig_dispatch[id(dual.secondary)]
        dual.secondary._dispatch_command(StateCommand.START)
        orig_secondary_dispatch.assert_called_once_with(StateCommand.START)

    def test_switch_stops_old_and_starts_new(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        dual = _make_dual()
        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)
        dual.primary._handle_stop_policy.assert_called_once()
        dual.secondary._resolve_control_gains.assert_called_once()
        dual.secondary._init_phase_components.assert_called_once()
        dual.secondary._handle_start_policy.assert_called_once()

    def test_switch_mode_injected_into_command_provider_mapping(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        dual = _make_dual()
        assert dual.primary._command_provider._mapping.get("X") == StateCommand.SWITCH_MODE
        assert dual.primary._command_provider._mapping.get("x") == StateCommand.SWITCH_MODE
        assert dual.secondary._command_provider._mapping.get("X") == StateCommand.SWITCH_MODE
        assert dual.secondary._command_provider._mapping.get("x") == StateCommand.SWITCH_MODE

    def test_joystick_state_carry_over(self):
        from unittest.mock import MagicMock

        from holosoma_inference.inputs.api.commands import StateCommand
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        dual = _make_dual()
        # Replace mock velocity inputs with real InterfaceInput instances
        mock_interface = MagicMock()
        pri_vel = InterfaceInput(mock_interface)
        pri_vel.key_states = {"X": True, "A": False}
        sec_vel = InterfaceInput(mock_interface)

        dual.primary._velocity_input = pri_vel
        dual.secondary._velocity_input = sec_vel

        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)

        assert sec_vel.key_states == {"X": True, "A": False}
        assert sec_vel.last_key_states == {"X": True, "A": False}
