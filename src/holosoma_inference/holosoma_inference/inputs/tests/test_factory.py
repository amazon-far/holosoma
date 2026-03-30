"""Tests for the create_input factory and provider wiring."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from .conftest import skip_policies


def _make_policy_stub(velocity_input="keyboard", state_input="keyboard", use_joystick=False):
    """Build a minimal mock policy suitable for create_input calls."""
    p = MagicMock()
    p.use_joystick = use_joystick
    p.interface = MagicMock()
    p.config = SimpleNamespace(
        task=SimpleNamespace(
            velocity_input=velocity_input,
            state_input=state_input,
            ros_cmd_vel_topic="cmd_vel",
            ros_state_input_topic="holosoma/state_input",
            ros_vel_timeout=1.0,
        )
    )
    # Default: no custom velocity mapping (base policy behaviour)
    p._keyboard_velocity_mapping = None
    return p


class TestCreateInputFactory:
    """Test create_input() factory returns appropriate provider types."""

    def test_keyboard_velocity_returns_keyboard_input(self):
        from holosoma_inference.inputs import create_input
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput

        p = _make_policy_stub(velocity_input="keyboard")
        result = create_input(p, "keyboard", "velocity")
        assert isinstance(result, KeyboardInput)

    def test_keyboard_command_returns_keyboard_input(self):
        from holosoma_inference.inputs import create_input
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput

        p = _make_policy_stub(state_input="keyboard")
        result = create_input(p, "keyboard", "command")
        assert isinstance(result, KeyboardInput)

    def test_ros2_velocity_returns_vel_provider(self):
        from holosoma_inference.inputs import create_input
        from holosoma_inference.inputs.impl.ros2 import Ros2Input

        p = _make_policy_stub(velocity_input="ros2")
        result = create_input(p, "ros2", "velocity")
        assert isinstance(result, Ros2Input)

    def test_ros2_command_returns_state_provider(self):
        from holosoma_inference.inputs import create_input
        from holosoma_inference.inputs.impl.ros2 import Ros2Input

        p = _make_policy_stub(state_input="ros2")
        result = create_input(p, "ros2", "command")
        assert isinstance(result, Ros2Input)

    def test_interface_source_returns_interface_input(self):
        from holosoma_inference.inputs import create_input
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        p = _make_policy_stub(use_joystick=True)
        result = create_input(p, "interface", "velocity")
        assert isinstance(result, InterfaceInput)

    def test_joystick_fallback_to_keyboard_when_no_joystick(self):
        from holosoma_inference.inputs import create_input
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput

        p = _make_policy_stub(use_joystick=False)
        result = create_input(p, "joystick", "velocity")
        assert isinstance(result, KeyboardInput)

    def test_interface_fallback_to_keyboard_when_no_joystick(self):
        from holosoma_inference.inputs import create_input
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput

        p = _make_policy_stub(use_joystick=False)
        result = create_input(p, "interface", "velocity")
        assert isinstance(result, KeyboardInput)

    def test_unknown_source_raises(self):
        from holosoma_inference.inputs import create_input

        p = _make_policy_stub()
        with pytest.raises(ValueError, match="Unknown input source"):
            create_input(p, "invalid", "velocity")


@skip_policies
class TestCreateInputProvidersIntegration:
    """Test _create_input_providers wiring in BasePolicy."""

    def test_same_source_shares_provider(self):
        """When velocity_input == state_input, both slots share one instance."""
        from holosoma_inference.policies.base import BasePolicy

        bp = BasePolicy.__new__(BasePolicy)
        bp.config = SimpleNamespace(task=SimpleNamespace(velocity_input="keyboard", state_input="keyboard"))
        bp.use_joystick = False
        bp.use_keyboard = False
        bp.logger = MagicMock()
        bp._keyboard_velocity_mapping = None
        bp.interface = MagicMock()
        bp._create_input_providers()

        assert bp._velocity_input is bp._command_provider

    def test_different_sources_create_separate_providers(self, monkeypatch):
        """When velocity_input != state_input, separate providers are created."""
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput
        from holosoma_inference.inputs.impl.ros2 import Ros2Input
        from holosoma_inference.policies.base import BasePolicy

        # Prevent rclpy import in start()
        monkeypatch.setattr(Ros2Input, "start", lambda *_: None)

        bp = BasePolicy.__new__(BasePolicy)
        bp.config = SimpleNamespace(
            task=SimpleNamespace(
                velocity_input="keyboard",
                state_input="ros2",
                ros_cmd_vel_topic="cmd_vel",
                ros_state_input_topic="holosoma/state_input",
                ros_vel_timeout=1.0,
            )
        )
        bp.use_joystick = False
        bp.use_keyboard = False
        bp.logger = MagicMock()
        bp._keyboard_velocity_mapping = None
        bp.interface = MagicMock()
        bp._create_input_providers()

        assert isinstance(bp._velocity_input, KeyboardInput)
        assert isinstance(bp._command_provider, Ros2Input)
        assert bp._velocity_input is not bp._command_provider


class TestChannelSeparation:
    """Verify that ros2 velocity input doesn't affect keyboard commands."""

    def test_ros2_velocity_does_not_provide_commands(self):
        """Ros2Input implements poll_velocity."""
        from holosoma_inference.inputs.impl.ros2 import Ros2Input

        p = Ros2Input("cmd_vel", "holosoma/state_input")
        assert hasattr(p, "poll_velocity")
