"""Unit tests for TaskConfig input source configuration.

Covers the InputSource literal type, velocity_input/state_input fields,
use_keyboard/use_joystick shortcut aliases, and mutual exclusion validation.
"""

import pytest

from holosoma_inference.config.config_types.task import (
    DEFAULT_STATE_INPUT,
    DEFAULT_VELOCITY_INPUT,
    TaskConfig,
)


class TestDefaults:
    def test_default_values(self):
        assert DEFAULT_VELOCITY_INPUT == "keyboard"
        assert DEFAULT_STATE_INPUT == "keyboard"

    def test_default_inputs(self):
        tc = TaskConfig(model_path="test.onnx")
        assert tc.velocity_input == "keyboard"
        assert tc.state_input == "keyboard"

    def test_shortcuts_default_false(self):
        tc = TaskConfig(model_path="test.onnx")
        assert tc.use_joystick is False
        assert tc.use_keyboard is False


class TestExplicitInputSelection:
    """Verify explicit velocity_input/state_input combinations."""

    @pytest.mark.parametrize(
        ("vel", "other"),
        [
            ("keyboard", "keyboard"),
            ("joystick", "joystick"),
            ("ros2", "ros2"),
            ("ros2", "keyboard"),
            ("ros2", "joystick"),
            ("joystick", "keyboard"),
            ("keyboard", "ros2"),
            ("joystick", "ros2"),
            ("keyboard", "joystick"),
        ],
    )
    def test_all_combinations(self, vel, other):
        tc = TaskConfig(model_path="test.onnx", velocity_input=vel, state_input=other)
        assert tc.velocity_input == vel
        assert tc.state_input == other


class TestUseJoystickShortcut:
    def test_sets_both_channels(self):
        tc = TaskConfig(model_path="test.onnx", use_joystick=True)
        assert tc.velocity_input == "interface"
        assert tc.state_input == "interface"

    def test_conflicts_with_velocity_input(self):
        with pytest.raises(Exception, match="Cannot combine"):
            TaskConfig(model_path="test.onnx", use_joystick=True, velocity_input="ros2")

    def test_conflicts_with_state_input(self):
        with pytest.raises(Exception, match="Cannot combine"):
            TaskConfig(model_path="test.onnx", use_joystick=True, state_input="ros2")

    def test_conflicts_with_both_inputs(self):
        with pytest.raises(Exception, match="Cannot combine"):
            TaskConfig(model_path="test.onnx", use_joystick=True, velocity_input="ros2", state_input="ros2")


class TestUseKeyboardShortcut:
    def test_sets_both_channels(self):
        tc = TaskConfig(model_path="test.onnx", use_keyboard=True)
        assert tc.velocity_input == "keyboard"
        assert tc.state_input == "keyboard"

    def test_conflicts_with_velocity_input(self):
        with pytest.raises(Exception, match="Cannot combine"):
            TaskConfig(model_path="test.onnx", use_keyboard=True, velocity_input="ros2")

    def test_conflicts_with_state_input(self):
        with pytest.raises(Exception, match="Cannot combine"):
            TaskConfig(model_path="test.onnx", use_keyboard=True, state_input="joystick")


class TestShortcutMutualExclusion:
    def test_both_shortcuts_rejected(self):
        with pytest.raises(Exception, match="Cannot combine.*use-keyboard.*use-joystick"):
            TaskConfig(model_path="test.onnx", use_keyboard=True, use_joystick=True)


class TestDefaultConfigs:
    def test_all_defaults_load(self):
        from holosoma_inference.config.config_values.task import DEFAULTS

        for name, config in DEFAULTS.items():
            assert isinstance(config.velocity_input, str), f"{name}: velocity_input not str"
            assert isinstance(config.state_input, str), f"{name}: state_input not str"

    def test_default_configs_use_keyboard(self):
        from holosoma_inference.config.config_values.task import DEFAULTS

        for name, config in DEFAULTS.items():
            assert config.velocity_input == "keyboard", f"{name}: unexpected velocity_input"
            assert config.state_input == "keyboard", f"{name}: unexpected state_input"
