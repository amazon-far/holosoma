from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma_inference.config.config_types.task import InputSource
from holosoma_inference.inputs.impl.interface import InterfaceInput
from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS
from holosoma_inference.inputs.impl.keyboard import KeyboardInput
from holosoma_inference.inputs.impl.ros2 import Ros2StateCommandProvider, Ros2VelCmdProvider

if TYPE_CHECKING:
    from holosoma_inference.policies.base import BasePolicy


def create_input(policy: BasePolicy, source: InputSource, role: str):
    """Create an input provider for the given source and role ("velocity" or "command")."""
    if not policy.use_joystick and source in ("interface", "joystick"):
        source = "keyboard"

    if source in ("interface", "joystick"):
        return InterfaceInput(policy.interface, JOYSTICK_COMMANDS)

    if source == "keyboard":
        velocity_keys = getattr(policy, "_keyboard_velocity_mapping", None)
        return KeyboardInput.create(policy, velocity_keys=velocity_keys)

    if source == "ros2":
        if role == "velocity":
            return Ros2VelCmdProvider(policy.config.task.ros_cmd_vel_topic)
        return Ros2StateCommandProvider(policy.config.task.ros_state_input_topic)

    raise ValueError(f"Unknown input source: {source}")
