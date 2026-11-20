"""MuJoCo command registry"""

from __future__ import annotations

from typing import Callable

import glfw
from loguru import logger

from holosoma.simulator.shared.virtual_gantry import GantryCommand, GantryCommandData


class CommandRegistry:
    """Unified command registry for MuJoCo key mappings."""

    def __init__(self, simulator):
        """Initialize command registry with simulator reference.

        Parameters
        ----------
        simulator : MuJoCo
            MuJoCo simulator instance
        """
        self.simulator = simulator
        self.on_command_executed: Callable | None = None  # Callback for UI updates

        # Robot commands
        self.robot_commands = {
            glfw.KEY_W: ("forward_command", lambda: self._adjust_command(0, 0.1)),
            glfw.KEY_S: ("backward_command", lambda: self._adjust_command(0, -0.1)),
            glfw.KEY_A: ("left_command", lambda: self._adjust_command(1, -0.1)),
            glfw.KEY_D: ("right_command", lambda: self._adjust_command(1, 0.1)),
            glfw.KEY_Q: ("heading_left_command", lambda: self._adjust_command(3, -0.1)),
            glfw.KEY_E: ("heading_right_command", lambda: self._adjust_command(3, 0.1)),
            glfw.KEY_Z: ("zero_command", lambda: self._zero_commands()),
            glfw.KEY_X: ("walk_stand_toggle", lambda: self._toggle_command(4)),
            glfw.KEY_U: ("height_up", lambda: self._adjust_command(8, 0.1)),
            glfw.KEY_L: ("height_down", lambda: self._adjust_command(8, -0.1)),
            glfw.KEY_I: ("waist_yaw_up", lambda: self._adjust_command(5, 0.1)),
            glfw.KEY_K: ("waist_yaw_down", lambda: self._adjust_command(5, -0.1)),
            glfw.KEY_P: ("toggle_ee_wrench", lambda: self._toggle_ee_wrench()),
        }

        # Gantry commands (using new enum-based system with parameters)
        self.gantry_commands = {
            glfw.KEY_7: GantryCommandData(GantryCommand.LENGTH_ADJUST, {"amount": -0.1}),
            glfw.KEY_8: GantryCommandData(GantryCommand.LENGTH_ADJUST, {"amount": 0.1}),
            glfw.KEY_9: GantryCommandData(GantryCommand.TOGGLE),
            glfw.KEY_0: GantryCommandData(GantryCommand.FORCE_ADJUST),
            glfw.KEY_MINUS: GantryCommandData(GantryCommand.FORCE_SIGN_TOGGLE),
        }

    def execute_command(self, keycode: int) -> bool:
        """Execute any command for keycode. Returns True if handled.

        Parameters
        ----------
        keycode : int
            GLFW keycode for the pressed key

        Returns
        -------
        bool
            True if command was handled, False otherwise
        """

        # Try gantry commands first (new enum-based system)
        if keycode in self.gantry_commands and self.simulator.virtual_gantry:
            command_data = self.gantry_commands[keycode]
            if self.simulator.virtual_gantry.handle_command(command_data):
                # Notify callback after gantry command
                if self.on_command_executed:
                    self.on_command_executed()
                return True

        # Try robot commands
        if keycode in self.robot_commands:
            name, action = self.robot_commands[keycode]
            action()
            logger.info(f"Current Command: {self.simulator.commands[0]}")
            # Notify callback after robot command
            if self.on_command_executed:
                self.on_command_executed()
            return True

        return False

    def _adjust_command(self, index: int, delta: float):
        """Adjust command value by delta."""
        self.simulator.commands[:, index] += delta

    def _toggle_command(self, index: int):
        """Toggle command value between 0 and 1."""
        self.simulator.commands[:, index] = 1 - self.simulator.commands[:, index]

    def _zero_commands(self):
        """Zero out movement commands."""
        self.simulator.commands[:, :4] = 0

    def _toggle_ee_wrench(self):
        """Toggle external EE wrench application on/off."""
        if hasattr(self.simulator, "apply_ee_wrench"):
            self.simulator.apply_ee_wrench = not self.simulator.apply_ee_wrench
            status = "ON" if self.simulator.apply_ee_wrench else "OFF"
            logger.info(f"EE wrench application: {status}")
            if not self.simulator.apply_ee_wrench and hasattr(self.simulator, "external_pusher"):
                self.simulator.external_pusher.reset()
        else:
            logger.warning("EE wrench not available on this simulator")
