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

        # System commands
        self.system_commands = {
            glfw.KEY_BACKSPACE: ("reset_simulation", lambda: self._reset_simulation()),
        }

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
            glfw.KEY_P: ("push_robots", lambda: logger.warning("Push Robots not implemented, ignoring...")),
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

        # Try system commands first
        if keycode in self.system_commands:
            name, action = self.system_commands[keycode]
            action()
            if self.on_command_executed:
                self.on_command_executed()
            return True

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

    def _reset_simulation(self):
        """Reset the simulation to initial state (works with both Classic and Warp backends)."""
        sim = self.simulator
        import mujoco as mj

        # Reset CPU-side data
        mj.mj_resetData(sim.root_model, sim.root_data)
        sim._set_robot_initial_state()
        mj.mj_forward(sim.root_model, sim.root_data)

        # Sync to GPU if using Warp backend
        if hasattr(sim.backend, 'initialize_state'):
            sim.backend.initialize_state()

        # Re-initialize gantry
        if sim.virtual_gantry is not None:
            sim.on_episode_start(env_id=0)

        # Note: the bridge keeps running so motor torques stay active (robot
        # doesn't go limp).  Press 'o' in the policy terminal to stop the
        # policy, then follow the normal startup flow (8 → 9 → ] → s).
        logger.info("Simulation reset to initial state (press 'o' in policy terminal to stop policy)")
