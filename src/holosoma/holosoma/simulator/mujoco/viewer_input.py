"""Interactive-viewer input handling for the MuJoCo passive viewer.

Owns the spectator-facing concerns that only matter when a viewer window exists: the
keyboard callback, the on-screen HUD help text, and the multi-env world switching. Robot
and gantry key bindings are delegated to the sibling :class:`CommandRegistry`; this
controller adds the overlay/tracking/world-switch keys on top.

Peer of ``command_registry.CommandRegistry``: it holds a back-reference to the simulator
and mutates simulator state (``current_world_id``, ``simulator_config``) in place, so the
render loop keeps reading those fields off the simulator.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from loguru import logger

from holosoma.simulator.mujoco.command_registry import CommandRegistry

if TYPE_CHECKING:
    from holosoma.simulator.mujoco.mujoco import MuJoCo


class ViewerInputController:
    """Keyboard callback + HUD overlay for the MuJoCo passive viewer."""

    def __init__(self, simulator: MuJoCo) -> None:
        """Initialize the controller with a simulator back-reference.

        Parameters
        ----------
        simulator : MuJoCo
            MuJoCo simulator instance whose viewer/commands/config this controller drives.
        """
        self.simulator = simulator
        self.show_text_overlay: bool = True
        # Built lazily on first robot/gantry key so a viewer with no such input never pays for it.
        self._command_registry: CommandRegistry | None = None

    def on_key(self, keycode: int) -> None:
        """Handle a viewer key press: overlay/tracking/world-switch keys, else delegate.

        Parameters
        ----------
        keycode : int
            GLFW keycode for the pressed key.
        """
        sim = self.simulator
        if sim.commands is None:
            return

        # Handle text overlay toggle
        # G key (71): Toggle text overlay visibility
        if keycode == 71:  # 'G' key
            self.show_text_overlay = not self.show_text_overlay
            status = "ON" if self.show_text_overlay else "OFF"
            logger.info(f"Text overlay: {status}")
            # Update overlay immediately when toggled
            self.update_text_overlay()
            return

        # Y key (89): Toggle camera tracking
        if keycode == 89:  # 'Y' key
            sim.simulator_config = dataclasses.replace(
                sim.simulator_config,
                viewer=dataclasses.replace(
                    sim.simulator_config.viewer, enable_tracking=not sim.simulator_config.viewer.enable_tracking
                ),
            )
            status = "ON" if sim.simulator_config.viewer.enable_tracking else "OFF"
            logger.info(f"Camera tracking: {status} (press 'Y' to toggle)")
            self.update_text_overlay()  # Update UI
            return

        # Handle world_id toggling for multi-environment visualization (WarpBackend only)
        # LEFT ARROW (263): Previous environment
        # RIGHT ARROW (262): Next environment
        # Numbers 0-9 (48-57): Jump to specific environment
        if sim.num_envs > 1:
            if keycode == 263:  # LEFT ARROW - Previous environment
                sim.current_world_id = (sim.current_world_id - 1) % sim.num_envs
                logger.info(f"Viewing environment: {sim.current_world_id + 1}/{sim.num_envs}")
                return
            if keycode == 262:  # RIGHT ARROW - Next environment
                sim.current_world_id = (sim.current_world_id + 1) % sim.num_envs
                logger.info(f"Viewing environment: {sim.current_world_id + 1}/{sim.num_envs}")
                return
            if 48 <= keycode <= 57:  # Number keys 0-9
                requested_id = keycode - 48  # Convert keycode to number (0-9)
                if requested_id < sim.num_envs:
                    sim.current_world_id = requested_id
                    logger.info(f"Viewing environment: {sim.current_world_id + 1}/{sim.num_envs}")
                else:
                    logger.warning(f"Environment {requested_id} does not exist (max: {sim.num_envs - 1})")
                return

        # Use unified command registry (robot + gantry keys)
        if self._command_registry is None:
            self._command_registry = CommandRegistry(sim)
            # Register callback for UI updates on command execution
            self._command_registry.on_command_executed = self.update_text_overlay

        # Single call handles both gantry and robot commands
        if self._command_registry.execute_command(keycode):
            return  # Command handled

        # Log unhandled keys
        logger.debug(f"Unhandled keycode: {keycode}")

    def update_text_overlay(self) -> None:
        """Update text overlay based on current state (event-driven).

        This method is called only when state changes occur (e.g., key presses),
        not on every render frame. This prevents the viewer's keyboard input
        system from being disrupted by frequent set_texts() calls.
        """
        sim = self.simulator
        if sim.viewer is None:
            return

        if not self.show_text_overlay:
            # Clear text overlays when disabled
            sim.viewer.set_texts([])
            return

        # Determine virtual gantry status
        if sim.virtual_gantry and sim.virtual_gantry.enabled:
            gantry_status = "active"
        else:
            gantry_status = "inactive"

        # Determine camera tracking status
        camera_status = "ON" if sim.simulator_config.viewer.enable_tracking else "OFF"

        # Build text overlay content
        text = (
            f"Virtual gantry is {gantry_status} \n"
            "Press '7' to raise it \n"
            "Press '8' to lower it \n"
            "Press '9' to toggle it \n"
            f"Camera tracking: {camera_status} \n"
            "Press 'y' to toggle camera tracking \n"
            "Press backspace to reset the environment \n"
            "Press 'g' to hide this menu"
        )

        # Use the passive viewer's set_texts method for screen-space HUD overlay.
        # Format: (font, gridpos, text1, text2); None font/gridpos use MuJoCo defaults.
        sim.viewer.set_texts((None, None, text, ""))
