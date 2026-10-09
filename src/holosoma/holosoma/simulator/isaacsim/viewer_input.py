"""IsaacSim interactive-viewer input.

Acquires the ``carb.input`` keyboard and registers a callback that drives the command tensor,
camera-tracking toggle, and virtual-gantry keys — the only code touching ``carb.input``, and
reachable only when a viewport (headful) exists. Holds a back-reference to the simulator and
mutates its state (``commands``, ``simulator_config``, ``push_requested``, gantry) in place,
mirroring the ``ViewportCameraController(env)`` pattern. Named to match MuJoCo's
``viewer_input.ViewerInputController`` (which additionally owns a HUD overlay + multi-env
world switching that IsaacSim's viewport has no equivalent of)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from loguru import logger

from holosoma.simulator.shared.virtual_gantry import GantryCommand, GantryCommandData

if TYPE_CHECKING:
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim


class ViewerInputController:
    """Keyboard-driven teleop + gantry controls for the IsaacSim viewport."""

    def __init__(self, simulator: IsaacSim) -> None:
        """Acquire the keyboard and register the callback. No-op (logged) if carb is unavailable.

        Parameters
        ----------
        simulator : IsaacSim
            The IsaacSim simulator instance whose commands/config/gantry this controller drives.
        """
        self.simulator = simulator
        try:
            # Import necessary modules
            import carb.input
            import omni.appwindow

            # Get the input interface
            self.input_interface = carb.input.acquire_input_interface()
            self.appwindow = omni.appwindow.get_default_app_window()
            self.keyboard = self.appwindow.get_keyboard()

            # Define key mappings
            self.key_commands = {
                "W": "forward_command",
                "S": "backward_command",
                "A": "left_command",
                "D": "right_command",
                "Q": "heading_left_command",
                "E": "heading_right_command",
                "Z": "zero_command",
                "X": "walk_stand_toggle",
                "U": "height_up",
                "L": "height_down",
                "I": "waist_yaw_up",
                "K": "waist_yaw_down",
                "P": "push_robots",
                "Y": "toggle_camera_tracking",
                # Virtual gantry controls (using enum)
                "KEY_7": GantryCommand.LENGTH_ADJUST,  # decrease
                "KEY_8": GantryCommand.LENGTH_ADJUST,  # increase
                "KEY_9": GantryCommand.TOGGLE,
                "KEY_0": GantryCommand.FORCE_ADJUST,
                "MINUS": GantryCommand.FORCE_SIGN_TOGGLE,
            }

            self._carb_input = carb.input
            self.keyboard_sub = self.input_interface.subscribe_to_keyboard_events(
                self.keyboard,
                lambda event, *args: self._on_keyboard_event(event, *args),
            )
            logger.info("Keyboard controls initialized")

        except Exception as e:
            logger.warning(f"Could not initialize keyboard controls: {e}")

    def _on_keyboard_event(self, event: Any, *args: Any, **kwargs: Any) -> bool:
        """Handle a carb keyboard event: mutate commands / tracking / gantry. Returns True if handled."""
        sim = self.simulator
        # Only process key press events
        if event.type != self._carb_input.KeyboardEventType.KEY_PRESS:
            return False
        if event.input.name not in self.key_commands:
            return False

        command = self.key_commands[event.input.name]
        if command == "forward_command":
            sim.commands[:, 0] += 0.1
            logger.info(f"Current Command: {sim.commands[:,]}")
        elif command == "backward_command":
            sim.commands[:, 0] -= 0.1
            logger.info(f"Current Command: {sim.commands[:,]}")
        elif command == "left_command":
            sim.commands[:, 1] -= 0.1
            logger.info(f"Current Command: {sim.commands[:,]}")
        elif command == "right_command":
            sim.commands[:, 1] += 0.1
            logger.info(f"Current Command: {sim.commands[:,]}")
        elif command == "heading_left_command":
            sim.commands[:, 3] -= 0.1
            logger.info(f"Current Command: {sim.commands[:,]}")
        elif command == "heading_right_command":
            sim.commands[:, 3] += 0.1
            logger.info(f"Current Command: {sim.commands[:,]}")
        elif command == "zero_command":
            sim.commands[:, :4] = 0
            logger.info(f"Current Command: {sim.commands[:,]}")
        elif command == "walk_stand_toggle":
            sim.commands[:, 4] = 1 - sim.commands[:, 4]
            logger.info(f"Current Command: {sim.commands[:,]}")
        elif command == "height_up":
            sim.commands[:, 8] += 0.1
            logger.info(f"Current Command: {sim.commands[:,]}")
        elif command == "height_down":
            sim.commands[:, 8] -= 0.1
            logger.info(f"Current Command: {sim.commands[:,]}")
        elif command == "waist_yaw_up":
            sim.commands[:, 5] += 0.1
            logger.info(f"Current Command: {sim.commands[:,]}")
        elif command == "waist_yaw_down":
            sim.commands[:, 5] -= 0.1
            logger.info(f"Current Command: {sim.commands[:,]}")
        elif command == "push_robots":
            logger.info("Push Robots Requested")
            sim.push_requested = True
        elif command == "toggle_camera_tracking":
            self._toggle_camera_tracking()
        # Virtual gantry commands (using enum)
        elif command == GantryCommand.LENGTH_ADJUST:
            if sim.virtual_gantry:
                # Differentiate between KEY_7 (decrease) and KEY_8 (increase)
                amount = -0.1 if event.input.name == "KEY_7" else 0.1
                sim.virtual_gantry.handle_command(GantryCommandData(GantryCommand.LENGTH_ADJUST, {"amount": amount}))
        elif command == GantryCommand.TOGGLE:
            if sim.virtual_gantry:
                sim.virtual_gantry.handle_command(GantryCommandData(GantryCommand.TOGGLE))
        elif command == GantryCommand.FORCE_ADJUST:
            if sim.virtual_gantry:
                sim.virtual_gantry.handle_command(GantryCommandData(GantryCommand.FORCE_ADJUST))
        elif command == GantryCommand.FORCE_SIGN_TOGGLE:
            if sim.virtual_gantry:
                sim.virtual_gantry.handle_command(GantryCommandData(GantryCommand.FORCE_SIGN_TOGGLE))
        return True

    def _toggle_camera_tracking(self) -> None:
        """Flip viewer camera tracking on the sim config and (dis)engage the viewport controller."""
        sim = self.simulator
        was_enabled = sim.simulator_config.viewer.enable_tracking
        sim.simulator_config = dataclasses.replace(
            sim.simulator_config,
            viewer=dataclasses.replace(sim.simulator_config.viewer, enable_tracking=not was_enabled),
        )

        if sim.viewport_camera_controller is not None:
            if sim.simulator_config.viewer.enable_tracking and not was_enabled:
                # ENABLING tracking: capture current camera offset first
                sim.viewport_camera_controller.capture_current_camera_offset()
                sim.viewport_camera_controller.update_view_to_asset_root("robot")
            elif not sim.simulator_config.viewer.enable_tracking:
                # DISABLING tracking: freeze camera at current position
                # The callback only runs when origin_type == "asset_root", so setting it to
                # anything else will stop tracking while keeping the camera at its current position
                sim.viewport_camera_controller.cfg.origin_type = "static"

        status = "ON" if sim.simulator_config.viewer.enable_tracking else "OFF"
        logger.info(f"Camera tracking: {status}")
