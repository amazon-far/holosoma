"""Simulator-agnostic bridge interface for robot control.

This module provides a unified interface for integrating robot SDK bridges
with different simulators (MuJoCo, IsaacGym, IsaacSim, etc.).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from loguru import logger

from holosoma.bridge import BasicSdk2Bridge, create_sdk2py_bridge
from holosoma.config_types.simulator import BridgeConfig
from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.utils.clock import ClockPub
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.base_simulator.hooks import HookRegistry


class SimulatorBridge:
    """Simulator-agnostic bridge interface for robot control.

    This class is intended to provide an interface between robot SDK bridges and simulators,
    allowing robot control to be added to any simulator without breaking existing
    functionality.

    Currently it is tested with MuJoCo-only via the base bridge interface.

    """

    def __init__(self, simulator: BaseSimulator, bridge_config: BridgeConfig):
        """Initialize the simulator bridge.

        Initializes the bridge system for robot SDK integration, including:
        - Robot SDK bridge for state publishing and command receiving
        - Clock publisher for motion synchronization (WBT policies)
        - Optional joystick/gamepad support

        Parameters
        ----------
        simulator : BaseSimulator
            The simulator instance to integrate with
        bridge_config : BridgeConfig
            Configuration for the bridge system
        """
        self.simulator: BaseSimulator = simulator
        self.bridge_config: BridgeConfig = bridge_config
        self.robot_bridge: BasicSdk2Bridge | None = None

        # Initialize clock publisher for WBT motion synchronization
        self.clock_pub: ClockPub = ClockPub()

        if self.bridge_config.interface is None:
            interface = self._auto_detect_interface()
            logger.info(f"Auto-detected bridge interface '{interface}'")
            self.bridge_config = replace(self.bridge_config, interface=interface)

        if bridge_config.enabled:
            logger.info("Robot bridge is enabled, initializing...")
            self._init_robot_bridge()
            # Start clock publisher for motion synchronization
            self.clock_pub.start()
            logger.info("Clock publisher initialized for motion synchronization")
        else:
            # We don't support runtime toggling on/off
            logger.info("Robot bridge disabled")

    def register_hooks(self, hooks: HookRegistry) -> None:
        """Register bridge lifecycle hooks with the simulator loop.

        step() is PRE_STEP: it applies SDK-computed torques, which must land before the substep
        integrates them.
        """
        hooks.add(Phase.PRE_STEP, self.step, name="bridge.step")
        hooks.add(Phase.CLOSE, self.close, name="bridge.close")

    def _init_robot_bridge(self):
        """Initialize the robot bridge using the copied factory function."""
        try:
            # Create robot bridge using the factory function from holosoma.bridge
            self.robot_bridge = create_sdk2py_bridge(self.simulator, self.simulator.robot_config, self.bridge_config)
            logger.info(
                f"Robot bridge initialized successfully with SDK type: {self.simulator.robot_config.bridge.sdk_type}"
            )

            # Setup joystick if enabled
            if self.bridge_config.use_joystick:
                self._setup_joystick()

        except Exception as e:
            logger.error(f"Failed to initialize robot bridge: {e}")
            raise

    def _setup_joystick(self):
        """Setup joystick/gamepad for robot control."""
        try:
            self.robot_bridge.setup_joystick(
                device_id=self.bridge_config.joystick_device, js_type=self.bridge_config.joystick_type
            )
            logger.info(
                f"Joystick initialized: device={self.bridge_config.joystick_device}, "
                f"type={self.bridge_config.joystick_type}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize joystick: {e}") from e

    def _auto_detect_interface(self):
        # Auto-detect interface based on platform (like holosoma_inference)
        if sys.platform == "linux":
            return "lo"
        if sys.platform == "darwin":
            return "lo0"
        raise NotImplementedError("Only support Linux and MacOS for Unitree SDK.")

    def step(self):
        """Execute bridge step during simulation.

        This method should be called during each simulation step when the bridge
        is enabled. It handles:
        - Publishing robot state to SDK
        - Publishing simulation clock for motion synchronization
        - Processing joystick input (if enabled)
        - Computing and applying torques from SDK commands
        """
        if not self.robot_bridge:
            return

        # Publish robot state to SDK
        self.robot_bridge.publish_low_state()

        # Publish base odometry over the SDK (SportModeState on rt/odommodestate) when configured.
        # Off by default.
        if self.bridge_config.publish_odom:
            self.robot_bridge.publish_odom()

        # Handle joystick input if available
        if hasattr(self.robot_bridge, "joystick") and self.robot_bridge.joystick:
            self.robot_bridge.publish_wireless_controller()
            logger.debug("Wireless controller input published")

        # Read incoming commands from DDS
        self.robot_bridge.low_cmd_handler()

        # Compute torques based on received commands
        self.robot_bridge.compute_torques()

        # Apply torques to simulator
        # (for now: convert to/from tensor for unified interface, which is unnecessary for mujoco...)
        torques_tensor = torch.from_numpy(self.robot_bridge.torques).to(
            device=self.simulator.device, dtype=torch.float32
        )
        self.simulator.apply_torques_at_dof(torques_tensor)

        # Publish simulation clock for e.g, WBT policies
        sim_time = self.simulator.time()
        self.clock_pub.publish(sim_time)

    def is_enabled(self) -> bool:
        """Check if the bridge is enabled and functional.

        Returns
        -------
        bool
            True if bridge is enabled and robot_bridge is initialized
        """
        return self.bridge_config.enabled and self.robot_bridge is not None

    def get_bridge_info(self) -> dict:
        """Get information about the current bridge configuration.

        Returns
        -------
        dict
            Dictionary containing bridge status and configuration info
        """
        return {
            "enabled": self.bridge_config.enabled,
            "sdk_type": self.simulator.robot_config.bridge.sdk_type if self.bridge_config.enabled else None,
            "robot_bridge_initialized": self.robot_bridge is not None,
            "has_joystick": self.robot_bridge is not None and self.robot_bridge.joystick is not None,
        }

    def close(self) -> None:
        """Tear down the bridge and its resources.

        Stops the clock publisher and, for the multiprocess Unitree bridge, joins its spawned DDS
        child (a plain in-process bridge has no ``close`` and is skipped). Safe to call more than once.
        """
        if self.robot_bridge is not None and hasattr(self.robot_bridge, "close"):
            self.robot_bridge.close()
        self.clock_pub.close()
