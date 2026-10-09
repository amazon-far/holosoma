"""Simulator-agnostic bridge interface for robot control.

This module provides a unified interface for integrating robot SDK bridges
with different simulators (MuJoCo, IsaacGym, IsaacSim, etc.).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Any

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
        self._started = False
        self._closed = False
        self._hooks_registered = False

        self.clock_pub: ClockPub = ClockPub()

        if self.bridge_config.interface is None:
            interface = self._auto_detect_interface()
            logger.info(f"Auto-detected bridge interface '{interface}'")
            self.bridge_config = replace(self.bridge_config, interface=interface)

        self.register_hooks(simulator.hooks)
        self.start()

    def register_hooks(self, hooks: HookRegistry) -> None:
        """Register bridge lifecycle hooks with the simulator loop."""
        if self._hooks_registered:
            return
        self._hooks_registered = True
        hooks.add(
            Phase.PRE_STEP,
            self.transport_step,
            name="bridge.transport",
            every=self.bridge_config.transport_decimation,
        )
        hooks.add(Phase.PRE_STEP, self.control_step, name="bridge.control")
        hooks.add(Phase.CLOSE, self.close, name="bridge.close")

    def start(self) -> None:
        """Acquire external resources after the simulator has registered ownership."""
        if self._started:
            return
        if self._closed:
            raise RuntimeError("Cannot start a closed simulator bridge")
        if not self.bridge_config.enabled:
            logger.info("Robot bridge disabled")
            self._started = True
            return

        logger.info("Robot bridge is enabled, initializing...")
        try:
            self._init_robot_bridge()
            self.clock_pub.start()
        except BaseException:
            try:
                self.close()
            except Exception:
                logger.exception("Bridge cleanup failed while preserving the startup error")
            raise
        self._started = True
        logger.info("Clock publisher initialized for motion synchronization")

    def _init_robot_bridge(self) -> None:
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

    def _setup_joystick(self) -> None:
        """Setup joystick/gamepad for robot control."""
        try:
            assert self.robot_bridge is not None
            self.robot_bridge.setup_joystick(
                device_id=self.bridge_config.joystick_device, js_type=self.bridge_config.joystick_type
            )
            logger.info(
                f"Joystick initialized: device={self.bridge_config.joystick_device}, "
                f"type={self.bridge_config.joystick_type}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize joystick: {e}") from e

    def _auto_detect_interface(self) -> str:
        # Auto-detect interface based on platform (like holosoma_inference). A dict lookup
        # instead of an if-chain: mypy runs with a fixed platform and warn_unreachable, so a
        # literal `sys.platform == "darwin"` branch is "unreachable" on the linux CI checker.
        loopback_by_platform = {"linux": "lo", "darwin": "lo0"}
        interface = loopback_by_platform.get(sys.platform)
        if interface is None:
            raise NotImplementedError("Only support Linux and MacOS for Unitree SDK.")
        return interface

    def transport_step(self) -> None:
        """Exchange SDK data at the configured transport cadence."""
        if not self.robot_bridge:
            return

        # Publish robot state to SDK
        self.robot_bridge.publish_low_state()

        # Publish base odometry over the SDK (SportModeState on rt/odommodestate) when configured.
        if self.bridge_config.publish_odom:
            self.robot_bridge.publish_odom()

        # Handle joystick input if available
        if hasattr(self.robot_bridge, "joystick") and self.robot_bridge.joystick:
            self.robot_bridge.publish_wireless_controller()
            logger.debug("Wireless controller input published")

        # Read incoming commands from DDS
        self.robot_bridge.low_cmd_handler()

        # Publish simulation clock for e.g, WBT policies
        sim_time = self.simulator.time()
        self.clock_pub.publish(sim_time)

    def control_step(self) -> None:
        """Recompute and apply torque before every physics step."""
        if not self.robot_bridge:
            return

        self.robot_bridge.compute_torques()

        # Apply torques to simulator
        # (for now: convert to/from tensor for unified interface, which is unnecessary for mujoco...)
        torques_tensor = torch.from_numpy(self.robot_bridge.torques).to(
            device=self.simulator.device,  # type: ignore[attr-defined]
            dtype=torch.float32,
        )
        # None when the bridge controls every DOF (fast full-width write); a list when it controls a
        # subset (scatter into only those DOFs, leaving a co-controller's slots untouched).
        self.simulator.apply_torques_at_dof(torques_tensor, dof_indices=self.robot_bridge._apply_indices)

    def is_enabled(self) -> bool:
        """Check if the bridge is enabled and functional.

        Returns
        -------
        bool
            True if bridge is enabled and robot_bridge is initialized
        """
        return self.bridge_config.enabled and self.robot_bridge is not None

    def get_bridge_info(self) -> dict[str, Any]:
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

        Stops the clock publisher and delegates resource cleanup to the SDK bridge's ``close``
        extension point. Safe to call more than once.
        """
        if self._closed:
            return
        self._closed = True

        failures: list[Exception] = []
        robot_bridge = self.robot_bridge
        self.robot_bridge = None
        if robot_bridge is not None and hasattr(robot_bridge, "close"):
            try:
                robot_bridge.close()
            except Exception as exc:
                failures.append(exc)
        try:
            self.clock_pub.close()
        except Exception as exc:
            failures.append(exc)

        if failures:
            details = "; ".join(repr(exc) for exc in failures)
            raise RuntimeError(f"Failed to close simulator bridge: {details}")
