import sys
from abc import ABC, abstractmethod

import numpy as np
import pygame
from loguru import logger

from holosoma.config_types.robot import RobotConfig
from holosoma.utils.safe_torch_import import torch


class BasicSdk2Bridge(ABC):
    """Abstract base class for SDK2Py bridge implementations."""

    def __init__(self, simulator, robot_config: RobotConfig, bridge_config, lcm=None):
        self.lcm = lcm
        self.robot = robot_config
        self.bridge_config = bridge_config
        self.sdk_type = robot_config.bridge.sdk_type
        self.motor_type = robot_config.bridge.motor_type

        # Store simulator reference for generic access
        self.simulator = simulator

        # Uses simulator actuator count (truly simulator-agnostic)
        self.num_motor = simulator.num_dof  # Generic actuator count
        self.torques = np.zeros(self.num_motor)  # Avoids config/model mismatches
        self.torque_limit = np.array(self.robot.dof_effort_limit_list)

        # Motor-joint mapping for robots where SDK motor count != simulator DOF count
        bridge_cfg = robot_config.bridge
        self.num_sdk_motors = bridge_cfg.num_sdk_motors or self.num_motor
        self.joint2motor = bridge_cfg.joint2motor

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }
        self.joystick = None

        # Initialize SDK-specific components
        self._init_sdk_components()

    @abstractmethod
    def _init_sdk_components(self):
        """Initialize SDK-specific components. Must be implemented by subclasses."""

    @abstractmethod
    def low_cmd_handler(self, msg):
        """Handle low-level command messages. Must be implemented by subclasses."""

    @abstractmethod
    def publish_low_state(self):
        """Publish low-level state. Must be implemented by subclasses."""

    @abstractmethod
    def compute_torques(self):
        """Compute motor torques. Must be implemented by subclasses."""

    def _compute_pd_torques(self, tau_ff, kp, kd, q_target, dq_target):
        """Helper method for PD control computation (shared logic).

        Parameters
        ----------
        tau_ff : array-like
            Feedforward torques (numpy array or torch tensor)
        kp : array-like
            Proportional gains (numpy array or torch tensor)
        kd : array-like
            Derivative gains (numpy array or torch tensor)
        q_target : array-like
            Target positions (numpy array or torch tensor)
        dq_target : array-like
            Target velocities (numpy array or torch tensor)

        Returns
        -------
        numpy.ndarray
            Computed torques with limits applied
        """
        # Get actual state from simulator
        q_actual = self.simulator.dof_pos[0]
        dq_actual = self.simulator.dof_vel[0]

        # Convert inputs to torch tensors if needed
        device = q_actual.device
        tau = torch.as_tensor(tau_ff, device=device, dtype=q_actual.dtype)
        kp_t = torch.as_tensor(kp, device=device, dtype=q_actual.dtype)
        kd_t = torch.as_tensor(kd, device=device, dtype=q_actual.dtype)
        q_des = torch.as_tensor(q_target, device=device, dtype=q_actual.dtype)
        dq_des = torch.as_tensor(dq_target, device=device, dtype=q_actual.dtype)

        # When SDK has more motors than simulator DOFs, extract the valid subset
        if self.joint2motor is not None:
            indices = list(self.joint2motor)
            tau = tau[indices]
            kp_t = kp_t[indices]
            kd_t = kd_t[indices]
            q_des = q_des[indices]
            dq_des = dq_des[indices]

        # PD control computation
        torques = tau + kp_t * (q_des - q_actual) + kd_t * (dq_des - dq_actual)
        # Convert to numpy and apply limits
        torques_np = torques.detach().cpu().numpy()
        self.torques = np.clip(torques_np, -self.torque_limit, self.torque_limit)
        return self.torques

    def publish_wireless_controller(self):
        """Publish wireless controller data."""
        if self.joystick is not None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(self.button_id["RB"])
            key_state[self.key_map["L1"]] = self.joystick.get_button(self.button_id["LB"])
            key_state[self.key_map["start"]] = self.joystick.get_button(self.button_id["START"])
            key_state[self.key_map["select"]] = self.joystick.get_button(self.button_id["SELECT"])
            key_state[self.key_map["R2"]] = self.joystick.get_axis(self.axis_id["RT"]) > 0
            key_state[self.key_map["L2"]] = self.joystick.get_axis(self.axis_id["LT"]) > 0
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            if hasattr(self, "wireless_controller"):
                self.wireless_controller.keys = key_value
                self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
                self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
                self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
                self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

                # Debug logging for joystick values
                logger.debug(
                    f"Joystick axes - LX: {self.wireless_controller.lx:.3f}, "
                    f"LY: {self.wireless_controller.ly:.3f}, "
                    f"RX: {self.wireless_controller.rx:.3f}, "
                    f"RY: {self.wireless_controller.ry:.3f}, "
                    f"keys: 0x{key_value:04x}"
                )

                # Only publish if the subclass has a publisher (C++ bindings handle this differently)
                if hasattr(self, "wireless_controller_puber"):
                    self.wireless_controller_puber.Write(self.wireless_controller)

    def setup_joystick(self, device_id=0, js_type="xbox"):
        """Setup joystick/gamepad."""

        # Platform check - pygame only works on Linux/macOS
        if sys.platform not in ["linux", "darwin"]:
            raise RuntimeError(f"Joystick not supported on {sys.platform}. Pygame joystick requires Linux or macOS.")

        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            raise RuntimeError("No joystick detected")

        if js_type == "xbox":
            if sys.platform.startswith("linux"):
                self.axis_id = {
                    "LX": 0,  # Left stick axis x
                    "LY": 1,  # Left stick axis y
                    "RX": 3,  # Right stick axis x
                    "RY": 4,  # Right stick axis y
                    "LT": 2,  # Left trigger
                    "RT": 5,  # Right trigger
                    "DX": 6,  # Directional pad x
                    "DY": 7,  # Directional pad y
                }
                self.button_id = {
                    "X": 2,
                    "Y": 3,
                    "B": 1,
                    "A": 0,
                    "LB": 4,
                    "RB": 5,
                    "SELECT": 6,
                    "START": 7,
                    "XBOX": 8,
                    "LSB": 9,
                    "RSB": 10,
                }
            elif sys.platform == "darwin":
                self.axis_id = {
                    "LX": 0,  # Left stick axis x
                    "LY": 1,  # Left stick axis y
                    "RX": 2,  # Right stick axis x
                    "RY": 3,  # Right stick axis y
                    "LT": 4,  # Left trigger
                    "RT": 5,  # Right trigger
                }
                self.button_id = {
                    "X": 2,
                    "Y": 3,
                    "B": 1,
                    "A": 0,
                    "LB": 9,
                    "RB": 10,
                    "SELECT": 4,
                    "START": 6,
                    "XBOX": 5,
                    "LSB": 7,
                    "RSB": 8,
                    "DYU": 11,
                    "DYD": 12,
                    "DXL": 13,
                    "DXR": 14,
                }
            else:
                print("Unsupported OS. ")

        elif js_type == "switch":
            # may differ for different OS, need to be checked
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 2,  # Right stick axis x
                "RY": 3,  # Right stick axis y
                "LT": 5,  # Left trigger
                "RT": 4,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 3,
                "Y": 4,
                "B": 1,
                "A": 0,
                "LB": 6,
                "RB": 7,
                "SELECT": 10,
                "START": 11,
            }
        else:
            print("Unsupported gamepad. ")

    def _get_dof_states(self):
        """Get DOF positions, velocities, accelerations (simulator-agnostic).

        Returns:
            tuple: (positions, velocities, accelerations) as numpy arrays
        """
        # Use generic simulator interface - works for all simulators
        positions = self.simulator.dof_pos[0].detach().cpu().numpy()
        velocities = self.simulator.dof_vel[0].detach().cpu().numpy()

        if not hasattr(self.simulator, "dof_acc"):
            raise RuntimeError("DOF acceleration not available (is the bridge enabled?)")

        accelerations = self.simulator.dof_acc[0].detach().cpu().numpy()

        return positions, velocities, accelerations

    @property
    def sim_time(self):
        """Get the simulation time."""
        return self.simulator.time()

    def _get_actuator_forces(self):
        """Get actuator forces (simulator-agnostic).

        Returns:
            numpy.ndarray: Actuator forces
        """
        # Bridge operates on env 0 by default
        env_id = getattr(self, "env_id", 0)
        forces = self.simulator.get_dof_forces(env_id)
        return forces[: self.num_motor].detach().cpu().numpy()

    def _get_base_imu_data(self):
        """Get base IMU data: quaternion, angular velocity, linear acceleration (simulator-agnostic).

        Returns:
            tuple: (quaternion, gyro, acceleration) as torch tensors
                - quaternion: [w, x, y, z] format (4 elements) - bridge SDK format
                - gyro: angular velocity [wx, wy, wz] (3 elements)
                - acceleration: linear acceleration [ax, ay, az] (3 elements)
        """
        quat_holosoma = self.simulator.robot_root_states[0, 3:7]  # [x, y, z, w]
        gyro = self.simulator.robot_root_states[0, 10:13]  # Angular velocity

        if not hasattr(self.simulator, "base_linear_acc"):
            logger.warning(
                "Base linear acceleration not available (bridge may be disabled in config). "
                "Returning zero acceleration."
            )
            acceleration = torch.zeros(3, device=quat_holosoma.device)
        else:
            acceleration = self.simulator.base_linear_acc[0]

        # Convert quaternion: holosoma [x, y, z, w] -> bridge SDK [w, x, y, z]
        quaternion = torch.stack([quat_holosoma[3], quat_holosoma[0], quat_holosoma[1], quat_holosoma[2]])

        return quaternion, gyro, acceleration

    def _get_sensor_data(self):
        """Get sensor data (Mujoco-only).

        Returns:
            numpy.ndarray: Raw sensor data array
        """
        if not hasattr(self.simulator, "root_data"):
            raise NotImplementedError(f"Sensor data access not implemented for {type(self.simulator).__name__}")

        return self.simulator.root_data.sensordata
