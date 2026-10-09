from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from booster_robotics_sdk import (
    B1LowCmdSubscriber,
    B1LowStatePublisher,
    LowCmd,
    LowCmdType,
    LowState,
    MotorCmd,
    MotorState,
)
from loguru import logger

from holosoma.bridge.base import BasicSdk2Bridge
from holosoma.utils.rotations import get_euler_xyz
from holosoma.utils.safe_torch_import import torch


class BoosterSdk2Bridge(BasicSdk2Bridge):
    """Booster SDK2Py bridge implementation."""

    SUPPORTED_ROBOT_TYPES = {"t1_23dof", "t1_29dof"}

    def _init_sdk_components(self) -> None:
        """Initialize Booster SDK-specific components."""

        from booster_robotics_sdk import ChannelFactory

        # Use bridge config for domain_id and interface
        domain_id = self.bridge_config.domain_id

        # Note: holosoma_inference/booster is not using interface
        ChannelFactory.Instance().Init(domain_id)

        logger.info(f"Booster SDK factory initialized with domain_id={domain_id}")

        robot_type = self.sdk_robot_type
        if robot_type in self.SUPPORTED_ROBOT_TYPES:
            self.LowCmd = LowCmd
            self.LowState = LowState
            self.LowCmdType = LowCmdType
            self.MotorCmd = MotorCmd
            self.low_cmd = self.LowCmd()
            if self.motor_type == "serial":
                self.low_cmd.cmd_type = self.LowCmdType.SERIAL
            elif self.motor_type == "parallel":
                self.low_cmd.cmd_type = self.LowCmdType.PARALLEL
            self.motor_cmds = [MotorCmd() for _ in range(self.num_motor)]
            self.low_cmd.motor_cmd = self.motor_cmds
            self._pending_low_cmd = self.low_cmd
        else:
            # Raise an error if robot_type is not valid
            raise ValueError(f"Invalid robot type '{robot_type}'. Booster SDK supports: {self.SUPPORTED_ROBOT_TYPES}")

        # Booster sdk message
        self.low_state = LowState()
        self.low_state.motor_state_serial = [MotorState() for _ in range(self.num_motor)]
        self.low_state.motor_state_parallel = [MotorState() for _ in range(self.num_motor)]

        # Initialize Booster SDK components (factory should be initialized by SimulatorBridge)
        self.low_state_puber = B1LowStatePublisher()
        self.low_cmd_suber = B1LowCmdSubscriber(self._on_low_cmd)
        self.low_state_puber.InitChannel()
        self.low_cmd_suber.InitChannel()
        logger.info("Booster SDK components initialized successfully")
        # TODO: wireless controller for booster

    def _on_low_cmd(self, msg: Any) -> None:
        """Receive the latest complete command from the SDK callback."""
        self._pending_low_cmd = msg

    def low_cmd_handler(self, msg: Any = None) -> None:
        """Sample the latest subscriber command for the control loop."""
        if msg is not None:
            self._on_low_cmd(msg)
        self.low_cmd = self._pending_low_cmd

    def publish_low_state(self) -> None:
        """Publish Booster low-level state using simulator-agnostic interface."""
        if self.low_state_puber is None:
            return

        num_motors = self.num_motor
        imu = self.low_state.imu_state

        if self.motor_type == "serial":
            motor_state = self.low_state.motor_state_serial
        elif self.motor_type == "parallel":
            motor_state = self.low_state.motor_state_parallel
        else:
            raise ValueError(f"Invalid motor type '{self.motor_type}'. Expected 'serial' or 'parallel'.")

        positions, velocities, accelerations = self._get_dof_states()
        actuator_forces = self._get_actuator_forces()
        for i in range(num_motors):
            m = motor_state[i]
            m.q = positions[i]
            m.dq = velocities[i]
            m.ddq = accelerations[i]
            m.tau_est = actuator_forces[i]

        quaternion, gyro, acceleration = self._get_base_imu_data()
        roll, pitch, yaw = get_euler_xyz(quaternion.unsqueeze(0), w_last=False)  # w_last=False for [w,x,y,z]
        rpy = torch.stack([roll, pitch, yaw], dim=-1).squeeze(0).detach().cpu().numpy()
        imu.rpy = rpy
        imu.gyro = gyro.detach().cpu().numpy()
        imu.acc = acceleration.detach().cpu().numpy()

        self.low_state_puber.Write(self.low_state)

    def compute_torques(self) -> npt.NDArray[np.floating[Any]]:
        """Compute torques from fresh simulator state and the latest cached Booster command."""
        low_cmd = self.low_cmd
        if low_cmd is None:
            return self.torques

        try:
            motor_cmds = list(low_cmd.motor_cmd)
            tau_ff = np.array([motor_cmds[i].tau for i in range(self.num_motor)])
            kp = np.array([motor_cmds[i].kp for i in range(self.num_motor)])
            kd = np.array([motor_cmds[i].kd for i in range(self.num_motor)])
            q_target = np.array([motor_cmds[i].q for i in range(self.num_motor)])
            dq_target = np.array([motor_cmds[i].dq for i in range(self.num_motor)])

            # Use shared PD computation
            return self._compute_pd_torques(tau_ff, kp, kd, q_target, dq_target)
        except Exception as e:
            logger.error(f"Error computing torques: {e}")
            raise
