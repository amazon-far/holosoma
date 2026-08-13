"""G1 29-DoF arm controller driving Unitree's ``rt/arm_sdk``.

Self-contained: all constants live in the local :mod:`.const` module and the
transport is direct DDS only (runs on the Jetson). Provides per-joint effort
clamping and an arm-only initialization trajectory.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # dds
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd  # idl for g1, h1_2
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
from unitree_sdk2py.utils.crc import CRC

from .const import (
    EFFORT_LIMITS as _EFFORT_LIMITS,
)
from .const import (
    G1_INIT_TRAJECTORY,
    KD_HOLOSOMA_MIXED_GAINS,
    KP_HOLOSOMA_MIXED_GAINS,
    HardwareG1j29JointArmIndex,
    HardwareG1j29JointIndex,
)

K_TOPIC_LOW_COMMAND_DEBUG = "rt/lowcmd"
K_TOPIC_LOW_COMMAND_MOTION = "rt/arm_sdk"
K_TOPIC_LOW_STATE = "rt/lowstate"

G1_29_NUM_MOTORS = 35


class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None


class G1j29LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_NUM_MOTORS)]


class G1j29ArmController:
    def __init__(self, iface: str = "eth0", motion_mode=True, simulation_mode=False, logger=None, arm_kp_scale=1.0):
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode

        # Scale shoulder KP to match stiffer setups (arm_kp_scale=2.0 reproduces
        # dev/orkedar/tracker_service). Elbow/wrist/legs/waist and KD are unchanged.
        _SHOULDER_JOINTS = {
            HardwareG1j29JointIndex.k_left_shoulder_pitch,
            HardwareG1j29JointIndex.k_left_shoulder_roll,
            HardwareG1j29JointIndex.k_left_shoulder_yaw,
            HardwareG1j29JointIndex.k_right_shoulder_pitch,
            HardwareG1j29JointIndex.k_right_shoulder_roll,
            HardwareG1j29JointIndex.k_right_shoulder_yaw,
        }
        self._kp_gains = {
            j: kp * arm_kp_scale if j in _SHOULDER_JOINTS else kp for j, kp in KP_HOLOSOMA_MIXED_GAINS.items()
        }
        self._kd_gains = KD_HOLOSOMA_MIXED_GAINS

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None
        self._iface = iface
        self._logger = logger or logging.getLogger(__name__)
        self._logger.info("Initialize G1_29_ArmController...")

        self._init_dds()

        # Cache for latest state
        self._latest_lowstate = None

        # Wait for initial connection
        connect_attempt = 0
        while self._latest_lowstate is None:
            connect_attempt += 1
            self._read_state(timeout=0.01)
            time.sleep(0.1)
            if connect_attempt % 100 == 0:
                self._logger.info("[G1_29_ArmController] Waiting to subscribe dds...")
        self._logger.info("[G1_29_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self._latest_debug_state = None

        self._logger.info("[G1_29_ArmController] Connected via DDS")
        self.all_motor_q = self.get_current_motor_q()
        self._logger.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        self._logger.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        self._logger.info("Lock all joints except two arms...")

        _WAIST_JOINTS = {
            HardwareG1j29JointIndex.k_waist_yaw,
            HardwareG1j29JointIndex.k_waist_roll,
            HardwareG1j29JointIndex.k_waist_pitch,
        }
        # Only initialize arm+waist joints — leave leg joints (0-11) untouched so
        # the loco controller retains full ownership of them (matches g1pilot approach).
        for joint_id in HardwareG1j29JointArmIndex:
            self.msg.motor_cmd[joint_id].mode = 1
            self.msg.motor_cmd[joint_id].kp = self._kp_gains[HardwareG1j29JointIndex(joint_id)]
            self.msg.motor_cmd[joint_id].kd = self._kd_gains[HardwareG1j29JointIndex(joint_id)]
            self.msg.motor_cmd[joint_id].q = self.all_motor_q[joint_id]
        for joint_id in _WAIST_JOINTS:
            self.msg.motor_cmd[joint_id].mode = 1
            self.msg.motor_cmd[joint_id].kp = self._kp_gains[joint_id]
            self.msg.motor_cmd[joint_id].kd = self._kd_gains[joint_id]
            self.msg.motor_cmd[joint_id].q = 0.0
        self._logger.info("Lock OK!")

        self._logger.info("Initialize G1_29_ArmController OK!")

    def _init_dds(self):
        """Initialize direct DDS communication with the MCU."""
        ChannelFactoryInitialize(0, self._iface)
        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(K_TOPIC_LOW_COMMAND_MOTION, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(K_TOPIC_LOW_COMMAND_DEBUG, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(K_TOPIC_LOW_STATE, hg_LowState)
        self.lowstate_subscriber.Init()

    def _read_state(self, timeout: float | None = None):
        """Read the latest state from DDS and cache it."""
        msg = self.lowstate_subscriber.Read(timeout=timeout)
        self._latest_debug_state = msg
        if msg is not None:
            lowstate = G1j29LowState()
            for motor_id in range(G1_29_NUM_MOTORS):
                lowstate.motor_state[motor_id].q = msg.motor_state[motor_id].q
                lowstate.motor_state[motor_id].dq = msg.motor_state[motor_id].dq
            self._latest_lowstate = lowstate
        return self._latest_lowstate

    def debug_get_motor_temps(self) -> list[list[float]]:
        if self._latest_debug_state is None:
            return []
        return [motor.temperature for motor in self._latest_debug_state.motor_state]

    def debug_get_motor_tau(self) -> list[float]:
        if self._latest_debug_state is None:
            return []
        return [motor.tau_est for motor in self._latest_debug_state.motor_state]

    def debug_motor_voltage(self) -> list[float]:
        if self._latest_debug_state is None:
            return []
        return [motor.vol for motor in self._latest_debug_state.motor_state]

    def publish_command(self):
        """Publish the current command message to the robot."""
        if self.motion_mode:
            self.msg.motor_cmd[HardwareG1j29JointIndex.k_not_used_joint_0].q = 1.0

        arm_q_target = self.q_target.copy()
        arm_tauff_target = self.tauff_target.copy()

        current_arm_q = self.get_current_dual_arm_q()
        for idx, joint_id in enumerate(HardwareG1j29JointArmIndex):
            q_cmd = arm_q_target[idx]
            kp = self._kp_gains[HardwareG1j29JointIndex(joint_id)]
            effort_limit = _EFFORT_LIMITS.get(HardwareG1j29JointIndex(joint_id))
            if effort_limit is not None and kp > 0:
                max_q_error = effort_limit / kp
                q_cmd = float(np.clip(q_cmd, current_arm_q[idx] - max_q_error, current_arm_q[idx] + max_q_error))
            self.msg.motor_cmd[joint_id].q = q_cmd
            self.msg.motor_cmd[joint_id].dq = 0
            self.msg.motor_cmd[joint_id].tau = arm_tauff_target[idx]

        self.msg.crc = self.crc.Crc(self.msg)
        self.lowcmd_publisher.Write(self.msg)

        if self._speed_gradual_max is True:
            t_elapsed = time.time() - self._gradual_start_time
            self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

    def set_dual_arm_target(self, q_target, tauff_target):
        """Set control target values q & tau of the left and right arm motors."""
        self.q_target = q_target
        self.tauff_target = tauff_target

    def ctrl_dual_arm(self, q_target, tauff_target):
        """Set control target values q & tau and immediately publish command."""
        self.set_dual_arm_target(q_target, tauff_target)
        self.publish_command()

    def track_dual_arm(self, q_target, tauff_target=None):
        """Velocity-clip ``q_target`` (using the current arm_velocity_limit) then
        publish. One call so an out-of-process proxy needs a single round-trip
        and the clip reads state on this side."""
        if tauff_target is None:
            tauff_target = np.zeros(14)
        self.ctrl_dual_arm(q_target, tauff_target)

    def get_mode_machine(self):
        """Return current dds mode machine."""
        return self.lowstate_subscriber.Read().mode_machine

    def get_current_motor_q(self):
        """Return current state q of all body motors."""
        self._read_state()
        return np.array([self._latest_lowstate.motor_state[joint_id].q for joint_id in HardwareG1j29JointIndex])

    def get_current_dual_arm_q(self):
        """Return current state q of the left and right arm motors."""
        self._read_state()
        return np.array([self._latest_lowstate.motor_state[joint_id].q for joint_id in HardwareG1j29JointArmIndex])

    def get_current_dual_arm_dq(self):
        """Return current state dq of the left and right arm motors."""
        self._read_state()
        return np.array([self._latest_lowstate.motor_state[joint_id].dq for joint_id in HardwareG1j29JointArmIndex])

    def get_imu_state(self) -> dict | None:
        """Return latest IMU state from LowState: quaternion (wxyz), gyroscope (xyz rad/s), rpy (rad)."""
        raw = self._latest_debug_state
        if raw is None:
            return None
        imu = raw.imu_state
        return {
            "quaternion": np.array(
                [imu.quaternion[0], imu.quaternion[1], imu.quaternion[2], imu.quaternion[3]], dtype=np.float32
            ),  # wxyz
            "gyroscope": np.array([imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]], dtype=np.float32),  # rad/s
            "rpy": np.array([imu.rpy[0], imu.rpy[1], imu.rpy[2]], dtype=np.float32),  # rad
        }

    def speed_gradual_max(self, t=5.0):
        """Set arms velocity to gradually increase to max value over t seconds (default 5.0)."""
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        """Set arms velocity to the maximum value immediately, instead of gradually increasing."""
        self.arm_velocity_limit = 30.0

    def ctrl_dual_arm_initialization_pose(self, steps_per_segment=500, sleep_time=0.004, settle_time=2.0):
        """Move both arms through the initialization trajectory defined in G1_INIT_TRAJECTORY.

        Returns the final target q (14-element np.ndarray) so the caller can
        hold that position instead of reading back the (possibly undershoot)
        actual motor positions.
        """

        self._logger.info("[G1_29_ArmController] Starting initialization trajectory...")

        current_arm_q = self.get_current_dual_arm_q()
        self._logger.info(f"Current arm joint positions:\n{current_arm_q}")

        tauff_target = np.zeros(14)

        trajectory_waypoints = [current_arm_q] + [np.array(waypoint) for waypoint in G1_INIT_TRAJECTORY]

        for segment_idx in range(len(trajectory_waypoints) - 1):
            start_q = trajectory_waypoints[segment_idx]
            end_q = trajectory_waypoints[segment_idx + 1]

            if segment_idx == 0:
                self._logger.info("Moving from current position to waypoint 0...")
            else:
                self._logger.info(f"Moving to waypoint {segment_idx}/{len(G1_INIT_TRAJECTORY) - 1}...")

            for step in range(steps_per_segment + 1):
                progress = step / steps_per_segment
                interpolated_q = start_q + (end_q - start_q) * progress

                self.ctrl_dual_arm(interpolated_q, tauff_target)
                time.sleep(sleep_time)

            self._logger.info(f"Reached waypoint {segment_idx}")

        final_target = trajectory_waypoints[-1].copy()

        settle_steps = int(settle_time / sleep_time)
        self._logger.info(f"Settling at target for {settle_time}s ({settle_steps} steps)...")
        for _ in range(settle_steps):
            self.ctrl_dual_arm(final_target, tauff_target)
            time.sleep(sleep_time)

        actual_q = self.get_current_dual_arm_q()
        error = np.abs(final_target - actual_q)
        max_err_rad = np.max(error)
        self._logger.info(
            f"Init trajectory complete. Max joint error: {max_err_rad:.4f} rad ({np.degrees(max_err_rad):.1f} deg)"
        )
        self._logger.info(f"Target: {np.round(final_target, 4)}")
        self._logger.info(f"Actual: {np.round(actual_q, 4)}")

        return final_target

    def _is_weak_motor(self, motor_index):
        weak_motors = [
            HardwareG1j29JointIndex.k_left_ankle_pitch.value,
            HardwareG1j29JointIndex.k_right_ankle_pitch.value,
            # Left arm
            HardwareG1j29JointIndex.k_left_shoulder_pitch.value,
            HardwareG1j29JointIndex.k_left_shoulder_roll.value,
            HardwareG1j29JointIndex.k_left_shoulder_yaw.value,
            HardwareG1j29JointIndex.k_left_elbow.value,
            # Right arm
            HardwareG1j29JointIndex.k_right_shoulder_pitch.value,
            HardwareG1j29JointIndex.k_right_shoulder_roll.value,
            HardwareG1j29JointIndex.k_right_shoulder_yaw.value,
            HardwareG1j29JointIndex.k_right_elbow.value,
        ]
        return motor_index.value in weak_motors

    def _is_wrist_motor(self, motor_index):
        wrist_motors = [
            HardwareG1j29JointIndex.k_left_wrist_roll.value,
            HardwareG1j29JointIndex.k_left_wrist_pitch.value,
            HardwareG1j29JointIndex.k_left_wrist_yaw.value,
            HardwareG1j29JointIndex.k_right_wrist_roll.value,
            HardwareG1j29JointIndex.k_right_wrist_pitch.value,
            HardwareG1j29JointIndex.k_right_wrist_yaw.value,
        ]
        return motor_index.value in wrist_motors


if __name__ == "__main__":
    arm = G1j29ArmController(iface="eth0", motion_mode=True, simulation_mode=False)

    print("=== INITIALIZATION TRAJECTORY TEST ===")
    print("Robot will move arms through the initialization trajectory defined in G1_INIT_TRAJECTORY")
    print(f"Trajectory has {len(G1_INIT_TRAJECTORY)} waypoints")

    # Execute initialization trajectory motion with high-frequency control
    arm.ctrl_dual_arm_initialization_pose(steps_per_segment=500, sleep_time=0.004)

    print("\nExiting...")
