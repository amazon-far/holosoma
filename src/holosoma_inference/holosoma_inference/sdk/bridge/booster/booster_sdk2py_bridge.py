from booster_robotics_sdk import (
    B1LowCmdSubscriber,
    B1LowStatePublisher,
    LowCmd,
    LowCmdType,
    LowState,
    MotorCmd,
    MotorState,
)

from holosoma_inference.sdk.bridge.base import BasicSdk2Bridge
from holosoma_inference.utils.math.quat import quat_to_rpy


class BoosterSdk2Bridge(BasicSdk2Bridge):
    """Booster SDK2Py bridge implementation."""

    def _init_sdk_components(self):
        """Initialize Booster SDK-specific components."""

        robot_type = self.robot.ROBOT_TYPE
        if robot_type in {"t1_23dof", "t1_29dof"}:
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
        else:
            # Raise an error if robot_type is not valid
            raise ValueError(f"Invalid robot type '{robot_type}'. Expected 't1_23dof' or 't1_29dof'.")

        # Booster sdk message
        self.low_state = LowState()
        self.low_state.motor_state_serial = [MotorState() for _ in range(self.num_motor)]
        self.low_state.motor_state_parallel = [MotorState() for _ in range(self.num_motor)]
        self.low_state_puber = B1LowStatePublisher()
        self.low_cmd_suber = B1LowCmdSubscriber(self.low_cmd_handler)
        self.low_state_puber.InitChannel()
        self.low_cmd_suber.InitChannel()
        # TODO: wireless controller for booster

    def low_cmd_handler(self, msg):
        """Handle Booster low-level command messages."""
        if msg:
            self.low_cmd = self.LowCmd()
            self.low_cmd.cmd_type = self.LowCmdType.SERIAL if self.motor_type == "serial" else self.LowCmdType.PARALLEL
            self.low_cmd.motor_cmd = msg.motor_cmd

    def publish_low_state(self):
        """Publish Booster low-level state."""
        if self.mj_data is None:
            return

        sensor = self.mj_data.sensordata
        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        qacc = self.mj_data.qacc
        actuator_force = self.mj_data.actuator_force
        num_motors = self.num_motor
        imu = self.low_state.imu_state

        if self.motor_type == "serial":
            motor_state = self.low_state.motor_state_serial
        elif self.motor_type == "parallel":
            motor_state = self.low_state.motor_state_parallel
        else:
            raise ValueError(f"Invalid motor type '{self.motor_type}'. Expected 'serial' or 'parallel'.")

        if self.use_sensor:
            for i in range(num_motors):
                m = motor_state[i]
                m.q = sensor[i]
                m.dq = sensor[i + num_motors]
                m.tau_est = sensor[i + 2 * num_motors]
        else:
            for i in range(num_motors):
                m = motor_state[i]
                m.q = qpos[7 + i]
                m.dq = qvel[6 + i]
                m.ddq = qacc[6 + i]
                m.tau_est = actuator_force[i]

        if self.use_sensor and self.have_frame_sensor_:
            quat = sensor[self.dim_motor_sensor + 0 : self.dim_motor_sensor + 4]
            gyro = sensor[self.dim_motor_sensor + 4 : self.dim_motor_sensor + 7]
            acc = sensor[self.dim_motor_sensor + 7 : self.dim_motor_sensor + 10]
        else:
            quat = qpos[3:7]
            quat = qpos[3:7]
            gyro = qvel[3:6]
            acc = qacc[0:3]

        rpy = quat_to_rpy(quat)
        imu.rpy = rpy
        imu.gyro = gyro
        imu.acc = acc
        self.low_state_puber.Write(self.low_state)
