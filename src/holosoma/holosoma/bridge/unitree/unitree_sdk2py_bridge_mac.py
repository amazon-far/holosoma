from loguru import logger
import numpy as np

# Import from unitree_sdk2_python (official SDK)
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.utils.crc import CRC

from holosoma.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge


class UnitreeSdk2Bridge(BasicSdk2Bridge):
    """Unitree SDK bridge implementation using unitree_sdk2_python (official Python SDK)."""

    SUPPORTED_ROBOT_TYPES = {"g1_29dof", "h1", "h1-2", "go2_12dof"}

    # Robot type to message type mapping
    # HG = humanoid (G1, H1-2), GO = quadruped (Go2, H1)
    ROBOT_MESSAGE_TYPE = {
        "g1_29dof": "hg",
        "h1": "go",
        "h1-2": "hg",
        "go2_12dof": "go",
    }

    def _init_sdk_components(self):
        """Initialize Unitree SDK-specific components using unitree_sdk2_python."""
        robot_type = self.robot.asset.robot_type

        # Validate robot type first
        if robot_type not in self.SUPPORTED_ROBOT_TYPES:
            raise ValueError(f"Invalid robot type '{robot_type}'. Unitree SDK supports: {self.SUPPORTED_ROBOT_TYPES}")

        # Get network interface from config
        interface_name = self.bridge_config.interface or "eth0"

        # Initialize DDS channel factory
        # First argument is domain_id (0 is default)
        ChannelFactoryInitialize(0, interface_name)

        # Determine message type based on robot
        msg_type = self.ROBOT_MESSAGE_TYPE[robot_type]

        # Import appropriate message types based on robot
        if msg_type == "hg":
            # Humanoid robots (G1, H1-2) use unitree_hg messages
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self._LowCmd = LowCmd_
            self._LowState = LowState_
        else:
            # Quadruped robots (Go2, H1) use unitree_go messages
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self._LowCmd = LowCmd_
            self._LowState = LowState_

            # Initialize Go2-specific command header
            self.low_cmd.head[0] = 0xFE
            self.low_cmd.head[1] = 0xEF
            self.low_cmd.level_flag = 0xFF
            self.low_cmd.gpio = 0

        # Initialize data structures
        self.low_state = None
        self.crc = CRC()
        self._msg_type = msg_type

        # Create subscriber for low-level commands (from external controller)
        self.lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", self._LowCmd)
        self.lowcmd_subscriber.Init(self._low_cmd_callback, 10)

        # Create publisher for low-level state (to external controller)
        self.lowstate_publisher = ChannelPublisher("rt/lowstate", self._LowState)
        self.lowstate_publisher.Init()

        # Initialize motor commands
        for i in range(self.num_motor):
            self.low_cmd.motor_cmd[i].mode = 0x01  # Enable motor (PMSM mode)
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0

        logger.info(f"Initialized unitree_sdk2_python bridge for {robot_type} on interface {interface_name}")

    def _low_state_callback(self, msg):
        """Callback for receiving low-level state from robot."""
        self.low_state = msg

    def _low_cmd_callback(self, msg):
        """Callback for receiving low-level commands from external controller."""
        self.low_cmd = msg

    def low_cmd_handler(self, msg=None):
        """Handle Unitree low-level command messages.

        Note: In unitree_sdk2_python, commands are received via the subscriber
        callback. This method is kept for interface compatibility.
        """
        # The low_state is updated via _low_state_callback
        # This method can be used to poll/process the latest state if needed
        pass

    def publish_low_state(self):
        """Publish Unitree low-level state using unitree_sdk2_python.

        This publishes the simulator state to the DDS network so external
        controllers can read it.
        """
        # Get simulator data
        positions, velocities, accelerations = self._get_dof_states()
        actuator_forces = self._get_actuator_forces()
        quaternion, gyro, acceleration = self._get_base_imu_data()

        # Create a new state message for publishing
        if self._msg_type == "hg":
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_

            state_msg = unitree_hg_msg_dds__LowState_()
        else:
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_

            state_msg = unitree_go_msg_dds__LowState_()

        # Populate motor state
        for i in range(self.num_motor):
            state_msg.motor_state[i].q = float(positions[i])
            state_msg.motor_state[i].dq = float(velocities[i])
            state_msg.motor_state[i].ddq = float(accelerations[i])
            state_msg.motor_state[i].tau_est = float(actuator_forces[i])

        # Populate IMU state
        quat_array = quaternion.detach().cpu().numpy()
        state_msg.imu_state.quaternion[0] = float(quat_array[0])  # w
        state_msg.imu_state.quaternion[1] = float(quat_array[1])  # x
        state_msg.imu_state.quaternion[2] = float(quat_array[2])  # y
        state_msg.imu_state.quaternion[3] = float(quat_array[3])  # z

        gyro_np = gyro.detach().cpu().numpy()
        state_msg.imu_state.gyroscope[0] = float(gyro_np[0])
        state_msg.imu_state.gyroscope[1] = float(gyro_np[1])
        state_msg.imu_state.gyroscope[2] = float(gyro_np[2])

        accel_np = acceleration.detach().cpu().numpy()
        state_msg.imu_state.accelerometer[0] = float(accel_np[0])
        state_msg.imu_state.accelerometer[1] = float(accel_np[1])
        state_msg.imu_state.accelerometer[2] = float(accel_np[2])

        # Calculate CRC and publish
        state_msg.crc = self.crc.Crc(state_msg)

        # Publish state to external controller
        self.lowstate_publisher.Write(state_msg)

    def publish_wireless_controller(self):
        """Publish wireless controller data using unitree_sdk2_python."""
        # Call base class to populate wireless_controller from joystick
        super().publish_wireless_controller()

        # Note: unitree_sdk2_python uses a different wireless controller interface
        # This would need to be implemented based on the specific requirements
        if self.joystick is not None:
            logger.debug("Wireless controller state updated")

    def compute_torques(self):
        """Compute torques using Unitree's unified command structure.

        Reads commands from the DDS subscriber and computes PD control torques.
        """

        try:
            # Extract motor commands from low_cmd (set by external controller)
            # The external controller writes to rt/lowcmd, we read and apply
            tau_ff = np.zeros(self.num_motor)
            kp = np.zeros(self.num_motor)
            kd = np.zeros(self.num_motor)
            q_target = np.zeros(self.num_motor)
            dq_target = np.zeros(self.num_motor)

            for i in range(self.num_motor):
                tau_ff[i] = self.low_cmd.motor_cmd[i].tau
                kp[i] = self.low_cmd.motor_cmd[i].kp
                kd[i] = self.low_cmd.motor_cmd[i].kd
                q_target[i] = self.low_cmd.motor_cmd[i].q
                dq_target[i] = self.low_cmd.motor_cmd[i].dq

            return self._compute_pd_torques(
                tau_ff=tau_ff,
                kp=kp,
                kd=kd,
                q_target=q_target,
                dq_target=dq_target,
            )
        except Exception as e:
            logger.error(f"Error computing torques: {e}")
            raise

    def send_command(self, tau_ff, kp, kd, q_target, dq_target):
        """Send motor commands to the robot via DDS.

        This method allows the simulator to send commands to a real robot.

        Parameters
        ----------
        tau_ff : array-like
            Feedforward torques
        kp : array-like
            Proportional gains
        kd : array-like
            Derivative gains
        q_target : array-like
            Target positions
        dq_target : array-like
            Target velocities
        """
        # Populate motor commands
        for i in range(self.num_motor):
            self.low_cmd.motor_cmd[i].mode = 0x01  # Enable
            self.low_cmd.motor_cmd[i].tau = float(tau_ff[i])
            self.low_cmd.motor_cmd[i].kp = float(kp[i])
            self.low_cmd.motor_cmd[i].kd = float(kd[i])
            self.low_cmd.motor_cmd[i].q = float(q_target[i])
            self.low_cmd.motor_cmd[i].dq = float(dq_target[i])

        # Calculate CRC for message integrity
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)

        # Publish command
        self.lowcmd_publisher.Write(self.low_cmd)
