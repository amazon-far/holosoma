from loguru import logger

from holosoma.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge


class UnitreeSdk2Bridge(BasicSdk2Bridge):
    """Unitree SDK bridge implementation using unitree_interface C++ bindings."""

    SUPPORTED_ROBOT_TYPES = {"g1_29dof", "h1", "h1-2", "go2_12dof"}

    # robot_type -> SDK enum member NAME (resolved to the real enum via getattr where the binding is
    # imported). Kept as strings so the parent process can build them without importing the C++
    # binding — the multiprocess subclass ships them to its child, which owns the CycloneDDS binding.
    _ROBOT_TYPE_NAMES = {"g1_29dof": "G1", "h1": "H1", "h1-2": "H1_2", "go2_12dof": "GO2"}
    # Message type (HG for humanoid robots with 35 motors, GO2 for others).
    _MESSAGE_TYPE_NAMES = {"g1_29dof": "HG", "h1": "GO2", "h1-2": "HG", "go2_12dof": "GO2"}

    def _init_sdk_components(self):
        """Initialize Unitree SDK-specific components."""
        # Imported here (not at module top level) so importing this module never loads the C++
        # binding's bundled CycloneDDS — the multiprocess subclass keeps the parent binding-free.
        from unitree_interface import (
            LowState,
            MessageType,
            MotorCommand,
            OdomState,
            RobotType,
            UnitreeInterface,
            WirelessController,
        )

        robot_type = self.robot.asset.robot_type

        # Validate robot type first
        if robot_type not in self.SUPPORTED_ROBOT_TYPES:
            raise ValueError(f"Invalid robot type '{robot_type}'. Unitree SDK supports: {self.SUPPORTED_ROBOT_TYPES}")

        sdk_robot_type = getattr(RobotType, self._ROBOT_TYPE_NAMES[robot_type])
        sdk_message_type = getattr(MessageType, self._MESSAGE_TYPE_NAMES[robot_type])

        # Get network interface from config
        interface_name = self.bridge_config.interface or "eth0"

        # Create interface (handles DDS initialization internally)
        self.interface = UnitreeInterface(interface_name, sdk_robot_type, sdk_message_type)

        # Initialize data structures
        self.low_state = LowState(self.num_motor)
        self.low_cmd = MotorCommand(self.num_motor)
        self.wireless_controller = WirelessController()
        self.odom_state = OdomState()

    def low_cmd_handler(self, msg=None):
        """Handle Unitree low-level command messages."""
        # Poll for incoming commands from DDS
        self.low_cmd = self.interface.read_incoming_command()

    def publish_low_state(self):
        """Publish Unitree low-level state using simulator-agnostic interface."""

        # Get simulator data
        positions, velocities, accelerations = self._get_dof_states()
        actuator_forces = self._get_actuator_forces()
        quaternion, gyro, acceleration = self._get_base_imu_data()

        # Populate motor state
        self.low_state.motor.q = positions.tolist()
        self.low_state.motor.dq = velocities.tolist()
        self.low_state.motor.ddq = accelerations.tolist()
        self.low_state.motor.tau_est = actuator_forces.tolist()

        # Populate IMU state
        # Convert quaternion from torch tensor to list [w, x, y, z]
        quat_array = quaternion.detach().cpu().numpy()
        self.low_state.imu.quat = [
            float(quat_array[0]),  # w
            float(quat_array[1]),  # x
            float(quat_array[2]),  # y
            float(quat_array[3]),  # z
        ]
        self.low_state.imu.omega = gyro.detach().cpu().numpy().tolist()
        self.low_state.imu.accel = acceleration.detach().cpu().numpy().tolist()

        # Set timestamp (milliseconds)
        self.low_state.tick = int(self.sim_time * 1e3)

        # Publish (CRC calculated automatically in C++)
        self.interface.publish_low_state(self.low_state)

    def publish_wireless_controller(self):
        """Publish wireless controller data using unitree_interface."""
        # Call base class to populate wireless_controller from joystick
        super().publish_wireless_controller()

        # Publish using C++ interface
        if self.joystick is not None:
            self.interface.publish_wireless_controller(self.wireless_controller)

    def publish_odom(self):
        """Publish base odometry as SportModeState on rt/odommodestate.

        Makes the simulator publish the same base-state channel the real robot's onboard
        sport/loco mode does, so a downstream consumer (e.g. the telemetry node's
        read_odom_state -> /telemetry/odom) is identical in sim and on hardware.
        """
        position, quat_wxyz, lin_vel_body, yaw_speed = self._get_base_odometry()
        self.odom_state.position = position
        self.odom_state.velocity = lin_vel_body
        self.odom_state.yaw_speed = yaw_speed
        self.odom_state.quat = quat_wxyz
        self.interface.publish_odom_state(self.odom_state)

    def compute_torques(self):
        """Compute torques using Unitree's unified command structure."""
        if not (hasattr(self, "low_cmd") and self.low_cmd):
            return self.torques

        try:
            # Extract from Unitree's unified structure
            return self._compute_pd_torques(
                tau_ff=self.low_cmd.tau_ff,
                kp=self.low_cmd.kp,
                kd=self.low_cmd.kd,
                q_target=self.low_cmd.q_target,
                dq_target=self.low_cmd.dq_target,
            )
        except Exception as e:
            logger.error(f"Error computing torques: {e}")
            raise
