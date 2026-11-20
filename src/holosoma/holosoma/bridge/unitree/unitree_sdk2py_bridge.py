import numpy as np
from loguru import logger
from unitree_interface import (
    LowState,
    MessageType,
    MotorCommand,
    RobotType,
    UnitreeInterface,
    WirelessController,
)

from holosoma.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge


class UnitreeSdk2Bridge(BasicSdk2Bridge):
    """Unitree SDK bridge implementation using unitree_interface C++ bindings."""

    SUPPORTED_ROBOT_TYPES = {"g1_29dof", "h1", "h1-2", "go2_12dof"}

    def _init_sdk_components(self):
        """Initialize Unitree SDK-specific components."""

        robot_type = self.robot.asset.robot_type

        # Validate robot type first
        if robot_type not in self.SUPPORTED_ROBOT_TYPES:
            raise ValueError(f"Invalid robot type '{robot_type}'. Unitree SDK supports: {self.SUPPORTED_ROBOT_TYPES}")

        # Map robot type to SDK enum
        robot_type_map = {
            "g1_29dof": RobotType.G1,
            "h1": RobotType.H1,
            "h1-2": RobotType.H1_2,
            "go2_12dof": RobotType.GO2,
        }

        # Map to message type (HG for humanoid robots with 35 motors, GO2 for others)
        message_type_map = {
            "g1_29dof": MessageType.HG,
            "h1": MessageType.GO2,
            "h1-2": MessageType.HG,
            "go2_12dof": MessageType.GO2,
        }

        sdk_robot_type = robot_type_map[robot_type]
        sdk_message_type = message_type_map[robot_type]

        # Get network interface from config
        interface_name = self.bridge_config.interface or "eth0"

        # Create interface (handles DDS initialization internally)
        self.interface = UnitreeInterface(interface_name, sdk_robot_type, sdk_message_type)

        # Initialize data structures
        self.low_state = LowState(self.num_motor)
        self.low_cmd = MotorCommand(self.num_motor)
        self.wireless_controller = WirelessController()

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
        self.low_state.motor.tau_est = actuator_forces.tolist()

        # Pack per-foot vertical GRF into imu.rpy[0:2] for sim-to-inference
        # transfer. rpy is unused by the G1 inference path, so this is a free
        # transported channel carrying a sim-only ground-truth signal (eval
        # logging reads it when is_sim=True). Mirrors the root-pos-in-accel
        # trick. (motor.ddq is not exposed by the SDK binding, so rpy is used.)
        left_grf_z, right_grf_z = self._get_foot_grf_z()
        self.low_state.imu.rpy = [left_grf_z, right_grf_z, 0.0]

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
        # Pack root position into imu.accel for sim-to-inference transfer.
        # The inference side reads this as base_pos when is_sim=True.
        root_pos = self.simulator.robot_root_states[0, :3].detach().cpu().numpy()
        self.low_state.imu.accel = root_pos.tolist()

        # Set timestamp (milliseconds)
        self.low_state.tick = int(self.sim_time * 1e3)

        # Publish (CRC calculated automatically in C++)
        self.interface.publish_low_state(self.low_state)

    # Foot body names for GRF extraction (G1 terminal foot links).
    _FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")

    def _get_foot_grf_z(self) -> tuple[float, float]:
        """Get per-foot vertical (world-z) ground reaction force from the sim.

        Computes contact forces directly from the current MuJoCo state via
        ``mj_contactForce`` and aggregates per foot body. This is done on
        demand because the simulator's ``contact_forces`` tensor is not
        refreshed on the MuJoCo run-sim path. Returns (0.0, 0.0) if the
        model/data or foot bodies are unavailable. Foot body ids are resolved
        once and cached.
        """
        import mujoco

        model = getattr(self.simulator, "root_model", None)
        data = getattr(self.simulator, "root_data", None)
        if model is None or data is None:
            return 0.0, 0.0

        if not hasattr(self, "_foot_body_ids"):
            try:
                self._foot_body_ids = [
                    self.simulator.find_rigid_body_indice(name)
                    for name in self._FOOT_BODY_NAMES
                ]
            except RuntimeError as e:
                logger.warning(f"Foot GRF disabled — body lookup failed: {e}")
                self._foot_body_ids = None

        if self._foot_body_ids is None:
            return 0.0, 0.0

        left_id, right_id = self._foot_body_ids
        left_grf_z = 0.0
        right_grf_z = 0.0
        forcetorque = np.zeros(6, dtype=np.float64)
        for c_idx in range(data.ncon):
            contact = data.contact[c_idx]
            mujoco.mj_contactForce(model, data, c_idx, forcetorque)
            # mj_contactForce returns the force in the CONTACT frame
            # [normal, tangent1, tangent2]. Rotate into the world frame using
            # contact.frame (3x3, rows = contact axes in world coords):
            #   f_world = R^T @ f_contact
            frame = np.asarray(contact.frame).reshape(3, 3)
            fz = float(frame[:, 2] @ forcetorque[:3])  # world-z component
            # Force is what geom1 exerts on geom2; body2 gets +, body1 gets -.
            body1 = model.geom_bodyid[contact.geom1]
            body2 = model.geom_bodyid[contact.geom2]
            if body1 == left_id:
                left_grf_z -= fz
            elif body1 == right_id:
                right_grf_z -= fz
            if body2 == left_id:
                left_grf_z += fz
            elif body2 == right_id:
                right_grf_z += fz
        return left_grf_z, right_grf_z

    def publish_wireless_controller(self):
        """Publish wireless controller data using unitree_interface."""
        # Call base class to populate wireless_controller from joystick
        super().publish_wireless_controller()

        # Publish using C++ interface
        if self.joystick is not None:
            self.interface.publish_wireless_controller(self.wireless_controller)

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
