from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.utils.crc import CRC

from holosoma_inference.sdk.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge


class UnitreeSdk2Bridge(BasicSdk2Bridge):
    """Unitree SDK2Py bridge implementation."""

    def _init_sdk_components(self):
        """Initialize Unitree SDK-specific components."""

        robot_type = self.robot.ROBOT_TYPE

        # Initialize based on robot type
        if "g1" in robot_type or "h1-2" in robot_type:
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        elif robot_type in {"h1", "go2"}:
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        else:
            # Raise an error if robot_type is not valid
            raise ValueError(f"Invalid robot type '{robot_type}'. Expected 'g1', 'h1', or 'go2'.")

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.low_cmd_handler, 1)

        # Initialize crc
        self.crc = CRC()

        # Initialize wireless controller for Unitree
        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
        self.wireless_controller_puber.Init()

    def low_cmd_handler(self, msg):
        """Handle Unitree low-level command messages."""
        if msg:
            self.low_cmd = msg

    def publish_low_state(self):
        """Publish Unitree low-level state."""
        if self.mj_data is None:
            return

        sensor = self.mj_data.sensordata
        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        qacc = self.mj_data.qacc
        actuator_force = self.mj_data.actuator_force
        num_motors = self.num_motor
        imu = self.low_state.imu_state

        motor_state = self.low_state.motor_state
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
            imu.quaternion[:] = sensor[self.dim_motor_sensor + 0 : self.dim_motor_sensor + 4]
            imu.gyroscope[:] = sensor[self.dim_motor_sensor + 4 : self.dim_motor_sensor + 7]
        else:
            imu.quaternion[:] = qpos[3:7]
            imu.gyroscope[:] = qvel[3:6]

        if self.use_sensor and self.have_frame_sensor_:
            imu.accelerometer[:] = sensor[self.dim_motor_sensor + 7 : self.dim_motor_sensor + 10]
        else:
            imu.accelerometer[:] = qacc[0:3]

        self.low_state.tick = int(self.mj_data.time * 1e3)
        self.low_state.crc = self.crc.Crc(self.low_state)
        self.low_state_puber.Write(self.low_state)
