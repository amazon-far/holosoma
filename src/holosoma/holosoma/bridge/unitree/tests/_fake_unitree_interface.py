"""An importable fake ``unitree_interface`` for the multiprocess-bridge test.

The MP bridge's child uses the ``spawn`` start method: a fresh interpreter that re-imports modules by
name, so an in-memory monkeypatch in the parent would not reach it. This module is a real, on-disk,
CycloneDDS-free stand-in that the child (and parent test) can inject onto ``sys.path`` and import as
``unitree_interface``. It records what the child publishes into a file so the parent can assert the
round-trip, and hands back a deterministic incoming command.

It mimics only the surface the bridge touches: ``RobotType``/``MessageType`` enums, ``UnitreeInterface``
(``publish_low_state``/``publish_odom_state``/``read_incoming_command``/``publish_wireless_controller``),
and the ``LowState``/``MotorCommand``/``WirelessController``/``OdomState`` data holders.
"""

from __future__ import annotations

import json
import os
from enum import Enum


class RobotType(Enum):
    G1 = "G1"
    H1 = "H1"
    H1_2 = "H1_2"
    GO2 = "GO2"


class MessageType(Enum):
    HG = "HG"
    GO2 = "GO2"


class _Motor:
    def __init__(self, n: int):
        self.q = [0.0] * n
        self.dq = [0.0] * n
        self.ddq = [0.0] * n
        self.tau_est = [0.0] * n


class _Imu:
    def __init__(self):
        self.quat = [0.0, 0.0, 0.0, 0.0]
        self.omega = [0.0, 0.0, 0.0]
        self.accel = [0.0, 0.0, 0.0]


class LowState:
    def __init__(self, n: int):
        self.motor = _Motor(n)
        self.imu = _Imu()
        self.tick = 0


class MotorCommand:
    def __init__(self, n: int):
        self.tau_ff = [0.0] * n
        self.kp = [0.0] * n
        self.kd = [0.0] * n
        self.q_target = [0.0] * n
        self.dq_target = [0.0] * n


class WirelessController:
    def __init__(self):
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.keys = 0


class OdomState:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.yaw_speed = 0.0
        self.quat = [0.0, 0.0, 0.0, 0.0]


# Where publishes are recorded and the canned incoming command is read (set by the test via env var
# so both the spawned child and the parent agree on the path).
_RECORD_PATH_ENV = "FAKE_UNITREE_RECORD"
# Number of motors the canned incoming command is sized for.
_NUM_MOTOR_ENV = "FAKE_UNITREE_NUM_MOTOR"


class UnitreeInterface:
    def __init__(self, interface_name, robot_type, message_type):
        self.interface_name = interface_name
        self.robot_type = robot_type
        self.message_type = message_type
        self._record(
            "init",
            {"interface": interface_name, "robot_type": robot_type.value, "message_type": message_type.value},
        )

    def _record(self, kind, payload):
        path = os.environ.get(_RECORD_PATH_ENV)
        if not path:
            return
        with open(path, "a") as f:
            f.write(json.dumps({"kind": kind, "payload": payload}) + "\n")

    def publish_low_state(self, low_state: LowState):
        self._record(
            "publish_low_state",
            {
                "q": list(low_state.motor.q),
                "dq": list(low_state.motor.dq),
                "ddq": list(low_state.motor.ddq),
                "tau_est": list(low_state.motor.tau_est),
                "quat": list(low_state.imu.quat),
                "omega": list(low_state.imu.omega),
                "accel": list(low_state.imu.accel),
                "tick": low_state.tick,
            },
        )

    def publish_odom_state(self, odom_state: OdomState):
        self._record(
            "publish_odom_state",
            {
                "position": list(odom_state.position),
                "velocity": list(odom_state.velocity),
                "yaw_speed": odom_state.yaw_speed,
                "quat": list(odom_state.quat),
            },
        )

    def read_incoming_command(self) -> MotorCommand:
        n = int(os.environ.get(_NUM_MOTOR_ENV, "1"))
        cmd = MotorCommand(n)
        # Deterministic, distinguishable-per-field values so the parent can assert exact plumbing.
        cmd.tau_ff = [1.0] * n
        cmd.kp = [2.0] * n
        cmd.kd = [3.0] * n
        cmd.q_target = [4.0] * n
        cmd.dq_target = [5.0] * n
        return cmd

    def publish_wireless_controller(self, wc: WirelessController):
        self._record(
            "publish_wireless_controller",
            {"lx": wc.lx, "ly": wc.ly, "rx": wc.rx, "ry": wc.ry, "keys": wc.keys},
        )
