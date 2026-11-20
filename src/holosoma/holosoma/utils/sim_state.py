"""Simulator ground-truth state broadcast for sim-to-sim inference.

The sim publishes the robot's true pelvis pose on a dedicated ZMQ channel so
that inference can use it as a ground-truth localization source. Only the simulator ever
publishes on this port, so a policy running on the real robot never receives
it and cannot mistake it for real localization.

Message schema (JSON, one object per message):
    t                 float  sim time [s]
    source            str    always "holosoma_sim"
    pelvis_pos        [3]    pelvis (root) position, world frame [m]
    pelvis_quat_wxyz  [4]    pelvis orientation, world frame (same quaternion as imu.quat)

Keep in sync with the subscriber in
loco_arm_tracking_inference/sim_localization.py.
"""

from __future__ import annotations

import json

import zmq
from loguru import logger

SIM_STATE_SOURCE = "holosoma_sim"


class SimStatePub:
    """Publishes the simulator's ground-truth pelvis pose via ZMQ."""

    def __init__(self, port: int = 5557) -> None:
        self.port: int = port
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.enabled: bool = False

    def start(self) -> None:
        """Start the publisher."""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.port}")
            self.enabled = True
            logger.info(f"Sim state publisher started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start sim state publisher: {e}")
            self.enabled = False

    def publish(self, sim_time: float, pelvis_pos, pelvis_quat_wxyz) -> None:
        """Publish one pelvis pose sample (non-blocking; drops if no subscriber keeps up)."""
        if not self.enabled or not self.socket:
            return
        message = {
            "t": float(sim_time),
            "source": SIM_STATE_SOURCE,
            "pelvis_pos": [float(v) for v in pelvis_pos],
            "pelvis_quat_wxyz": [float(v) for v in pelvis_quat_wxyz],
        }
        try:
            self.socket.send_string(json.dumps(message), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.enabled = False
