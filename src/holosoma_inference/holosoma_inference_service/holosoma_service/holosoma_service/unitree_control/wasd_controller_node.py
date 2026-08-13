#!/usr/bin/env python3
"""Minimal WASD keyboard teleop -> CmdExoskeleton (twist only).

Publishes base velocity from keyboard on ``/holosoma/tracker_command``; the arm
fields are left zero. Reads stdin in cbreak mode via select(), so it works over
SSH (no X / pynput).

  w/s : forward / back   (linear x)
  a/d : left / right     (linear y)
  q/e : yaw left / right (angular z)
  space : stop (zero velocity)
  Ctrl-C : quit

Each keypress steps the velocity; with no key held the command is republished at
a fixed rate so the consumer keeps a fresh value.
"""

from __future__ import annotations

import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from holosoma_msgs.msg import CmdExoskeleton
from rclpy.node import Node

from holosoma_service.unitree_control.unitree_split_controller import EXOSKELETON_TOPIC

LIN_STEP = 0.1  # m/s per keypress
ANG_STEP = 0.2  # rad/s per keypress
RATE_HZ = 20.0


class WasdControllerNode(Node):
    def __init__(self):
        super().__init__("wasd_controller")
        self._pub = self.create_publisher(CmdExoskeleton, EXOSKELETON_TOPIC, 10)
        self._vx = 0.0
        self._vy = 0.0
        self._vyaw = 0.0
        self.create_timer(1.0 / RATE_HZ, self._publish)

    def handle_key(self, k: str) -> None:
        if k == "w":
            self._vx += LIN_STEP
        elif k == "s":
            self._vx -= LIN_STEP
        elif k == "a":
            self._vy += LIN_STEP
        elif k == "d":
            self._vy -= LIN_STEP
        elif k == "q":
            self._vyaw += ANG_STEP
        elif k == "e":
            self._vyaw -= ANG_STEP
        elif k == " ":
            self._vx = self._vy = self._vyaw = 0.0
        else:
            return  # ignore other keys, don't reprint
        print(f"vel: vx={self._vx:+.2f} vy={self._vy:+.2f} vyaw={self._vyaw:+.2f}", flush=True)

    def _publish(self) -> None:
        msg = CmdExoskeleton()
        msg.header.stamp = self.get_clock().now().to_msg()
        twist = Twist()
        twist.linear.x = self._vx
        twist.linear.y = self._vy
        twist.angular.z = self._vyaw
        msg.base_velocity = twist
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WasdControllerNode()
    print(__doc__)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            if select.select([sys.stdin], [], [], 1.0 / RATE_HZ)[0]:
                node.handle_key(sys.stdin.read(1))
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
