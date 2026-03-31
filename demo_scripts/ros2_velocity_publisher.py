"""Demo ROS2 velocity publisher for testing the input provider model.

Publishes a sequence of velocity commands to the cmd_vel topic, allowing
manual testing of the ROS2 velocity input channel without a joystick.

Usage:
    source scripts/source_mujoco_uv_setup.sh
    python demo_scripts/ros2_velocity_publisher.py

    # Custom topic or pattern:
    python demo_scripts/ros2_velocity_publisher.py --topic /cmd_vel --pattern circle
"""

import argparse
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from loguru import logger
from rclpy.node import Node
from std_msgs.msg import String


class VelocityPublisher(Node):
    """Publishes velocity commands in a configurable pattern."""

    PATTERNS = {
        "forward": "Walk forward at constant speed",
        "shuttle": "Shuttle: forward 0.5 m/s then backward 0.5 m/s, repeating",
        "circle": "Circle: forward + yaw rotation",
        "figure8": "Figure-8: alternating yaw direction",
        "square": "Square: forward segments with 90-degree turns",
        "stop": "Send zero velocity (stop the robot)",
    }

    def __init__(self, topic: str, other_topic: str, pattern: str, hz: float):
        super().__init__("holosoma_velocity_publisher")
        self.pub_vel = self.create_publisher(TwistStamped, topic, 10)
        self.pub_other = self.create_publisher(String, other_topic, 10)
        self.pattern = pattern
        self.hz = hz
        self._t = 0.0
        self._dt = 1.0 / hz
        self.timer = self.create_timer(self._dt, self._publish)
        logger.info(f"Publishing '{pattern}' on {topic} at {hz:.0f} Hz")
        logger.info(f"Other input topic: {other_topic}")
        logger.info("Press Ctrl+C to stop")

    def _make_twist(self, lin_x: float, lin_y: float, ang_z: float) -> TwistStamped:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = lin_x
        msg.twist.linear.y = lin_y
        msg.twist.angular.z = ang_z
        return msg

    def _publish(self):
        self._t += self._dt

        if self.pattern == "forward":
            msg = self._make_twist(0.5, 0.0, 0.0)

        elif self.pattern == "shuttle":
            # 3s forward at 0.5 m/s, 3s backward at 0.5 m/s, repeat
            period = 6.0
            phase = self._t % period
            if phase < 3.0:
                msg = self._make_twist(0.5, 0.0, 0.0)
            else:
                msg = self._make_twist(-0.5, 0.0, 0.0)

        elif self.pattern == "circle":
            msg = self._make_twist(0.4, 0.0, 0.5)

        elif self.pattern == "figure8":
            # Alternate yaw direction every 4 seconds
            period = 4.0
            yaw = 0.5 if (self._t % (2 * period)) < period else -0.5
            msg = self._make_twist(0.4, 0.0, yaw)

        elif self.pattern == "square":
            # 3s forward, 1s turn left (90 deg at 1.57 rad/s), repeat
            period = 4.0
            phase = self._t % period
            if phase < 3.0:
                msg = self._make_twist(0.4, 0.0, 0.0)
            else:
                msg = self._make_twist(0.0, 0.0, 1.57)

        elif self.pattern == "stop":
            msg = self._make_twist(0.0, 0.0, 0.0)

        else:
            msg = self._make_twist(0.0, 0.0, 0.0)

        self.pub_vel.publish(msg)

        # Log every second
        if int(self._t) != int(self._t - self._dt):
            logger.info(
                f"t={self._t:.1f}s  lin=({msg.twist.linear.x:.2f}, {msg.twist.linear.y:.2f})"
                f"  ang={msg.twist.angular.z:.2f}"
            )

    def send_command(self, cmd: str):
        """Send a discrete command to the other_input topic (e.g. 'walk', 'stand', 'start')."""
        msg = String()
        msg.data = cmd
        self.pub_other.publish(msg)
        logger.info(f"Sent command: {cmd}")


def main():
    parser = argparse.ArgumentParser(description="Publish ROS2 velocity commands for holosoma input provider testing")
    parser.add_argument("--topic", default="cmd_vel", help="TwistStamped velocity topic (default: cmd_vel)")
    parser.add_argument(
        "--other-topic",
        default="holosoma/other_input",
        help="String other_input topic (default: holosoma/other_input)",
    )
    parser.add_argument(
        "--pattern",
        default="forward",
        choices=VelocityPublisher.PATTERNS.keys(),
        help="Velocity pattern to publish (default: forward)",
    )
    parser.add_argument("--hz", type=float, default=20.0, help="Publish rate in Hz (default: 20)")
    parser.add_argument(
        "--start-cmd",
        default="start",
        help="Discrete command to send before publishing (default: start). Use 'none' to skip.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = VelocityPublisher(args.topic, args.other_topic, args.pattern, args.hz)

    # Send an initial command (e.g. 'start') so the policy begins executing
    if args.start_cmd.lower() != "none":
        time.sleep(0.5)  # brief wait for subscribers to connect
        node.send_command(args.start_cmd)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        node.send_command("stop")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
