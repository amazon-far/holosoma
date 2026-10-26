"""Depth input for the contextual policy: one path for hardware and simulation.

``far_zed_driver --depth`` publishes a 32FC1 ``sensor_msgs/Image`` in metres on
``<ns>/depth/depth_registered``, and a simulator publishes the same message on a topic of its
own, so both reach the policy through :class:`DepthConsumer`.

Nothing here delays depth. A real pipeline is already about as stale as training modelled it, so
added delay would double the staleness; a simulator's depth is instantaneous, so it emulates that
latency by delaying what it publishes.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

HEIGHT, WIDTH = 58, 87
# The range the policy was trained against; outside it the encoder has seen nothing.
NEAR_M, FAR_M = 0.3, 2.0
DEPTH_TOPIC = "depth/depth_registered"


def preprocess(depth_m, near_m: float = NEAR_M, far_m: float = FAR_M) -> np.ndarray:
    """Resize a metric depth frame to 58x87 and normalise it to [-0.5, +0.5]."""
    frame = np.asarray(depth_m, dtype=np.float32)
    if frame.ndim != 2:
        raise ValueError(f"expected a 2-D depth frame, got shape {frame.shape}")
    if not far_m > near_m:
        raise ValueError(f"far_m must exceed near_m, got {near_m} and {far_m}")
    # The SDK marks "no return" as non-finite; those pixels read as far, not as holes.
    frame = np.nan_to_num(frame, nan=far_m, posinf=far_m, neginf=near_m)
    # Clip before resizing as well as after, so out-of-range pixels cannot bleed into their
    # neighbours through the interpolation kernel.
    frame = np.clip(frame, near_m, far_m)
    if frame.shape != (HEIGHT, WIDTH):
        zoom = (HEIGHT / frame.shape[0], WIDTH / frame.shape[1])
        # Cubic spline rather than training's torchvision BICUBIC: sub-millimetre difference on
        # depth this smooth, and scipy is a declared dependency where OpenCV is not.
        frame = ndimage.zoom(frame, zoom, order=3, mode="nearest")
    return (np.clip(frame, near_m, far_m) - near_m) / (far_m - near_m) - 0.5


def frame_from_image(msg) -> np.ndarray:
    """Read a 32FC1 ``sensor_msgs/Image`` into an (h, w) float32 array of metres."""
    if msg.encoding != "32FC1":
        raise ValueError(f"expected 32FC1 depth, got {msg.encoding!r}")
    if msg.step != msg.width * 4:
        raise ValueError(f"expected unpadded rows, got step {msg.step} for width {msg.width}")
    return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)


def stamp_seconds(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class DepthConsumer:
    """Feeds a contextual policy from a 32FC1 depth topic, on its own executor thread.

    ``rclpy`` is imported on construction rather than at module scope: it is a ROS-deployment
    extra, and ``preprocess`` is useful without it.
    """

    def __init__(self, policy, topic: str = DEPTH_TOPIC, *, near_m: float = NEAR_M, far_m: float = FAR_M):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        self._policy = policy
        self._near_m, self._far_m = near_m, far_m
        self.frames = 0
        self.rejected = 0
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("contextual_depth_consumer")
        # Matches far_zed_driver's publisher, so the subscription is compatible as published.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._node.create_subscription(Image, topic, self._on_depth, qos)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = None
        self._topic = topic

    def _on_depth(self, msg) -> None:
        frame = preprocess(frame_from_image(msg), self._near_m, self._far_m)
        if self._policy.set_depth(frame, stamp_seconds(msg)):
            self.frames += 1
        else:
            self.rejected += 1

    def start(self) -> None:
        """Spin the subscription on a daemon thread, leaving the policy loop blocking."""
        import threading

        if self._thread is not None:
            return
        self._node.get_logger().info(f"contextual depth from {self._topic}")
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._executor.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._node.destroy_node()
