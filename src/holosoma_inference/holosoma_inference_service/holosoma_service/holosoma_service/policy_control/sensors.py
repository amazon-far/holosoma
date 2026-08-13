"""ROS2 sensor implementations for the service node.

The ``Sensor`` ABC lives in ``holosoma_inference.sensors.base``; policies
depend only on that interface. The concrete classes here are a service-layer
detail — they subscribe ROS2 topics on the shared ``ServiceIONode`` and
implement the ``Sensor`` protocol.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np
from loguru import logger
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from holosoma_inference.sensors.base import Sensor


def _decode_depth(msg: Image, topic: str) -> np.ndarray | None:
    """Decode a 32FC1 depth Image to an (H, W) float32 array, or None on bad encoding."""
    if msg.encoding != Ros2DepthConsumer.EXPECTED_ENCODING:
        logger.warning(
            f"Ros2DepthConsumer[{topic}]: expected encoding "
            f"{Ros2DepthConsumer.EXPECTED_ENCODING!r}, got {msg.encoding!r} — frame discarded"
        )
        return None
    return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)


class Ros2DepthConsumer(Sensor):
    """Consumes one or more ``sensor_msgs/Image`` depth topics (encoding
    ``32FC1``, raw metric depth in meters) and exposes a preprocessed,
    policy-ready stacked float32 array.

    Preprocessing per camera (matches the on-robot image_server's
    ``_resize_clip_expand_transpose`` that the policy was trained against):

    1. resize to ``(resized_height, resized_width)`` with area-averaging interpolation
    2. clip to ``[near_clip, far_clip]`` meters
    3. normalize to ``[-0.5, 0.5]``: ``(d - near) / (far - near) - 0.5``

    Output shape: ``(N, 1, resized_height, resized_width)`` where
    ``N == len(topics)``. Topic order is the camera order in the stack (front
    first, back second — image_server convention). Single-camera is ``N == 1``;
    the consumer always returns the same stacked layout.

    Multi-camera frames are time-aligned with ``message_filters``'
    ``ApproximateTimeSynchronizer`` so the stack is from (approximately) one
    instant — independent ROS2 topics are otherwise unsynchronized, unlike the
    image_server which grabbed both cameras in one call. Single-camera needs no
    sync and uses a plain subscription.

    ``get_latest()`` returns ``None`` until a (synchronized) frame set has
    arrived and while the latest set is older than ``timeout`` seconds, so the
    policy can zero the observation rather than feed a stale/partial stack.

    ``frame_delay_ms`` re-introduces the depth latency the policy trained with.
    The ROS2 transport is effectively instantaneous, so without this the policy
    would see fresher frames than it did in training. When set, the consumer
    buffers timestamped frame sets and ``get_latest()`` returns the freshest set
    at least ``frame_delay_ms`` old — a fixed wall-clock delay that is robust to
    the publish rate. ``0.0`` (default) returns the freshest set, as before.

    The publisher must produce ``32FC1`` images; a wrong encoding is logged and
    that frame set is dropped.
    """

    EXPECTED_ENCODING = "32FC1"

    # message_filters slop (s): max timestamp spread within a synced frame set.
    SYNC_SLOP_S = 0.05

    def __init__(
        self,
        node: Node,
        topics: list[str],
        resized_height: int = 27,
        resized_width: int = 48,
        near_clip: float = 0.1,
        far_clip: float = 2.0,
        timeout: float = 0.5,
        frame_delay_ms: float = 0.0,
    ):
        if not topics:
            raise ValueError("Ros2DepthConsumer requires at least one topic")
        if frame_delay_ms < 0:
            raise ValueError(f"frame_delay_ms must be >= 0, got {frame_delay_ms}")
        self._topics = list(topics)
        self._resized_height = resized_height
        self._resized_width = resized_width
        self._near_clip = near_clip
        self._far_clip = far_clip
        self._timeout = timeout
        self._frame_delay_s = frame_delay_ms / 1000.0
        # Ring buffer of (monotonic_stamp, (N, 1, H, W)) sets, newest last. A
        # depth of 1 reproduces the original freshest-frame behavior; otherwise
        # we keep enough history to look back ``frame_delay_ms``. The publish
        # rate is unknown here, so size generously and cap by age, not count.
        self._buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=128)
        self._lock = threading.Lock()

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        if len(self._topics) == 1:
            node.create_subscription(Image, self._topics[0], self._single_cb, qos)
        else:
            # Sync all camera topics so the stack is from one instant.
            subs = [Subscriber(node, Image, t, qos_profile=qos) for t in self._topics]
            self._sync = ApproximateTimeSynchronizer(subs, queue_size=2, slop=self.SYNC_SLOP_S)
            self._sync.registerCallback(self._synced_cb)

        logger.info(
            f"Ros2DepthConsumer subscribed to {len(self._topics)} camera(s): "
            f"{self._topics} (encoding={self.EXPECTED_ENCODING}, "
            f"resize={self._resized_height}x{self._resized_width}, "
            f"clip=[{self._near_clip}, {self._far_clip}], "
            f"frame_delay={frame_delay_ms}ms"
            f"{'' if len(self._topics) == 1 else f', sync slop={self.SYNC_SLOP_S}s'})"
        )

    def start(self) -> None:
        pass  # subscriptions live on the caller's node; nothing to start

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """resize -> clip -> normalize to [-0.5, 0.5]. Returns (1, H, W) float32."""
        resized = cv2.resize(frame, (self._resized_width, self._resized_height), interpolation=cv2.INTER_AREA)
        clipped = np.clip(resized, self._near_clip, self._far_clip)
        normalized = (clipped - self._near_clip) / (self._far_clip - self._near_clip) - 0.5
        return normalized[np.newaxis, :, :].astype(np.float32)

    def _store(self, frames: list[np.ndarray]) -> None:
        # Each raw (H_raw, W_raw) -> preprocessed (1, H, W) -> stack to (N, 1, H, W).
        stacked = np.stack([self._preprocess(f) for f in frames], axis=0)
        with self._lock:
            self._buffer.append((time.monotonic(), stacked))

    def _single_cb(self, msg: Image) -> None:
        frame = _decode_depth(msg, self._topics[0])
        if frame is not None:
            self._store([frame])

    def _synced_cb(self, *msgs: Image) -> None:
        frames = [_decode_depth(m, t) for m, t in zip(msgs, self._topics)]
        if any(f is None for f in frames):
            return  # a bad-encoding frame in the set; drop the whole set
        self._store(frames)

    def get_latest(self) -> np.ndarray | None:
        """Return ``(N, 1, H, W)`` float32 array, or ``None`` if absent/stale.

        With ``frame_delay_ms == 0`` this is the freshest set. Otherwise it is
        the freshest set at least ``frame_delay_ms`` old; ``None`` if not enough
        history has accumulated yet to honor the delay.
        """
        now = time.monotonic()
        with self._lock:
            if not self._buffer:
                return None
            # Stream liveness is measured against the newest set: if even that
            # is older than ``timeout``, the publisher has stopped — zero the
            # obs. (The delayed set is expected to lag, so don't time it out.)
            newest_stamp, _ = self._buffer[-1]
            if self._timeout > 0 and (now - newest_stamp) > self._timeout:
                return None
            # Walk newest -> oldest for the first set at least frame_delay old.
            target = now - self._frame_delay_s
            for stamp, frame in reversed(self._buffer):
                if stamp <= target:
                    return frame.copy()
            return None
