"""Depth input for the contextual policy: one path for hardware and simulation.

``far_zed_driver --depth`` publishes a 32FC1 ``sensor_msgs/Image`` in metres on
``<ns>/depth/depth_registered``, and a simulator publishes the same message on a topic of its
own, so both reach the policy through :class:`DepthConsumer`. Producers that preprocess frames
themselves (holosoma's ``DepthShmPlugin``, an on-robot depth server) write them to a shared-memory
block instead, read by :class:`DepthShmConsumer`.

Nothing here delays depth. A real pipeline is already about as stale as training modelled it, so
added delay would double the staleness; a simulator's depth is instantaneous, so it emulates that
latency by delaying what it publishes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy import ndimage

HEIGHT, WIDTH = 58, 87
# The range the policy was trained against; outside it the encoder has seen nothing.
NEAR_M, FAR_M = 0.3, 2.0
DEPTH_TOPIC = "depth/depth_registered"
# The plane a pixel with no reading takes. "far" is what the primitive models were trained on; a
# continuous-velocity export declares its own in its ONNX metadata, and that is what is applied.
INVALID_FILL = "far"
INVALID_FILLS = ("far", "near")
# torchvision's antialiased bicubic, which the continuous-velocity exports were trained with.
BICUBIC_ANTIALIAS = "bicubic_antialias"
RESIZES = (BICUBIC_ANTIALIAS,)
FRAME_SHAPE_MISMATCH = "depth frame does not have the size the model was trained to resize"
UNFRAMED_CONTRACT = "the model's depth contract does not say how a depth topic frame is resized"


@dataclass(frozen=True)
class DepthSettings:
    """How a metric depth frame becomes the policy's input: clip range, fill, framing, resize.

    ``raw_shape_hw`` is the only frame size accepted, and ``resize`` how it is resized; both None
    accepts any size and resizes with a cubic spline, which is what the primitive models get.
    """

    near_m: float = NEAR_M
    far_m: float = FAR_M
    invalid_fill: str = INVALID_FILL
    raw_shape_hw: tuple[int, int] | None = None
    resize: str | None = None


def settings_for(contract, default: DepthSettings | None = None) -> DepthSettings:
    """The preprocessing a depth topic must apply for a model's declared depth contract.

    ``contract`` is the active model's ``depth_contract`` (None for a model that declares none,
    such as the primitive exports, which get ``default``). A declared contract fixes everything,
    so nothing is left for an operator to choose; one that does not name the frame size and
    resize it was trained with is refused, since a topic of any other framing would reach the
    model with a different field of view or resampling than training saw.
    """
    if contract is None:
        return default or DepthSettings()
    raw, resize = contract.get("raw_shape_hw"), contract.get("resize")
    if raw is None or resize is None:
        raise ValueError(f"{UNFRAMED_CONTRACT}: it must declare depth_raw_shape_hw and depth_resize")
    if resize not in RESIZES:
        raise ValueError(f"{UNFRAMED_CONTRACT}: depth_resize {resize!r} is not one of {RESIZES}")
    return DepthSettings(
        near_m=float(contract["near_m"]),
        far_m=float(contract["far_m"]),
        invalid_fill=contract["invalid_fill"],
        raw_shape_hw=(int(raw[0]), int(raw[1])),
        resize=resize,
    )


def _cubic(x: np.ndarray, a: float = -0.5) -> np.ndarray:
    x = np.abs(x)
    near = ((a + 2) * x - (a + 3)) * x * x + 1
    far = ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return np.where(x < 1, near, np.where(x < 2, far, 0.0))


def _antialias_weights(n_in: int, n_out: int) -> np.ndarray:
    # One row per output pixel. When shrinking, the kernel widens by the scale so every input
    # pixel contributes (the antialiasing); the window is cut at the edges and renormalised.
    scale = n_in / n_out
    stretch = max(scale, 1.0)
    support = 2.0 * stretch
    weights = np.zeros((n_out, n_in))
    for i in range(n_out):
        center = (i + 0.5) * scale
        lo, hi = max(int(center - support + 0.5), 0), min(int(center + support + 0.5), n_in)
        w = _cubic((np.arange(lo, hi) - center + 0.5) / stretch)
        weights[i, lo:hi] = w / w.sum()
    return weights


def resize_bicubic_antialias(frame: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D frame as ``torch.nn.functional.interpolate(mode="bicubic", antialias=True)``.

    Separable, with half-pixel centres (``align_corners=False``) and the a = -0.5 cubic.
    """
    rows = _antialias_weights(frame.shape[0], out_hw[0])
    cols = _antialias_weights(frame.shape[1], out_hw[1])
    return (rows @ frame.astype(np.float64) @ cols.T).astype(np.float32)


def preprocess(
    depth_m,
    near_m: float = NEAR_M,
    far_m: float = FAR_M,
    invalid_fill: str = INVALID_FILL,
    *,
    raw_shape_hw: tuple[int, int] | None = None,
    resize: str | None = None,
) -> np.ndarray:
    """Resize a metric depth frame to 58x87 and normalise it to [-0.5, +0.5].

    Pixels with no reading (NaN, or zero or negative depth) take the ``invalid_fill`` plane;
    ``-inf`` (too close) reads as near and ``+inf`` (too far) as far whatever the fill. With
    ``raw_shape_hw`` a frame of any other size is refused, and ``resize`` names the resize.
    """
    frame = np.asarray(depth_m, dtype=np.float32)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    if frame.ndim != 2:
        raise ValueError(f"expected a 2-D depth frame, got shape {frame.shape}")
    if not far_m > near_m:
        raise ValueError(f"far_m must exceed near_m, got {near_m} and {far_m}")
    if invalid_fill not in INVALID_FILLS:
        raise ValueError(f"invalid_fill must be one of {INVALID_FILLS}, got {invalid_fill!r}")
    if resize is not None and resize not in RESIZES:
        raise ValueError(f"resize must be one of {RESIZES}, got {resize!r}")
    if raw_shape_hw is not None and frame.shape != tuple(raw_shape_hw):
        raise ValueError(
            f"{FRAME_SHAPE_MISMATCH}: got {frame.shape[0]}x{frame.shape[1]}, the model expects "
            f"{raw_shape_hw[0]}x{raw_shape_hw[1]}; publish depth at that resolution"
        )
    frame = frame.astype(np.float32, copy=True)
    # The ZED reports a pixel with no reading as NaN *and* as zero or negative, and clipping a
    # zero would read as an obstacle at the near plane whatever the model was trained on, so both
    # take the fill. For the primitive models that is far: a missed ray was written as 1000 m in
    # training and clamps to far. A continuous-velocity export declares its fill, and holosoma
    # applies whatever it declares rather than choosing one: the fill is part of the model's
    # contract with its depth source, so its export is where to change it.
    fill = far_m if invalid_fill == "far" else near_m
    frame[np.isneginf(frame)] = near_m
    frame[np.isposinf(frame)] = far_m
    frame[np.isnan(frame)] = fill
    frame[frame <= 0.0] = fill
    # Clip before resizing as well as after, so out-of-range pixels cannot bleed into their
    # neighbours through the interpolation kernel.
    frame = np.clip(frame, near_m, far_m)
    if resize == BICUBIC_ANTIALIAS:
        frame = resize_bicubic_antialias(frame, (HEIGHT, WIDTH))
    elif frame.shape != (HEIGHT, WIDTH):
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

    Each frame is preprocessed for the policy's active model: its declared depth contract when it
    has one (read per frame, so a policy switch brings its own), else ``near_m``/``far_m`` with
    the far fill. A frame the active contract cannot accept (the wrong size, or a contract that
    does not say how to frame one) is dropped and counted in ``rejected``, so the policy's depth
    goes stale and it stops rather than act on a misframed image.

    ``rclpy`` is imported on construction rather than at module scope: it is a ROS-deployment
    extra, and ``preprocess`` is useful without it.
    """

    def __init__(self, policy, topic: str = DEPTH_TOPIC, *, near_m: float = NEAR_M, far_m: float = FAR_M):
        self._policy = policy
        self._default = DepthSettings(near_m=near_m, far_m=far_m)
        # Fails here, before anything is subscribed, for a model this topic cannot feed.
        self.settings()
        self.frames = 0
        self.rejected = 0
        self._warned: set[str] = set()
        self._attach(topic)

    def settings(self) -> DepthSettings:
        """The preprocessing the active model needs; raises if its contract cannot frame a topic."""
        return settings_for(getattr(self._policy, "depth_contract", None), self._default)

    def _attach(self, topic: str) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

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
        self.push(frame_from_image(msg), stamp_seconds(msg))

    def push(self, depth_m, stamp_s=None) -> bool:
        """Preprocess one metric frame for the active model and hand it to the policy."""
        try:
            s = self.settings()
            frame = preprocess(depth_m, s.near_m, s.far_m, s.invalid_fill, raw_shape_hw=s.raw_shape_hw, resize=s.resize)
        except ValueError as exc:
            self.rejected += 1
            reason = str(exc)
            if reason not in self._warned:
                self._warned.add(reason)
                logger.error(f"[DepthConsumer] dropping depth frames: {reason}")
            return False
        if self._policy.set_depth(frame, stamp_s):
            self.frames += 1
            return True
        self.rejected += 1
        return False

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


class DepthShmConsumer:
    """Feeds a contextual policy from a depth shared-memory block, on its own thread.

    The producer writes frames already preprocessed for the policy (shape
    ``(num_cameras, 1, height, width)``), so frames pass through unchanged. With the producer's
    heartbeat (``holosoma_inference.sensors.depth_heartbeat``), a frame is pushed once per new
    ``seq``, stamped with its ``t_capture``; frames are never pushed before the block is mapped,
    since unmapped reads are zeros, and a zero frame is a valid depth reading. While a heartbeat is
    expected but none has arrived, nothing is pushed. With ``heartbeat=False`` every poll is pushed
    unstamped, and the policy cannot tell a stalled producer from a live one.

    An exception on the polling thread stops it; it is logged and re-raised by :meth:`stop`.
    """

    def __init__(
        self,
        policy,
        *,
        shape: tuple[int, ...],
        name: str = "depth_img_shm",
        heartbeat: bool = True,
        heartbeat_endpoint: str | None = None,
        camera: int = 0,
        poll_hz: float = 200.0,
        heartbeat_wait_warn_s: float = 2.0,
        clock=time.monotonic,
    ):
        from holosoma_inference.sensors.depth_heartbeat import default_heartbeat_endpoint
        from holosoma_inference.sensors.depth_shm import DepthShmSensor

        if len(shape) != 4 or shape[1] != 1:
            raise ValueError(f"expected a (num_cameras, 1, height, width) block, got shape {shape}")
        endpoint = (heartbeat_endpoint or default_heartbeat_endpoint(name)) if heartbeat else None
        self._sensor = DepthShmSensor(shape, name=name, required=False, heartbeat_endpoint=endpoint)
        self._name = name
        self._clock = clock
        self._heartbeat_wait_warn_s = heartbeat_wait_warn_s
        self._waiting_since: float | None = None
        self._warned_no_heartbeat = False
        self.error: BaseException | None = None
        self._policy = policy
        self._camera = camera
        self._period_s = 1.0 / poll_hz
        self._last_seq: int | None = None
        self.frames = 0
        self.rejected = 0
        self._stop = None
        self._thread = None

    def poll_once(self) -> bool:
        """Push the newest frame if there is one; returns whether a frame was pushed."""
        if not self._sensor.try_attach():
            return False
        liveness = self._sensor.liveness()
        if liveness is not None and liveness.checked:
            if liveness.state == "FAILED":
                return False  # the producer gave up; its block no longer receives frames
            if liveness.restarted:
                self._last_seq = None
            if liveness.seq == self._last_seq:
                return False
            self._last_seq = liveness.seq
            stamp = liveness.t_capture
        elif liveness is not None:
            self._warn_if_no_heartbeat()
            return False  # heartbeat expected but none received yet
        else:
            stamp = None
        frame = self._sensor.get_latest()[self._camera, 0]
        if self._policy.set_depth(frame, stamp):
            self.frames += 1
            return True
        self.rejected += 1
        return False

    def _warn_if_no_heartbeat(self) -> None:
        now = self._clock()
        if self._waiting_since is None:
            self._waiting_since = now
        if not self._warned_no_heartbeat and now - self._waiting_since > self._heartbeat_wait_warn_s:
            logger.warning(
                f"[DepthShmConsumer] no heartbeat from the producer of '{self._name}': no depth frames "
                f"will be delivered until one arrives. If the producer runs without a heartbeat, pass "
                f"--no-depth-heartbeat (frames are then unstamped and a stalled producer goes undetected)."
            )
            self._warned_no_heartbeat = True

    def start(self) -> None:
        """Poll the block on a daemon thread, leaving the policy loop blocking."""
        import threading

        if self._thread is not None:
            return
        self._sensor.start()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self.poll_once()
                self._stop.wait(self._period_s)
        except Exception as exc:
            # The policy keeps its last frame; nothing is pushed from here on.
            logger.error(f"[DepthShmConsumer] stopped reading '{self._name}': {exc}")
            self.error = exc

    def stop(self) -> None:
        """Stop polling; re-raises an exception that stopped the polling thread."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        self._sensor.stop()
        if self.error is not None:
            raise RuntimeError(f"depth shared-memory consumer for '{self._name}' failed") from self.error
