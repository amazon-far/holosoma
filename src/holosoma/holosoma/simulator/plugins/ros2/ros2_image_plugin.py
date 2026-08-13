"""ROS2 image-publishing plugin: publishes rendered sim camera frames as sensor_msgs/Image or
CompressedImage (+ optional latched CameraInfo), one node fanning out to many camera routes.

This is the only camera-egress module that touches ROS2. It is imported solely via
``ROS2ImagePluginConfig.get_cls`` (deferred), so the rest of holosoma stays importable without
``rclpy``. The heavy imports (rclpy, sensor_msgs) are deferred further into ``start()`` so even
importing this module does not hard-require a ROS environment — only constructing+starting the
plugin does. All non-ROS logic (encoding, K-matrix, drop-oldest queue) lives in sibling ROS-free
modules (``encode``/``camera_info``/``worker``) and is unit-tested there.

Per route: when ``async_publish`` the sim thread submits the encoded payload to a per-route
drop-oldest worker (no head-of-line blocking, sim never waits); otherwise it encodes+publishes
inline (lossless/every-frame). Timestamps come from the frame's sim_time, not wall-clock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from holosoma.simulator.plugins.camera_consumer import CameraConsumerPlugin, CameraIntrinsics
from holosoma.simulator.plugins.ros2.camera_info import camera_info_from_intrinsics
from holosoma.simulator.plugins.ros2.encode import EncodedImage, encode_frame
from holosoma.simulator.plugins.ros2.worker import PublishWorker

if TYPE_CHECKING:
    from holosoma.config_types.plugin import ROS2ImagePluginConfig, ROS2ImageRoute
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.plugins.camera_consumer import FramePacket, StreamKey


def _sim_time_to_stamp(sim_time: float):
    """Build a builtin_interfaces/Time from sim seconds (deferred import; only after start())."""
    from builtin_interfaces.msg import Time

    sec = int(sim_time)
    nanosec = round((sim_time - sec) * 1e9)
    if nanosec >= 1_000_000_000:  # rounding carry
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


class ROS2ImagePlugin(CameraConsumerPlugin):
    """One ROS2 node publishing the configured camera routes (a camera-consumer plugin)."""

    config: ROS2ImagePluginConfig

    def __init__(self, config: ROS2ImagePluginConfig, simulator: BaseSimulator) -> None:
        # rclpy/sensor_msgs objects are typed Any: their stubs are absent in non-ROS envs (e.g. the
        # mujoco venv), and these are only populated in start() under a real ROS environment.
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: Any = None
        self._publishers: dict[str, Any] = {}  # topic -> rclpy publisher
        self._info_publishers: dict[str, Any] = {}  # camera -> CameraInfo publisher (latched)
        self._workers: dict[str, PublishWorker[FramePacket]] = {}  # topic -> worker (async only)
        # Validates routes against the configured cameras and registers the publish/close callbacks.
        super().__init__(config, simulator)

    def wanted_streams(self) -> set[StreamKey]:
        # One env per node (config.env_id; default 0 = the single real-time robot).
        return {(r.camera, r.modality, self.config.env_id) for r in self.config.routes.values()}

    # ----- lifecycle -----

    def start(self) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image

        if not rclpy.ok():
            rclpy.init()
        self._node = Node(self.config.node_name)
        self._Image = Image
        self._CompressedImage = CompressedImage
        self._CameraInfo = CameraInfo

        reliability = ReliabilityPolicy.RELIABLE if self.config.qos == "reliable" else ReliabilityPolicy.BEST_EFFORT
        qos = QoSProfile(reliability=reliability, history=HistoryPolicy.KEEP_LAST, depth=1)

        for route in self.config.routes.values():
            msg_type = self._CompressedImage if route.format in ("jpeg", "png") else self._Image
            # Publish on the route's topic verbatim — no auto-suffixing. A CompressedImage topic
            # conventionally ends in `/compressed`, but that is the config author's choice to spell
            # out, so the configured topic is exactly the published topic.
            self._publishers[route.topic] = self._node.create_publisher(msg_type, route.topic, qos)
            if self.config.async_publish:
                worker = PublishWorker(
                    self._make_route_sender(route),
                    maxlen=self.config.queue_maxlen,
                    name=f"egress:{self.config.node_name}:{route.topic}",
                )
                worker.start()
                self._workers[route.topic] = worker

        if self.config.publish_camera_info:
            self._start_camera_info(qos_history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # Spin in a daemon thread so subscriptions/QoS handshakes progress without a sim-side spin.
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        import threading

        self._spin_thread = threading.Thread(
            target=self._executor.spin, name=f"egress-spin:{self.config.node_name}", daemon=True
        )
        self._spin_thread.start()
        logger.info(f"ROS2 image egress '{self.config.node_name}' up: {len(self.config.routes)} route(s)")

    def _start_camera_info(self, *, qos_history, durability) -> None:
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        # CameraInfo is STATIC: derived from the camera's configured intrinsics and published ONCE
        # here on a latched (TRANSIENT_LOCAL) topic, so late subscribers still get it and it never
        # rides the per-frame path. Topic = the ROS-conventional sibling of the image topic: its last
        # segment replaced by ``camera_info`` (e.g. ``/cam/head/compressed`` -> ``/cam/head/camera_info``).
        # Image topics are validated unique within the node, so these are unique too.
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=qos_history, depth=1, durability=durability
        )
        cams_by_name = dict(self.sensors_config)
        seen: set[str] = set()
        for route in self.config.routes.values():
            if route.camera in seen:
                continue
            seen.add(route.camera)
            info_topic = f"{route.topic.rsplit('/', 1)[0]}/camera_info"
            pub = self._node.create_publisher(self._CameraInfo, info_topic, latched)
            self._info_publishers[route.camera] = pub
            cam = cams_by_name[route.camera]  # present: driver validated routes against sensors_config
            intr = CameraIntrinsics(
                width=cam.width, height=cam.height, vertical_fov=cam.vertical_fov, near=cam.near, far=cam.far
            )
            pub.publish(self._build_camera_info_msg(route.camera, intr))

    # ----- per-frame publish -----

    def publish(self, frames: dict[StreamKey, FramePacket]) -> None:
        # One batch per step; each route consumes its (camera, modality, 0) packet if present.
        # Encode on the calling (sim) thread when inline; when async, hand the packet to the route's
        # worker and encode there to keep the sim thread's per-frame cost to the snapshot only.
        for route in self.config.routes.values():
            packet = frames.get((route.camera, route.modality, self.config.env_id))
            if packet is None:
                continue
            if self.config.async_publish:
                self._workers[route.topic].submit(packet)
            else:
                self._publish_encoded(route, packet, self._encode(route, packet))

    def _encode(self, route: ROS2ImageRoute, packet: FramePacket) -> EncodedImage:
        """Encode a packet for ``route``, colorizing a depth frame when the format is an rgb one."""
        return encode_frame(
            packet.array,
            route.format,
            modality=route.modality,
            jpeg_quality=self.config.jpeg_quality,
            depth_colormap=route.depth_colormap,
            depth_range=(route.depth_range[0], route.depth_range[1]) if route.depth_range is not None else None,
        )

    def _make_route_sender(self, route: ROS2ImageRoute):
        """Return a worker callback that encodes a FramePacket and publishes it for ``route``."""

        def _send(packet: FramePacket) -> None:
            self._publish_encoded(route, packet, self._encode(route, packet))

        return _send

    def _publish_encoded(self, route: ROS2ImageRoute, packet: FramePacket, enc: EncodedImage) -> None:
        pub = self._publishers.get(route.topic)
        if pub is None:
            return
        stamp = _sim_time_to_stamp(packet.sim_time)
        if enc.compressed:
            msg = self._CompressedImage()
            msg.header.stamp = stamp
            msg.header.frame_id = packet.camera
            msg.format = enc.compressed_format
            msg.data = enc.data
        else:
            msg = self._Image()
            msg.header.stamp = stamp
            msg.header.frame_id = packet.camera
            msg.height = enc.height
            msg.width = enc.width
            msg.encoding = enc.encoding
            msg.is_bigendian = 0
            msg.step = enc.step
            msg.data = enc.data
        pub.publish(msg)

    def _build_camera_info_msg(self, camera: str, intr: CameraIntrinsics):
        """Build a static CameraInfo message for ``camera`` (published once, latched, in start())."""
        info = camera_info_from_intrinsics(intr)
        msg = self._CameraInfo()
        msg.header.frame_id = camera  # static: no per-frame stamp
        msg.height = info.height
        msg.width = info.width
        msg.distortion_model = info.distortion_model
        msg.d = info.d
        msg.k = info.k
        msg.r = info.r
        msg.p = info.p
        return msg

    # ----- teardown -----

    def stop(self) -> None:
        for worker in self._workers.values():
            worker.stop()
        self._workers.clear()
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self._spin_thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        self._publishers.clear()
        self._info_publishers.clear()
