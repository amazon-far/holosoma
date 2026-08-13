"""Offline behavior tests for ROS2ImagePlugin message construction (no DDS, no sim, no spin).

ROS2-behavior is tested the way PR #124 does it: ``pytest.importorskip("rclpy")`` so the test runs
in the ROS env (bazel/RoboStack) and skips cleanly on a host without ROS. The production code is
left untouched — it always creates and spins its own node (nothing in holosoma shares a node with
it, so adding an inject-a-node hook would be test-only contamination). Instead the FIXTURE
monkeypatches the rclpy pieces ``start()`` reaches for (``Node`` -> a fake that records publishers;
the executor + spin thread -> no-ops), so publishers are captured and no real DDS graph / spin /
``rclpy.init`` happens. We then drive ``publish()`` / start-time CameraInfo directly and assert the
exact messages built (topic, type, encoding, frame_id, stamp, K) — the logic the ROS-free helpers
(encode/camera_info) cannot cover on their own.

Live end-to-end publishing over DDS is validated manually on the cluster (see
``../ros2/INTEGRATION_TEST.md``), not in this suite.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("sensor_msgs")

from holosoma.config_types.plugin import ROS2ImagePluginConfig, ROS2ImageRoute
from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.simulator.base_simulator.hooks import HookRegistry
from holosoma.simulator.plugins.camera_consumer import CameraIntrinsics, FramePacket

_MOUNT = SensorMountConfig(target_kind="robot_link", target="torso_link")


class _FakeSimEngineCfg:
    fps = 200.0
    control_decimation_steps = 4


class _FakeSimulatorConfig:
    sim = _FakeSimEngineCfg()


class _FakeVideoConfig:
    save_dir = None


class _FakeTrainingConfig:
    num_envs = 1


class _FakeSimulator:
    """Minimal stand-in exposing what the egress hook reads/registers on (no backend, no DDS)."""

    def __init__(self, *cams):
        self.hooks = HookRegistry()
        self.sensor_config = dict(cams)
        self.training_config = _FakeTrainingConfig()
        self.num_envs = 1
        self.headless = True
        self.simulator_config = _FakeSimulatorConfig()
        self.video_config = _FakeVideoConfig()
        self.sensor_manager = None

    def sensor_config_by_name(self, name):
        return self.sensor_config[name]


class _FakePublisher:
    """Captures every published message instead of sending it on DDS."""

    def __init__(self, msg_type, topic):
        self.msg_type = msg_type
        self.topic = topic
        self.published: list = []

    def publish(self, msg):
        self.published.append(msg)


class _FakeNode:
    """Stand-in for an rclpy Node: records publishers, never touches DDS."""

    def __init__(self, *args, **kwargs):
        self.publishers: list[_FakePublisher] = []

    def create_publisher(self, msg_type, topic, qos):
        pub = _FakePublisher(msg_type, topic)
        self.publishers.append(pub)
        return pub

    def destroy_node(self):
        pass


class _NoopExecutor:
    def add_node(self, node):
        pass

    def spin(self):  # referenced as the spin-thread target; never actually run (thread is a no-op)
        pass

    def shutdown(self):
        pass


class _NoopThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def is_alive(self):
        return False


@pytest.fixture
def fake_ros(monkeypatch):
    """Monkeypatch the rclpy pieces ``ROS2ImagePlugin.start`` uses so no real node/spin/DDS runs.

    Patches at the egress module's import sites (``start`` does ``from rclpy.node import Node`` etc.
    at call time, so we patch the source modules). Captures the constructed fake node on the
    returned holder so tests can inspect the publishers.
    """
    import threading

    import rclpy
    import rclpy.executors
    import rclpy.node

    holder = {}

    def _node_factory(*args, **kwargs):
        node = _FakeNode(*args, **kwargs)
        holder["node"] = node
        return node

    # start() does `from rclpy.node import Node`, `from rclpy.executors import SingleThreadedExecutor`
    # and `import threading` at call time, so patching these source modules covers those binds.
    monkeypatch.setattr(rclpy.node, "Node", _node_factory)
    monkeypatch.setattr(rclpy, "ok", lambda: True)  # skip rclpy.init()
    monkeypatch.setattr(rclpy.executors, "SingleThreadedExecutor", _NoopExecutor)
    monkeypatch.setattr(threading, "Thread", _NoopThread)
    return holder


def _make(cfg, *cams):
    """Construct the egress hook with a fake simulator built from the given cameras."""
    return cfg.get_cls()(cfg, _FakeSimulator(*cams))


def _cam(name, **kw):
    """Return a ``(name, CameraSensorConfig)`` pair for building the fake sim's cameras dict."""
    return name, CameraSensorConfig(mount=_MOUNT, **kw)


def _rgb_packet(camera, h=8, w=8, sim_time=1.5):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = 200  # distinctive R so we can tell the decode apart
    intr = CameraIntrinsics(width=w, height=h, vertical_fov=45.0, near=0.01, far=100.0)
    return FramePacket(camera=camera, modality="rgb", env_id=0, array=arr, sim_time=sim_time, intrinsics=intr)


def _depth_packet(camera, h=8, w=8, sim_time=1.5):
    arr = np.full((h, w, 1), 1.0, dtype=np.float32)  # [H,W,1] float meters, as get_camera_data gives
    intr = CameraIntrinsics(width=w, height=h, vertical_fov=45.0, near=0.01, far=100.0)
    return FramePacket(camera=camera, modality="depth", env_id=0, array=arr, sim_time=sim_time, intrinsics=intr)


def _publish(egress, packet):
    """Hand the egress a single-packet batch (publish always takes a StreamKey->packet dict)."""
    egress.publish({packet.key: packet})


def _start(egress, fake_ros):
    """Start the egress under the monkeypatched rclpy; return the captured fake node."""
    egress.start()
    return fake_ros["node"]


def _pub_for(node, topic):
    for p in node.publishers:
        if p.topic == topic:
            return p
    raise AssertionError(f"no publisher created for topic {topic}; have {[p.topic for p in node.publishers]}")


# ----- topic + type wiring -----


def test_jpeg_route_publishes_compressedimage_on_verbatim_topic(fake_ros):
    from sensor_msgs.msg import CompressedImage

    cfg = ROS2ImagePluginConfig(
        async_publish=False,
        publish_camera_info=False,
        routes={"head": ROS2ImageRoute(camera="head", topic="/cam/head/compressed", modality="rgb", format="jpeg")},
    )
    egress = _make(cfg, _cam("head"))
    node = _start(egress, fake_ros)
    # jpeg => CompressedImage on the VERBATIM configured topic (no auto-suffixing).
    pub = _pub_for(node, "/cam/head/compressed")
    assert pub.msg_type is CompressedImage

    _publish(egress, _rgb_packet("head"))
    assert len(pub.published) == 1
    msg = pub.published[0]
    assert msg.format == "jpeg"
    assert msg.header.frame_id == "head"
    assert len(msg.data) > 0
    egress.stop()


def test_raw_rgb8_route_publishes_image_with_dimensions(fake_ros):
    from sensor_msgs.msg import Image

    cfg = ROS2ImagePluginConfig(
        async_publish=False,
        publish_camera_info=False,
        routes={"head": ROS2ImageRoute(camera="head", topic="/cam/head/image_raw", modality="rgb", format="rgb8")},
    )
    egress = _make(cfg, _cam("head"))
    node = _start(egress, fake_ros)
    # rgb8 => raw Image on the un-suffixed topic.
    pub = _pub_for(node, "/cam/head/image_raw")
    assert pub.msg_type is Image

    _publish(egress, _rgb_packet("head", h=8, w=8))
    msg = pub.published[0]
    assert msg.encoding == "rgb8"
    assert (msg.height, msg.width, msg.step) == (8, 8, 24)  # step = 3*w
    assert msg.is_bigendian == 0
    assert len(msg.data) == 8 * 8 * 3
    egress.stop()


def test_sim_time_becomes_message_stamp(fake_ros):
    cfg = ROS2ImagePluginConfig(
        async_publish=False,
        publish_camera_info=False,
        routes={"head": ROS2ImageRoute(camera="head", topic="/cam/head/compressed", modality="rgb", format="jpeg")},
    )
    egress = _make(cfg, _cam("head"))
    node = _start(egress, fake_ros)
    _publish(egress, _rgb_packet("head", sim_time=2.25))
    stamp = _pub_for(node, "/cam/head/compressed").published[0].header.stamp
    # 2.25 s -> sec=2, nanosec=0.25e9. Stamp is built from sim_time, NOT wall-clock.
    assert stamp.sec == 2
    assert stamp.nanosec == 250_000_000
    egress.stop()


def test_publish_routes_only_matching_camera_modality(fake_ros):
    cfg = ROS2ImagePluginConfig(
        async_publish=False,
        publish_camera_info=False,
        routes={
            "head": ROS2ImageRoute(camera="head", topic="/cam/head/compressed", modality="rgb", format="jpeg"),
            "wrist": ROS2ImageRoute(camera="wrist", topic="/cam/wrist/compressed", modality="rgb", format="jpeg"),
        },
    )
    egress = _make(cfg, _cam("head"), _cam("wrist"))
    node = _start(egress, fake_ros)
    _publish(egress, _rgb_packet("head"))  # only the head packet
    assert len(_pub_for(node, "/cam/head/compressed").published) == 1
    assert len(_pub_for(node, "/cam/wrist/compressed").published) == 0
    egress.stop()


# ----- colorized depth (depth modality + rgb format) -----


def test_depth_route_colorized_publishes_compressed_rgb(fake_ros):
    import cv2
    from sensor_msgs.msg import CompressedImage

    # A depth camera routed on an rgb (jpeg) format => the depth map is colorized to RGB, then
    # published as CompressedImage. The raw float32 depth never reaches the wire on this route.
    cfg = ROS2ImagePluginConfig(
        async_publish=False,
        publish_camera_info=False,
        routes={
            "waist": ROS2ImageRoute(
                camera="waist",
                topic="/cam/waist/depth_color/compressed",
                modality="depth",
                format="jpeg",
                depth_colormap="turbo",
                depth_range=[0.1, 4.0],
            )
        },
    )
    egress = _make(cfg, _cam("waist", data_types=["depth"]))
    node = _start(egress, fake_ros)
    pub = _pub_for(node, "/cam/waist/depth_color/compressed")
    assert pub.msg_type is CompressedImage

    egress.publish({("waist", "depth", 0): _depth_packet("waist")})
    assert len(pub.published) == 1
    msg = pub.published[0]
    assert msg.format == "jpeg"
    assert msg.header.frame_id == "waist"
    # Decodes back to a 3-channel RGB image (colorized), not raw depth.
    bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    assert bgr.shape == (8, 8, 3)
    egress.stop()


def test_depth_route_raw_still_publishes_float_image(fake_ros):
    from sensor_msgs.msg import Image

    # A depth route on a depth format (32FC1) is unchanged: raw float32-meter Image, not colorized.
    cfg = ROS2ImagePluginConfig(
        async_publish=False,
        publish_camera_info=False,
        routes={"waist": ROS2ImageRoute(camera="waist", topic="/cam/waist/depth", modality="depth", format="32FC1")},
    )
    egress = _make(cfg, _cam("waist", data_types=["depth"]))
    node = _start(egress, fake_ros)
    pub = _pub_for(node, "/cam/waist/depth")
    assert pub.msg_type is Image

    egress.publish({("waist", "depth", 0): _depth_packet("waist", h=8, w=8)})
    msg = pub.published[0]
    assert msg.encoding == "32FC1"
    assert (msg.height, msg.width, msg.step) == (8, 8, 32)  # 4 bytes * w
    assert len(msg.data) == 8 * 8 * 4
    egress.stop()


# ----- latched CameraInfo (published once at start, not per-frame) -----


def test_camera_info_published_once_at_start_with_correct_k(fake_ros):
    import math

    cfg = ROS2ImagePluginConfig(
        async_publish=False,
        publish_camera_info=True,
        routes={"head": ROS2ImageRoute(camera="head", topic="/cam/head/image", modality="rgb", format="rgb8")},
    )
    egress = _make(cfg, _cam("head", width=320, height=240, vertical_fov=60.0))
    node = _start(egress, fake_ros)
    # CameraInfo topic is the ROS-conventional sibling of the image topic (last segment -> camera_info).
    info_pub = _pub_for(node, "/cam/head/camera_info")
    # Published exactly once, during start() — before any frame.
    assert len(info_pub.published) == 1
    info = info_pub.published[0]
    assert (info.width, info.height) == (320, 240)
    f = (240 / 2.0) / math.tan(math.radians(60.0) / 2.0)
    assert info.k[0] == pytest.approx(f) and info.k[4] == pytest.approx(f)
    assert info.k[2] == pytest.approx(160.0) and info.k[5] == pytest.approx(120.0)

    # A frame must NOT add another CameraInfo (it is static/latched, off the per-frame path).
    _publish(egress, _rgb_packet("head", h=240, w=320))
    assert len(info_pub.published) == 1
    egress.stop()


def test_no_camera_info_when_disabled(fake_ros):
    cfg = ROS2ImagePluginConfig(
        async_publish=False,
        publish_camera_info=False,
        routes={"head": ROS2ImageRoute(camera="head", topic="/cam/head", modality="rgb", format="jpeg")},
    )
    egress = _make(cfg, _cam("head"))
    node = _start(egress, fake_ros)
    assert not any(p.topic.endswith("/camera_info") for p in node.publishers)
    egress.stop()


# ----- teardown -----


def test_stop_tears_down_cleanly(fake_ros):
    cfg = ROS2ImagePluginConfig(
        async_publish=False,
        publish_camera_info=False,
        routes={"head": ROS2ImageRoute(camera="head", topic="/cam/head", modality="rgb", format="jpeg")},
    )
    egress = _make(cfg, _cam("head"))
    _start(egress, fake_ros)
    egress.stop()  # must not raise; clears node/executor/publishers
    assert egress._node is None
    assert egress._publishers == {}
