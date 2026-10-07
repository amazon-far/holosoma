# mypy: disable-error-code="attr-defined,possibly-undefined"
"""ROS2 PointCloud2 message tests using isolated fake ROS modules."""

from __future__ import annotations

import os
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from holosoma.config_types.plugin import ROS2PointCloudPluginConfig, ROS2PointCloudRoute
from holosoma.config_types.sensor import GridLidarRayPatternConfig, LidarSensorConfig, SensorMountConfig
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.plugins.lidar_consumer import PointCloudPacket
from holosoma.simulator.plugins.ros2 import ros2_pointcloud_plugin
from holosoma.simulator.plugins.ros2.ros2_pointcloud_plugin import ROS2PointCloudPlugin
from holosoma.simulator.shared.sensor_manager import SensorManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


class _TrainingConfig:
    num_envs = 1


class _FakeSimulator:
    def __init__(self, *, include_rear_lidar: bool = False) -> None:
        self.hooks = HookRegistry()
        lidar = LidarSensorConfig(
            mount=SensorMountConfig(target_kind="robot_link", target="pelvis"),
            pattern=GridLidarRayPatternConfig(horizontal_angles=[0.0, 90.0]),
            update_decimation=2,
        )
        self.sensor_config = {"lidar": lidar}
        self.training_config = _TrainingConfig()
        self.sensor_manager = SensorManager("cpu", control_hz=50.0)
        self.sensor_manager.register_lidar("lidar", lidar)
        if include_rear_lidar:
            rear_lidar = LidarSensorConfig(
                mount=SensorMountConfig(target_kind="robot_link", target="pelvis"),
                pattern=GridLidarRayPatternConfig(horizontal_angles=[180.0]),
                update_decimation=2,
            )
            self.sensor_config["rear_lidar"] = rear_lidar
            self.sensor_manager.register_lidar("rear_lidar", rear_lidar)
        self._time = 1.25

    def time(self) -> float:
        return self._time

    def get_lidar_data(
        self,
        name: str,
        data_type: str = "points",
        env_ids: Any = None,
        device: Any = None,
    ) -> torch.Tensor:
        buffer = self.sensor_manager.get_lidar(name).buffer_on(data_type, device)
        return buffer if env_ids is None else buffer[env_ids]


class _Time:
    def __init__(self, *, sec: int = 0, nanosec: int = 0) -> None:
        self.sec = sec
        self.nanosec = nanosec


class _PointField:
    FLOAT32 = 7

    def __init__(self, *, name: str, offset: int, datatype: int, count: int) -> None:
        self.name = name
        self.offset = offset
        self.datatype = datatype
        self.count = count


class _PointCloud2:
    def __init__(self) -> None:
        self.header = SimpleNamespace(stamp=_Time(), frame_id="")
        self.height = 0
        self.width = 0
        self.fields: list[_PointField] = []
        self.is_bigendian = False
        self.point_step = 0
        self.row_step = 0
        self.data = b""
        self.is_dense = False


class _FakePublisher:
    def __init__(self, msg_type: type, topic: str, qos: Any) -> None:
        self.msg_type = msg_type
        self.topic = topic
        self.qos = qos
        self.published: list[Any] = []

    def publish(self, msg: Any) -> None:
        self.published.append(msg)


class _FakeNode:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.publishers: list[_FakePublisher] = []
        self.destroy_count = 0

    def create_publisher(self, msg_type: type, topic: str, qos: Any) -> _FakePublisher:
        publisher = _FakePublisher(msg_type, topic, qos)
        self.publishers.append(publisher)
        return publisher

    def destroy_node(self) -> None:
        self.destroy_count += 1


class _NoopExecutor:
    instances: list[_NoopExecutor] = []

    def __init__(self, *, context: Any = None) -> None:
        self.node: Any = None
        self.shutdown_count = 0
        self.instances.append(self)

    def add_node(self, node: Any) -> None:
        self.node = node

    def spin_once(self, timeout_sec: float | None = None) -> None:
        pass

    def shutdown(self, timeout_sec: float | None = None) -> bool:
        self.shutdown_count += 1
        return True

    def remove_node(self, node: Any) -> None:
        assert node is self.node
        self.node = None


class _NoopThread:
    instances: list[_NoopThread] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.alive = False
        self.join_count = 0
        self.instances.append(self)

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.join_count += 1
        self.alive = False


def _module(name: str, **members: Any) -> ModuleType:
    module = ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def fake_ros(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    holder: dict[str, Any] = {}
    _NoopExecutor.instances.clear()
    _NoopThread.instances.clear()

    def _node_factory(*args: Any, **kwargs: Any) -> _FakeNode:
        node = _FakeNode(*args, **kwargs)
        holder["node"] = node
        holder["node_count"] = holder.get("node_count", 0) + 1
        return node

    class _QoSProfile:
        def __init__(self, *, reliability: str, history: str, depth: int) -> None:
            self.reliability = reliability
            self.history = history
            self.depth = depth

    rclpy = _module("rclpy", ok=lambda: True, init=lambda: None)
    rclpy_node = _module("rclpy.node", Node=_node_factory)
    rclpy_executors = _module("rclpy.executors", SingleThreadedExecutor=_NoopExecutor)
    rclpy_qos = _module(
        "rclpy.qos",
        HistoryPolicy=SimpleNamespace(KEEP_LAST="keep_last"),
        ReliabilityPolicy=SimpleNamespace(RELIABLE="reliable", BEST_EFFORT="best_effort"),
        QoSProfile=_QoSProfile,
    )
    sensor_msgs = _module("sensor_msgs")
    sensor_msgs_msg = _module("sensor_msgs.msg", PointCloud2=_PointCloud2, PointField=_PointField)
    builtin_interfaces = _module("builtin_interfaces")
    builtin_interfaces_msg = _module("builtin_interfaces.msg", Time=_Time)

    rclpy.node = rclpy_node
    rclpy.executors = rclpy_executors
    rclpy.qos = rclpy_qos
    sensor_msgs.msg = sensor_msgs_msg
    builtin_interfaces.msg = builtin_interfaces_msg
    for module in (
        rclpy,
        rclpy_node,
        rclpy_executors,
        rclpy_qos,
        sensor_msgs,
        sensor_msgs_msg,
        builtin_interfaces,
        builtin_interfaces_msg,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(threading, "Thread", _NoopThread)
    runtime = type("_Runtime", (), {"start": lambda _self: object()})()
    monkeypatch.setattr(ros2_pointcloud_plugin, "get_ros2_runtime", lambda _simulator: runtime)
    holder["PointCloud2"] = _PointCloud2
    return holder


def _start_plugin(
    fake_ros: dict[str, Any],
    *,
    frame_id: str | None = "lidar_link",
    qos: str = "best_effort",
    async_publish: bool | None = False,
) -> tuple[ROS2PointCloudPlugin, _FakePublisher]:
    route = ROS2PointCloudRoute(lidar="lidar", topic="/scan", frame_id=frame_id)
    kwargs: dict[str, Any] = {} if async_publish is None else {"async_publish": async_publish}
    config = ROS2PointCloudPluginConfig(qos=qos, routes={"scan": route}, **kwargs)
    plugin = cast("ROS2PointCloudPlugin", config.get_cls()(config, _FakeSimulator()))
    plugin.start()
    publisher = cast("_FakeNode", fake_ros["node"]).publishers[0]
    return plugin, publisher


def test_publishes_xyz_cloud_with_sim_stamp_and_field_metadata(fake_ros: dict[str, Any]) -> None:
    plugin, publisher = _start_plugin(fake_ros)
    assert publisher.msg_type is fake_ros["PointCloud2"]
    assert publisher.topic == "/scan"

    points = np.array(
        [[1.0, 2.0, 3.0], [float("nan"), float("nan"), float("nan")]],
        dtype=np.float32,
    )
    packet = PointCloudPacket(
        lidar="lidar",
        env_id=0,
        points=points,
        sim_time=2.25,
        height=1,
        width=2,
    )
    plugin.publish({packet.key: packet})

    msg = publisher.published[0]
    assert msg.header.frame_id == "lidar_link"
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (2, 250_000_000)
    assert (msg.height, msg.width, msg.point_step, msg.row_step) == (1, 2, 12, 24)
    assert [(field.name, field.offset, field.datatype, field.count) for field in msg.fields] == [
        ("x", 0, _PointField.FLOAT32, 1),
        ("y", 4, _PointField.FLOAT32, 1),
        ("z", 8, _PointField.FLOAT32, 1),
    ]
    assert not msg.is_bigendian
    assert not msg.is_dense
    decoded = np.frombuffer(msg.data, dtype="<f4").reshape(2, 3)
    assert np.array_equal(decoded[0], points[0])
    assert np.isnan(decoded[1]).all()
    plugin.stop()


def test_constructs_real_sensor_msgs_pointcloud2_when_ros_is_installed() -> None:
    """Exercise the real generated ROS message classes without starting DDS or a simulator."""
    try:
        from builtin_interfaces import msg as builtin_interfaces
        from sensor_msgs import msg as sensor_msgs
    except ImportError:
        if os.environ.get("HOLOSOMA_REQUIRE_ROS2_POINTCLOUD_TEST") == "1":
            pytest.fail("strict ROS2 PointCloud2 test requires builtin_interfaces.msg and sensor_msgs.msg")
        pytest.skip("ROS2 generated Python message packages are not installed")
    publisher = _FakePublisher(sensor_msgs.PointCloud2, "/scan", qos=None)
    plugin = object.__new__(ROS2PointCloudPlugin)
    plugin._PointCloud2 = sensor_msgs.PointCloud2
    plugin._PointField = sensor_msgs.PointField
    plugin._publishers = {"/scan": publisher}
    route = ROS2PointCloudRoute(lidar="lidar", topic="/scan", frame_id="lidar_link")
    packet = PointCloudPacket(
        lidar="lidar",
        env_id=0,
        points=np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]], dtype=np.float32),
        sim_time=2.25,
        height=1,
        width=2,
    )

    plugin._publish_encoded(route, packet, plugin._encode(route, packet))

    assert len(publisher.published) == 1
    msg = publisher.published[0]
    assert isinstance(msg, sensor_msgs.PointCloud2)
    assert isinstance(msg.header.stamp, builtin_interfaces.Time)
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (2, 250_000_000)
    assert [(field.name, field.offset, field.datatype, field.count) for field in msg.fields] == [
        ("x", 0, sensor_msgs.PointField.FLOAT32, 1),
        ("y", 4, sensor_msgs.PointField.FLOAT32, 1),
        ("z", 8, sensor_msgs.PointField.FLOAT32, 1),
    ]
    assert (msg.height, msg.width, msg.point_step, msg.row_step) == (1, 2, 12, 24)
    decoded = np.frombuffer(msg.data, dtype="<f4").reshape(2, 3)
    assert np.array_equal(decoded[0], packet.points[0])
    assert np.isnan(decoded[1]).all()
    assert not msg.is_dense


def test_route_transform_rotates_and_translates_points_into_frame(fake_ros: dict[str, Any]) -> None:
    route = ROS2PointCloudRoute(
        lidar="lidar",
        topic="/scan",
        frame_id="mid360_link",
        point_rotation=[0.5000005, 0.5000005, -0.5000005, -0.5000005],
        point_translation=[1.0, 2.0, 3.0],
    )
    config = ROS2PointCloudPluginConfig(async_publish=False, routes={"scan": route})
    plugin = cast("ROS2PointCloudPlugin", config.get_cls()(config, _FakeSimulator()))
    plugin.start()
    packet = PointCloudPacket(
        lidar="lidar",
        env_id=0,
        points=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        ),
        sim_time=2.25,
        height=1,
        width=4,
    )

    plugin.publish({packet.key: packet})

    msg = cast("_FakeNode", fake_ros["node"]).publishers[0].published[0]
    points = np.frombuffer(msg.data, dtype="<f4").reshape(4, 3)
    np.testing.assert_allclose(
        points[:3],
        [
            [1.0, 1.0, 3.0],
            [1.0, 2.0, 4.0],
            [0.0, 2.0, 3.0],
        ],
        atol=1e-6,
    )
    assert np.isnan(points[3]).all()
    assert msg.header.frame_id == "mid360_link"
    assert not msg.is_dense
    plugin.stop()


def test_fans_out_one_fresh_scan_to_every_configured_route(fake_ros: dict[str, Any]) -> None:
    routes = {
        "primary": ROS2PointCloudRoute(lidar="lidar", topic="/lidar/points", frame_id="lidar"),
        "debug": ROS2PointCloudRoute(lidar="lidar", topic="/debug/lidar/points", frame_id="lidar_debug"),
    }
    config = ROS2PointCloudPluginConfig(async_publish=False, routes=routes)
    plugin = cast("ROS2PointCloudPlugin", config.get_cls()(config, _FakeSimulator()))
    plugin.start()
    packet = PointCloudPacket(
        lidar="lidar",
        env_id=0,
        points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        sim_time=4.5,
        height=1,
        width=1,
    )

    plugin.publish({packet.key: packet})

    publishers = {publisher.topic: publisher for publisher in cast("_FakeNode", fake_ros["node"]).publishers}
    assert set(publishers) == {"/lidar/points", "/debug/lidar/points"}
    for topic, frame_id in (("/lidar/points", "lidar"), ("/debug/lidar/points", "lidar_debug")):
        published = publishers[topic].published
        assert len(published) == 1
        assert published[0].header.frame_id == frame_id
        assert np.array_equal(np.frombuffer(published[0].data, dtype="<f4").reshape(1, 3), packet.points)
    plugin.stop()


def test_async_fanout_routes_each_lidar_packet_to_its_configured_topics(
    fake_ros: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    workers: list[Any] = []

    class _ImmediateWorker:
        def __init__(self, publish_fn: Any, *, maxlen: int, name: str) -> None:
            self.publish_fn = publish_fn
            self.started = False
            self.submitted: list[PointCloudPacket] = []
            workers.append(self)

        def start(self) -> None:
            self.started = True

        def submit(self, packet: PointCloudPacket) -> None:
            self.submitted.append(packet)
            self.publish_fn(packet)

        def stop(self) -> None:
            pass

    monkeypatch.setattr(ros2_pointcloud_plugin, "PublishWorker", _ImmediateWorker)
    routes = {
        "front": ROS2PointCloudRoute(
            lidar="lidar",
            topic="/front/points",
            frame_id="front_lidar",
            point_rotation=[0.5, 0.5, -0.5, -0.5],
        ),
        "front_debug": ROS2PointCloudRoute(lidar="lidar", topic="/front/debug", frame_id="front_debug"),
        "rear": ROS2PointCloudRoute(lidar="rear_lidar", topic="/rear/points", frame_id="rear_lidar"),
    }
    config = ROS2PointCloudPluginConfig(async_publish=True, routes=routes)
    plugin = cast("ROS2PointCloudPlugin", config.get_cls()(config, _FakeSimulator(include_rear_lidar=True)))
    plugin.start()
    front_packet = PointCloudPacket(
        lidar="lidar",
        env_id=0,
        points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        sim_time=4.5,
        height=1,
        width=1,
    )
    rear_packet = PointCloudPacket(
        lidar="rear_lidar",
        env_id=0,
        points=np.array([[-4.0, 5.0, -6.0]], dtype=np.float32),
        sim_time=4.5,
        height=1,
        width=1,
    )

    plugin.publish({front_packet.key: front_packet, rear_packet.key: rear_packet})

    assert len(workers) == 3
    assert all(worker.started for worker in workers)
    publishers = {publisher.topic: publisher for publisher in cast("_FakeNode", fake_ros["node"]).publishers}
    expected = {
        "/front/points": (np.array([[-3.0, -1.0, 2.0]], dtype=np.float32), "front_lidar"),
        "/front/debug": (front_packet.points, "front_debug"),
        "/rear/points": (rear_packet.points, "rear_lidar"),
    }
    assert set(publishers) == set(expected)
    for topic, (points, frame_id) in expected.items():
        published = publishers[topic].published
        assert len(published) == 1
        assert published[0].header.frame_id == frame_id
        assert np.array_equal(np.frombuffer(published[0].data, dtype="<f4").reshape(1, 3), points)
    plugin.stop()


def test_default_frame_dense_cloud_timestamp_carry_and_idempotent_stop(fake_ros: dict[str, Any]) -> None:
    plugin, publisher = _start_plugin(fake_ros, frame_id=None, qos="reliable")
    packet = PointCloudPacket(
        lidar="lidar",
        env_id=0,
        points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        sim_time=1.999_999_999_6,
        height=1,
        width=1,
    )
    plugin.publish({packet.key: packet})

    msg = publisher.published[0]
    assert msg.header.frame_id == "lidar"
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (2, 0)
    assert msg.is_dense
    assert publisher.qos.reliability == "reliable"
    node = cast("_FakeNode", fake_ros["node"])
    plugin.stop()
    plugin.stop()
    assert node.destroy_count == 1
    assert _NoopExecutor.instances[0].shutdown_count == 1
    assert _NoopThread.instances[0].join_count == 1


def test_frame_end_drives_lazy_start_publish_and_skips_non_due_scan(fake_ros: dict[str, Any]) -> None:
    route = ROS2PointCloudRoute(lidar="lidar", topic="/scan")
    config = ROS2PointCloudPluginConfig(async_publish=False, routes={"scan": route})
    sim = _FakeSimulator()
    points = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    def _render_sensor() -> None:
        due = sim.sensor_manager.collect_lidars_due()
        if due:
            sim.sensor_manager.get_lidar("lidar").set_buffer("points", points)

    sim.hooks.add(Phase.FRAME_END, _render_sensor, name="sensors.render")
    plugin = cast("ROS2PointCloudPlugin", config.get_cls()(config, sim))
    sim.hooks.emit(Phase.FRAME_END)
    publisher_node = cast("_FakeNode", fake_ros["node"])
    publisher = publisher_node.publishers[0]
    assert len(publisher.published) == 1
    assert np.array_equal(
        np.frombuffer(publisher.published[0].data, dtype="<f4").reshape(2, 3),
        points[0].numpy(),
    )

    sim._time = 1.27
    sim.hooks.emit(Phase.FRAME_END)
    assert len(publisher.published) == 1

    sim._time = 1.29
    sim.hooks.emit(Phase.FRAME_END)
    assert fake_ros["node"] is publisher_node
    assert fake_ros["node_count"] == 1
    assert len(publisher.published) == 2
    plugin.stop()


def test_default_sync_route_encodes_and_publishes_without_worker(
    fake_ros: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_worker(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("synchronous publication must not create a worker")

    monkeypatch.setattr(ros2_pointcloud_plugin, "PublishWorker", reject_worker)
    plugin, publisher = _start_plugin(fake_ros, async_publish=None)
    assert not plugin.config.async_publish
    packet = PointCloudPacket(
        lidar="lidar",
        env_id=0,
        points=np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
        sim_time=3.5,
        height=1,
        width=1,
    )
    plugin.publish({packet.key: packet})

    assert len(publisher.published) == 1
    assert np.array_equal(np.frombuffer(publisher.published[0].data, dtype="<f4"), packet.points.reshape(-1))
    plugin.stop()


def test_stop_retains_ros_resources_until_publish_workers_terminate(
    fake_ros: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    workers = []

    class _DelayedWorker:
        def __init__(self, publish_fn: Any, *, maxlen: int, name: str) -> None:
            self.allow_stop = False
            self.stop_calls = 0
            workers.append(self)

        def start(self) -> None:
            pass

        def submit(self, packet: PointCloudPacket) -> None:
            pass

        def stop(self) -> None:
            self.stop_calls += 1
            if not self.allow_stop:
                raise RuntimeError("worker did not stop")

    monkeypatch.setattr(ros2_pointcloud_plugin, "PublishWorker", _DelayedWorker)
    plugin, _ = _start_plugin(fake_ros, async_publish=True)
    node = cast("_FakeNode", fake_ros["node"])
    executor = _NoopExecutor.instances[0]
    spin_thread = _NoopThread.instances[0]
    worker = workers[0]

    with pytest.raises(RuntimeError, match="worker did not stop"):
        plugin.stop()
    assert worker.stop_calls == 1
    assert cast("dict[str, Any]", plugin._workers) == {"/scan": worker}
    assert plugin._node is node
    assert plugin._executor is executor
    assert executor.shutdown_count == 0
    assert node.destroy_count == 0
    assert spin_thread.join_count == 0

    worker.allow_stop = True
    plugin.stop()
    assert worker.stop_calls == 2
    assert plugin._workers == {}
    assert plugin._node is None
    assert plugin._executor is None
    assert executor.shutdown_count == 1
    assert node.destroy_count == 1
    assert spin_thread.join_count == 1
