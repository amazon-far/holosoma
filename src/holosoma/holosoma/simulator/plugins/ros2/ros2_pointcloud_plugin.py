"""ROS2 PointCloud2 publisher for mounted LiDAR scans."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from holosoma.simulator.plugins.lidar_consumer import LidarConsumerPlugin
from holosoma.simulator.plugins.ros2.pointcloud2 import EncodedPointCloud, encode_xyz_points, transform_xyz_points
from holosoma.simulator.plugins.ros2.worker import PublishWorker
from holosoma.simulator.shared.ros2_lifecycle import (
    close_ros2_executor,
    get_ros2_runtime,
    spin_executor_until_stopped,
)

if TYPE_CHECKING:
    from holosoma.config_types.plugin import ROS2PointCloudPluginConfig, ROS2PointCloudRoute
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.plugins.lidar_consumer import LidarStreamKey, PointCloudPacket


def _sim_time_to_stamp(sim_time: float) -> Any:
    from builtin_interfaces.msg import Time

    sec = int(sim_time)
    nanosec = round((sim_time - sec) * 1e9)
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


class ROS2PointCloudPlugin(LidarConsumerPlugin):
    """One ROS2 node publishing configured LiDAR routes as PointCloud2."""

    config: ROS2PointCloudPluginConfig

    def __init__(self, config: ROS2PointCloudPluginConfig, simulator: BaseSimulator) -> None:
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: Any = None
        self._spin_stop = threading.Event()
        self._publishers: dict[str, Any] = {}
        self._workers: dict[str, PublishWorker[PointCloudPacket]] = {}
        self._ros2_runtime = get_ros2_runtime(simulator)
        super().__init__(config, simulator)

    def wanted_streams(self) -> set[LidarStreamKey]:
        return {(route.lidar, self.config.env_id) for route in self.config.routes.values()}

    def start(self) -> None:
        """Create ROS resources, rolling back partial startup on failure."""
        self._spin_stop = threading.Event()
        try:
            self._start()
        except BaseException:
            try:
                self.stop()
            except Exception:
                logger.exception("ROS2 point-cloud cleanup failed while preserving the startup error")
            raise

    def _start(self) -> None:
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2, PointField

        context = self._ros2_runtime.start()
        self._node = Node(self.config.node_name, context=context)
        self._PointCloud2 = PointCloud2
        self._PointField = PointField

        reliability = ReliabilityPolicy.RELIABLE if self.config.qos == "reliable" else ReliabilityPolicy.BEST_EFFORT
        qos = QoSProfile(reliability=reliability, history=HistoryPolicy.KEEP_LAST, depth=1)
        for route in self.config.routes.values():
            self._publishers[route.topic] = self._node.create_publisher(PointCloud2, route.topic, qos)
            if self.config.async_publish:
                worker = PublishWorker(
                    self._make_route_sender(route),
                    maxlen=self.config.queue_maxlen,
                    name=f"egress:{self.config.node_name}:{route.topic}",
                )
                self._workers[route.topic] = worker
                worker.start()

        self._executor = SingleThreadedExecutor(context=context)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._spin,
            name=f"egress-spin:{self.config.node_name}",
            daemon=True,
        )
        self._spin_thread.start()
        logger.info(f"ROS2 point-cloud egress '{self.config.node_name}' up: {len(self.config.routes)} route(s)")

    def _spin(self) -> None:
        spin_executor_until_stopped(self._executor, self._spin_stop)

    def publish(self, clouds: dict[LidarStreamKey, PointCloudPacket]) -> None:
        for route in self.config.routes.values():
            packet = clouds.get((route.lidar, self.config.env_id))
            if packet is None:
                continue
            if self.config.async_publish:
                self._workers[route.topic].submit(packet)
            else:
                self._publish_encoded(route, packet, self._encode(route, packet))

    @staticmethod
    def _encode(route: ROS2PointCloudRoute, packet: PointCloudPacket) -> EncodedPointCloud:
        points = transform_xyz_points(
            packet.points,
            translation=route.point_translation,
            rotation=route.point_rotation,
        )
        return encode_xyz_points(points, height=packet.height, width=packet.width)

    def _make_route_sender(self, route: ROS2PointCloudRoute) -> Callable[[PointCloudPacket], None]:
        def _send(packet: PointCloudPacket) -> None:
            self._publish_encoded(route, packet, self._encode(route, packet))

        return _send

    def _publish_encoded(
        self,
        route: ROS2PointCloudRoute,
        packet: PointCloudPacket,
        encoded: EncodedPointCloud,
    ) -> None:
        publisher = self._publishers.get(route.topic)
        if publisher is None:
            return
        msg = self._PointCloud2()
        msg.header.stamp = _sim_time_to_stamp(packet.sim_time)
        msg.header.frame_id = route.frame_id or packet.lidar
        msg.height = encoded.height
        msg.width = encoded.width
        msg.fields = [
            self._PointField(name="x", offset=0, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name="y", offset=4, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name="z", offset=8, datatype=self._PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = encoded.point_step
        msg.row_step = encoded.row_step
        msg.data = encoded.data
        msg.is_dense = encoded.is_dense
        publisher.publish(msg)

    def stop(self) -> None:
        failures: list[str] = []
        for worker in self._workers.values():
            try:
                worker.stop()
            except Exception as exc:  # noqa: PERF203 - stop every worker before reporting failures.
                failures.append(str(exc))
        if failures:
            raise RuntimeError("; ".join(failures))

        self._workers.clear()
        try:
            close_ros2_executor(
                executor=self._executor,
                node=self._node,
                spin_thread=self._spin_thread,
                stop_event=self._spin_stop,
                label=f"ROS2 point-cloud egress {self.config.node_name!r}",
            )
        except Exception as exc:
            failures.append(str(exc))
        self._executor = None
        self._spin_thread = None
        self._node = None
        self._publishers.clear()
        if failures:
            raise RuntimeError("; ".join(failures))
