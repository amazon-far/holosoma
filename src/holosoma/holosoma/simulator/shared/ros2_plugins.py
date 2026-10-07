"""ROS2 example plugins.

Reference plugin implementations that talk ROS2 (each constructed as
``cls(cfg, simulator)`` and registering hooks on ``simulator.hooks`` — no base class):

- :class:`ClockPublishPlugin` publishes sim time as ``rosgraph_msgs/msg/Clock``.
- :class:`GantryControlPlugin` controls and reports virtual-gantry state over independent topics.
- :class:`ROS2OdometryPlugin` publishes a robot-attached frame as ``nav_msgs/Odometry`` — a
  self-sourced egress that reads robot rigid-body state each control step (no camera frames).

rclpy and the ROS message packages are an **optional** dependency (``holosoma[ros2]``).
They are imported lazily inside methods (never at module top), mirroring
``holosoma_inference/inputs/impl/ros2.py``, so this module — and the configs that point
at it — import cleanly on a bare install without ROS.

ROS callbacks run on a background spin thread; they only stash the latest value under a
lock. The values are read and applied on the simulator thread inside the lifecycle-phase
callbacks, so no simulator state is touched off-thread.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from loguru import logger

from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.simulator.shared.ros2_lifecycle import (
    close_ros2_executor,
    get_ros2_runtime,
    spin_executor_until_stopped,
)

if TYPE_CHECKING:
    from torch import Tensor

    from holosoma.config_types.plugin import (
        ClockPublishPluginConfig,
        GantryControlPluginConfig,
        ROS2OdometryPluginConfig,
    )
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class ClockPublishPlugin:
    """Publish the simulator clock as ``rosgraph_msgs/msg/Clock`` on a ROS2 topic.

    Fires on ``POST_STEP`` — right after ``simulate_at_each_physics_step``
    advances the clock — so it publishes the freshest sim time. ``publish_every`` is
    therefore a decimation of the PHYSICS rate (resolved against ``fps``), letting the
    clock tick faster than the control loop. ROS2 nodes running with ``use_sim_time``
    follow this clock.
    """

    cfg: ClockPublishPluginConfig

    def __init__(self, cfg: ClockPublishPluginConfig, simulator: BaseSimulator) -> None:
        self.cfg = cfg
        self.simulator = simulator
        self._node: Any = None
        self._pub: Any = None
        runtime = get_ros2_runtime(simulator)
        self.simulator.hooks.add(Phase.CLOSE, self.close, name="clock_publish.close")

        import rclpy
        from rosgraph_msgs.msg import Clock

        self._clock_msg_cls = Clock
        self._node = rclpy.create_node(cfg.node_name, context=runtime.start())
        self._pub = self._node.create_publisher(Clock, cfg.topic, 10)
        logger.info(f"ClockPublishPlugin publishing sim time on '{cfg.topic}' (node '{cfg.node_name}')")

        # `every` accepts an int or a frequency string ("100Hz"); the registry decimates natively.
        self.simulator.hooks.add(Phase.POST_STEP, self.publish, name="clock_publish.publish", every=cfg.publish_every)

    def publish(self) -> None:
        sim_time = float(self.simulator.time())
        msg = self._clock_msg_cls()
        # rosgraph_msgs/Clock carries a builtin_interfaces/Time (sec + nanosec).
        msg.clock.sec = int(sim_time)
        msg.clock.nanosec = int((sim_time - int(sim_time)) * 1e9)
        self._pub.publish(msg)

    def close(self) -> None:
        """Tear down the ROS2 node (idempotent; safe from Phase.CLOSE)."""
        node = getattr(self, "_node", None)
        self._node = None
        self._pub = None
        if node is not None:
            node.destroy_node()


class GantryControlPlugin:
    """Control and monitor the virtual gantry over independent ROS2 topics.

    Each property has an independent command subscription and state publisher:

    - ``position_topic`` (``geometry_msgs/msg/Point``) -> gantry anchor point.
    - ``length_topic`` (``std_msgs/msg/Float64``) -> elastic-band rest length.
    - ``enabled_topic`` (``std_msgs/msg/Bool``) -> enable / disable.
    - ``position_readback_topic`` (``geometry_msgs/msg/Point``) <- current anchor point.
    - ``length_readback_topic`` (``std_msgs/msg/Float64``) <- current rest length.
    - ``enabled_readback_topic`` (``std_msgs/msg/Bool``) <- current enabled state.

    A topic configured as ``None`` does not create its subscription or publisher.

    Subscription callbacks run on a background spin thread and only stash the latest
    command under a lock. The commands are drained and applied to the gantry on the
    simulator thread in the ``FRAME_BEGIN`` callback — before
    ``PRE_STEP``, where the gantry's own ``step()`` reads ``point`` /
    ``length`` / ``enabled`` to compute the band force — so a command takes effect on
    the same control cycle it arrives in rather than one cycle late.
    """

    cfg: GantryControlPluginConfig

    def __init__(self, cfg: GantryControlPluginConfig, simulator: BaseSimulator) -> None:
        self.cfg = cfg
        self.simulator = simulator
        import rclpy
        from geometry_msgs.msg import Point
        from std_msgs.msg import Bool, Float64

        self._Point = Point
        self._Float64 = Float64
        self._Bool = Bool
        self._lock = threading.Lock()
        # Pending commands; None means "no new value for this property".
        self._pending_position: tuple[float, float, float] | None = None
        self._pending_length: float | None = None
        self._pending_enabled: bool | None = None
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._spin_stop = threading.Event()
        runtime = get_ros2_runtime(simulator)
        self.simulator.hooks.add(Phase.CLOSE, self.close, name="gantry_control.close")

        context = runtime.start()
        self._node = rclpy.create_node(cfg.node_name, context=context)
        if cfg.position_topic is not None:
            self._node.create_subscription(Point, cfg.position_topic, self._on_position, 10)
        if cfg.length_topic is not None:
            self._node.create_subscription(Float64, cfg.length_topic, self._on_length, 10)
        if cfg.enabled_topic is not None:
            self._node.create_subscription(Bool, cfg.enabled_topic, self._on_enabled, 10)

        self._position_pub = (
            self._node.create_publisher(Point, cfg.position_readback_topic, 10)
            if cfg.position_readback_topic is not None
            else None
        )
        self._length_pub = (
            self._node.create_publisher(Float64, cfg.length_readback_topic, 10)
            if cfg.length_readback_topic is not None
            else None
        )
        self._enabled_pub = (
            self._node.create_publisher(Bool, cfg.enabled_readback_topic, 10)
            if cfg.enabled_readback_topic is not None
            else None
        )
        # Own a dedicated executor for this node. A bare rclpy.spin_once(node) shares a global
        # default executor across every ROS2 plugin's spin thread, so with >1 such plugin all but
        # the first racing thread raise "Executor is already spinning" and die — silently killing
        # this plugin's subscriptions. A private SingleThreadedExecutor with just this node avoids it.
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor(context=context)
        self._executor.add_node(self._node)
        logger.info(
            f"GantryControlPlugin command topics: {cfg.position_topic}, {cfg.length_topic}, {cfg.enabled_topic}"
        )
        logger.info(
            "GantryControlPlugin readback topics: "
            f"{cfg.position_readback_topic}, {cfg.length_readback_topic}, {cfg.enabled_readback_topic}"
        )

        self._spin_thread = threading.Thread(target=self._spin, name="gantry_control_spin", daemon=True)
        self._spin_thread.start()

        self.simulator.hooks.add(Phase.FRAME_BEGIN, self.apply, name="gantry_control.apply")

    # ----- ROS callbacks (spin thread): stash only, never touch the simulator -----

    def _on_position(self, msg: Any) -> None:
        with self._lock:
            self._pending_position = (float(msg.x), float(msg.y), float(msg.z))

    def _on_length(self, msg: Any) -> None:
        with self._lock:
            self._pending_length = float(msg.data)

    def _on_enabled(self, msg: Any) -> None:
        with self._lock:
            self._pending_enabled = bool(msg.data)

    def _spin(self) -> None:
        spin_executor_until_stopped(self._executor, self._spin_stop)

    # ----- Applied on the simulator thread -----

    def apply(self) -> None:
        """Apply pending commands and publish the resulting gantry state."""
        with self._lock:
            position, length, enabled = self._pending_position, self._pending_length, self._pending_enabled
            self._pending_position = self._pending_length = self._pending_enabled = None

        gantry = self.simulator.virtual_gantry
        if gantry is None:
            if position is not None or length is not None or enabled is not None:
                logger.warning("GantryControlPlugin received a command but no virtual gantry is present")
            return

        import numpy as np

        if position is not None:
            gantry.point = np.array(position)
            logger.info(f"Gantry position set to {position}")
        if length is not None:
            gantry.length = length
            logger.info(f"Gantry length set to {length}")
        if enabled is not None:
            gantry.set_enable(enabled)
            logger.info(f"Gantry {'enabled' if gantry.enabled else 'disabled'}")

        if self._position_pub is not None:
            msg = self._Point()
            msg.x, msg.y, msg.z = (float(value) for value in gantry.point)
            self._position_pub.publish(msg)
        if self._length_pub is not None:
            msg = self._Float64()
            msg.data = float(gantry.length)
            self._length_pub.publish(msg)
        if self._enabled_pub is not None:
            msg = self._Bool()
            msg.data = bool(gantry.enabled)
            self._enabled_pub.publish(msg)

    def close(self) -> None:
        """Stop the spin thread and tear down the ROS2 node (idempotent)."""
        node = getattr(self, "_node", None)
        if node is None:
            return
        executor = getattr(self, "_executor", None)
        spin_thread = getattr(self, "_spin_thread", None)
        self._executor = None
        self._spin_thread = None
        self._node = None
        self._position_pub = None
        self._length_pub = None
        self._enabled_pub = None
        close_ros2_executor(
            executor=executor,
            node=node,
            spin_thread=spin_thread,
            stop_event=self._spin_stop,
            label="GantryControlPlugin",
        )


def _sim_time_to_stamp(sim_time: float) -> Any:
    """Build a builtin_interfaces/Time from sim seconds (deferred import; only after start())."""
    from builtin_interfaces.msg import Time

    sec = int(sim_time)
    nanosec = round((sim_time - sec) * 1e9)
    if nanosec >= 1_000_000_000:  # rounding carry
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


def _compose_frame_pose(
    body_pos_world: Tensor,
    body_quat_world: Tensor,
    frame_position_body: list[float],
    frame_orientation_body_wxyz: list[float],
) -> tuple[Tensor, Tensor]:
    """Compose the body's world pose with the configured body-to-frame transform."""
    import torch

    from holosoma.utils.rotations import (
        transform_from_rotation_translation,
        transform_mul,
        transform_rotation,
        transform_translation,
    )

    position_body = torch.as_tensor(
        frame_position_body, dtype=body_pos_world.dtype, device=body_pos_world.device
    ).unsqueeze(0)
    orientation_wxyz = torch.as_tensor(
        frame_orientation_body_wxyz, dtype=body_quat_world.dtype, device=body_quat_world.device
    )
    orientation_xyzw = orientation_wxyz[[1, 2, 3, 0]]
    orientation_xyzw = (orientation_xyzw / torch.linalg.vector_norm(orientation_xyzw)).unsqueeze(0)

    body_transform_world = transform_from_rotation_translation(
        r=body_quat_world.unsqueeze(0), t=body_pos_world.unsqueeze(0)
    )
    frame_transform_body = transform_from_rotation_translation(r=orientation_xyzw, t=position_body)
    frame_transform_world = transform_mul(body_transform_world, frame_transform_body)
    return (
        transform_translation(frame_transform_world).squeeze(0),
        transform_rotation(frame_transform_world).squeeze(0),
    )


def _compute_frame_twist(
    body_pos_world: Tensor,
    body_lin_vel_world: Tensor,
    body_ang_vel_world: Tensor,
    frame_pos_world: Tensor,
    frame_quat_world: Tensor,
) -> tuple[Tensor, Tensor]:
    """Shift the body twist to the frame origin, then express it in the frame."""
    import torch

    from holosoma.utils.rotations import quat_rotate_inverse

    offset_world = frame_pos_world - body_pos_world
    frame_lin_vel_world = body_lin_vel_world + torch.cross(body_ang_vel_world, offset_world, dim=0)

    frame_quat_batch = frame_quat_world.unsqueeze(0)
    frame_lin_vel = quat_rotate_inverse(frame_quat_batch, frame_lin_vel_world.unsqueeze(0), w_last=True).squeeze(0)
    frame_ang_vel = quat_rotate_inverse(frame_quat_batch, body_ang_vel_world.unsqueeze(0), w_last=True).squeeze(0)
    return frame_lin_vel, frame_ang_vel


class ROS2OdometryPlugin:
    """Publish a robot-attached frame as ``nav_msgs/Odometry`` on a ROS2 topic.

    Each control step it reads either ``simulator.robot_root_states`` or a configured robot body's
    state, composes the configured body-to-frame transform, and publishes one
    ``nav_msgs/Odometry``. Fires on ``FRAME_END`` (body tensors fresh after the frame's refresh), so
    ``publish_every`` resolves against the control rate.

    Simulator body velocities are world-frame. The configured frame's linear velocity includes the
    rigid offset term ``angular_velocity x offset``, then both twist components are rotated into
    ``child_frame_id`` as required by ``nav_msgs/Odometry``. Timestamps use sim time.
    """

    cfg: ROS2OdometryPluginConfig

    def __init__(self, cfg: ROS2OdometryPluginConfig, simulator: BaseSimulator) -> None:
        self.cfg = cfg
        self.simulator = simulator
        # Resolve the configured body lazily on first publish.
        self._body_index: int | None = None
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._spin_stop = threading.Event()
        runtime = get_ros2_runtime(simulator)
        self.simulator.hooks.add(Phase.CLOSE, self.close, name="odometry.close")

        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self._Odometry = Odometry

        context = runtime.start()
        self._node = rclpy.create_node(cfg.node_name, context=context)
        reliability = ReliabilityPolicy.RELIABLE if cfg.qos == "reliable" else ReliabilityPolicy.BEST_EFFORT
        qos = QoSProfile(reliability=reliability, history=HistoryPolicy.KEEP_LAST, depth=1)
        self._pub = self._node.create_publisher(Odometry, cfg.topic, qos)

        # Private executor (see GantryControlPlugin): bare rclpy.spin_once(node) shares a global
        # default executor across ROS2 plugin spin threads and races to "Executor is already
        # spinning". A dedicated SingleThreadedExecutor with just this node avoids it.
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor(context=context)
        self._executor.add_node(self._node)

        # A daemon spin thread so QoS handshakes progress without a sim-side spin (publish-only node).
        self._spin_thread = threading.Thread(target=self._spin, name=f"odometry_spin:{cfg.node_name}", daemon=True)
        self._spin_thread.start()

        source = cfg.body_name if cfg.body_name is not None else "robot root"
        logger.info(
            f"ROS2OdometryPlugin publishing frame '{cfg.child_frame_id}' attached to {source!r} "
            f"on '{cfg.topic}' (node '{cfg.node_name}')"
        )

        # `every` accepts an int decimation or a frequency string ("50Hz"); the registry decimates natively.
        self.simulator.hooks.add(Phase.FRAME_END, self.publish, name="odometry.publish", every=cfg.publish_every)

    def _spin(self) -> None:
        spin_executor_until_stopped(self._executor, self._spin_stop)

    def publish(self) -> None:
        pos, quat_xyzw, lin_vel_frame, ang_vel_frame, sim_time = self._read_frame_state()
        msg = self._Odometry()
        msg.header.stamp = _sim_time_to_stamp(sim_time)
        msg.header.frame_id = self.cfg.frame_id
        msg.child_frame_id = self.cfg.child_frame_id

        msg.pose.pose.position.x = pos[0]
        msg.pose.pose.position.y = pos[1]
        msg.pose.pose.position.z = pos[2]
        # Simulator quaternions and ROS geometry_msgs/Quaternion are both xyzw.
        msg.pose.pose.orientation.x = quat_xyzw[0]
        msg.pose.pose.orientation.y = quat_xyzw[1]
        msg.pose.pose.orientation.z = quat_xyzw[2]
        msg.pose.pose.orientation.w = quat_xyzw[3]

        msg.twist.twist.linear.x = lin_vel_frame[0]
        msg.twist.twist.linear.y = lin_vel_frame[1]
        msg.twist.twist.linear.z = lin_vel_frame[2]
        msg.twist.twist.angular.x = ang_vel_frame[0]
        msg.twist.twist.angular.y = ang_vel_frame[1]
        msg.twist.twist.angular.z = ang_vel_frame[2]

        self._pub.publish(msg)

    def _resolve_body_index(self, body_name: str) -> int:
        """Resolve and cache the configured robot-body index."""
        if self._body_index is None:
            body_index = int(self.simulator.find_rigid_body_indice(body_name))
            body_names = self.simulator.body_names
            if body_index < 0 or body_index >= len(body_names):
                raise ValueError(
                    f"ROS2OdometryPlugin body '{body_name}' was not found. Available robot bodies: {body_names}."
                )
            self._body_index = body_index
        return self._body_index

    def _read_body_state(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Read the configured body's world pose and twist for ``cfg.env_id``."""
        env = self.cfg.env_id
        body_name = self.cfg.body_name
        if body_name is None:
            root = self.simulator.robot_root_states[env]
            return root[0:3], root[3:7], root[7:10], root[10:13]

        body_index = self._resolve_body_index(body_name)
        return (
            self.simulator.rigid_body_pos_w[env, body_index],
            self.simulator.rigid_body_quat_w[env, body_index],
            self.simulator.rigid_body_lin_vel_w[env, body_index],
            self.simulator.rigid_body_ang_vel_w[env, body_index],
        )

    def _read_frame_state(self) -> tuple[list[float], list[float], list[float], list[float], float]:
        """Read the configured frame's world pose and frame-expressed twist as plain floats."""
        body_pos_world, body_quat_world, body_lin_vel_world, body_ang_vel_world = self._read_body_state()
        frame_pos_world, frame_quat_world = _compose_frame_pose(
            body_pos_world,
            body_quat_world,
            self.cfg.position,
            self.cfg.orientation,
        )
        frame_lin_vel, frame_ang_vel = _compute_frame_twist(
            body_pos_world,
            body_lin_vel_world,
            body_ang_vel_world,
            frame_pos_world,
            frame_quat_world,
        )

        pos = frame_pos_world.detach().cpu().tolist()
        quat = frame_quat_world.detach().cpu().tolist()
        lin = frame_lin_vel.detach().cpu().tolist()
        ang = frame_ang_vel.detach().cpu().tolist()
        return pos, quat, lin, ang, self.simulator.time()

    def close(self) -> None:
        """Stop the spin thread and tear down the ROS2 node (idempotent; safe from Phase.CLOSE)."""
        node = getattr(self, "_node", None)
        if node is None:
            return
        executor = getattr(self, "_executor", None)
        spin_thread = getattr(self, "_spin_thread", None)
        self._executor = None
        self._spin_thread = None
        self._node = None
        self._pub = None
        close_ros2_executor(
            executor=executor,
            node=node,
            spin_thread=spin_thread,
            stop_event=self._spin_stop,
            label="ROS2OdometryPlugin",
        )
