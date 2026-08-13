"""ROS2 example plugins.

Reference plugin implementations that talk ROS2 (each constructed as
``cls(cfg, simulator)`` and registering hooks on ``simulator.hooks`` — no base class):

- :class:`ClockPublishPlugin` publishes sim time as ``rosgraph_msgs/msg/Clock``.
- :class:`GantryControlPlugin` drives the virtual gantry from three independent topics.
- :class:`ROS2OdometryPlugin` publishes the robot base pose/velocity as ``nav_msgs/Odometry`` — a
  self-sourced egress that reads ``robot_root_states`` each control step (no camera frames).

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

if TYPE_CHECKING:
    from holosoma.config_types.plugin import (
        ClockPublishPluginConfig,
        GantryControlPluginConfig,
        ROS2OdometryPluginConfig,
    )
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


# Guards rclpy.init() against concurrent calls from multiple plugins/threads.
_ros2_init_lock = threading.Lock()


def _ensure_ros2_init() -> None:
    """Call ``rclpy.init()`` once; a second call is a harmless no-op."""
    import rclpy

    with _ros2_init_lock:
        try:
            rclpy.init(args=None)
        except RuntimeError:
            pass  # Already initialized (rclpy raises if init() is called twice).


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
        import rclpy
        from rosgraph_msgs.msg import Clock

        self._clock_msg_cls = Clock

        _ensure_ros2_init()
        self._node = rclpy.create_node(cfg.node_name)
        self._pub = self._node.create_publisher(Clock, cfg.topic, 10)
        logger.info(f"ClockPublishPlugin publishing sim time on '{cfg.topic}' (node '{cfg.node_name}')")

        # `every` accepts an int or a frequency string ("100Hz"); the registry decimates natively.
        self.simulator.hooks.add(Phase.POST_STEP, self.publish, name="clock_publish.publish", every=cfg.publish_every)
        self.simulator.hooks.add(Phase.CLOSE, self.close, name="clock_publish.close")

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
        if node is not None:
            node.destroy_node()
            self._node = None


class GantryControlPlugin:
    """Control the virtual gantry over ROS2 via three independent topics.

    Each property is its own subscription, so publishing to one topic changes only that
    property:

    - ``position_topic`` (``geometry_msgs/msg/Point``) -> gantry anchor point.
    - ``length_topic`` (``std_msgs/msg/Float64``) -> elastic-band rest length.
    - ``enabled_topic`` (``std_msgs/msg/Bool``) -> enable / disable.

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

        self._lock = threading.Lock()
        # Pending commands; None means "no new value for this property".
        self._pending_position: tuple[float, float, float] | None = None
        self._pending_length: float | None = None
        self._pending_enabled: bool | None = None

        _ensure_ros2_init()
        self._node = rclpy.create_node(cfg.node_name)
        self._node.create_subscription(Point, cfg.position_topic, self._on_position, 10)
        self._node.create_subscription(Float64, cfg.length_topic, self._on_length, 10)
        self._node.create_subscription(Bool, cfg.enabled_topic, self._on_enabled, 10)
        logger.info(
            "GantryControlPlugin listening: position "
            f"'{cfg.position_topic}', length '{cfg.length_topic}', enabled '{cfg.enabled_topic}'"
        )

        self._spin_stop = threading.Event()
        self._spin_thread: threading.Thread | None = threading.Thread(
            target=self._spin, name="gantry_control_spin", daemon=True
        )
        self._spin_thread.start()

        self.simulator.hooks.add(Phase.FRAME_BEGIN, self.apply, name="gantry_control.apply")
        self.simulator.hooks.add(Phase.CLOSE, self.close, name="gantry_control.close")

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
        import rclpy

        # spin_once with a timeout so the loop can notice the stop event and exit.
        while not self._spin_stop.is_set():
            rclpy.spin_once(self._node, timeout_sec=0.1)

    # ----- Applied on the simulator thread -----

    def apply(self) -> None:
        """Apply any commands received since the last control step to the gantry."""
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

    def close(self) -> None:
        """Stop the spin thread and tear down the ROS2 node (idempotent)."""
        stop = getattr(self, "_spin_stop", None)
        if stop is not None:
            stop.set()
        thread = getattr(self, "_spin_thread", None)
        if thread is not None:
            thread.join(timeout=1.0)
            self._spin_thread = None
        node = getattr(self, "_node", None)
        if node is not None:
            node.destroy_node()
            self._node = None


def _sim_time_to_stamp(sim_time: float):
    """Build a builtin_interfaces/Time from sim seconds (deferred import; only after start())."""
    from builtin_interfaces.msg import Time

    sec = int(sim_time)
    nanosec = round((sim_time - sec) * 1e9)
    if nanosec >= 1_000_000_000:  # rounding carry
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


class ROS2OdometryPlugin:
    """Publish the robot base pose/velocity as ``nav_msgs/Odometry`` on a ROS2 topic.

    A self-sourced (non-camera) egress plugin: each control step it reads the base state off
    ``simulator.robot_root_states`` — the sim analog of the robot's onboard sport/odom estimate — and
    publishes one ``nav_msgs/Odometry``. Fires on ``FRAME_END`` (base tensors fresh after the frame's
    refresh), so ``publish_every`` resolves against the control rate. Rides the same in-process rclpy
    transport the image egress uses (no CycloneDDS entanglement with the Unitree SDK bridge).

    Base velocities in ``robot_root_states`` are WORLD-frame on every backend (the unified contract);
    ``nav_msgs/Odometry`` expresses its twist in the ``child_frame_id`` (body) frame, so they are
    rotated world->body via :func:`quat_rotate_inverse`. Timestamps come from sim_time, not wall-clock.
    """

    cfg: ROS2OdometryPluginConfig

    def __init__(self, cfg: ROS2OdometryPluginConfig, simulator: BaseSimulator) -> None:
        self.cfg = cfg
        self.simulator = simulator
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self._Odometry = Odometry

        _ensure_ros2_init()
        self._node = rclpy.create_node(cfg.node_name)
        reliability = ReliabilityPolicy.RELIABLE if cfg.qos == "reliable" else ReliabilityPolicy.BEST_EFFORT
        qos = QoSProfile(reliability=reliability, history=HistoryPolicy.KEEP_LAST, depth=1)
        self._pub = self._node.create_publisher(Odometry, cfg.topic, qos)

        # A daemon spin thread so QoS handshakes progress without a sim-side spin (publish-only node).
        self._spin_stop = threading.Event()
        self._spin_thread: threading.Thread | None = threading.Thread(
            target=self._spin, name=f"odometry_spin:{cfg.node_name}", daemon=True
        )
        self._spin_thread.start()

        logger.info(f"ROS2OdometryPlugin publishing base odometry on '{cfg.topic}' (node '{cfg.node_name}')")

        # `every` accepts an int decimation or a frequency string ("50Hz"); the registry decimates natively.
        self.simulator.hooks.add(Phase.FRAME_END, self.publish, name="odometry.publish", every=cfg.publish_every)
        self.simulator.hooks.add(Phase.CLOSE, self.close, name="odometry.close")

    def _spin(self) -> None:
        import rclpy

        while not self._spin_stop.is_set():
            rclpy.spin_once(self._node, timeout_sec=0.1)

    def publish(self) -> None:
        pos, quat_xyzw, lin_vel_body, ang_vel_body, sim_time = self._read_base_state()
        msg = self._Odometry()
        msg.header.stamp = _sim_time_to_stamp(sim_time)
        msg.header.frame_id = self.cfg.frame_id
        msg.child_frame_id = self.cfg.child_frame_id

        msg.pose.pose.position.x = pos[0]
        msg.pose.pose.position.y = pos[1]
        msg.pose.pose.position.z = pos[2]
        # robot_root_states quaternion is xyzw; ROS geometry_msgs/Quaternion is also xyzw — direct copy.
        msg.pose.pose.orientation.x = quat_xyzw[0]
        msg.pose.pose.orientation.y = quat_xyzw[1]
        msg.pose.pose.orientation.z = quat_xyzw[2]
        msg.pose.pose.orientation.w = quat_xyzw[3]

        msg.twist.twist.linear.x = lin_vel_body[0]
        msg.twist.twist.linear.y = lin_vel_body[1]
        msg.twist.twist.linear.z = lin_vel_body[2]
        msg.twist.twist.angular.x = ang_vel_body[0]
        msg.twist.twist.angular.y = ang_vel_body[1]
        msg.twist.twist.angular.z = ang_vel_body[2]

        self._pub.publish(msg)

    def _read_base_state(self) -> tuple[list[float], list[float], list[float], list[float], float]:
        """Read env ``cfg.env_id`` base state off the sim as plain floats.

        Returns ``(position, quat_xyzw, lin_vel_body, ang_vel_body, sim_time)``. The unified
        ``robot_root_states`` 13-vector is ``[pos(3), quat_xyzw(4), lin_vel_world(3), ang_vel_world(3)]``;
        the world-frame velocities are rotated into the base (body) frame for the Odometry twist.
        """
        from holosoma.utils.rotations import quat_rotate_inverse

        env = self.cfg.env_id
        root = self.simulator.robot_root_states[env]  # [13]
        quat_xyzw = root[3:7]  # xyzw
        lin_vel_world = root[7:10].unsqueeze(0)
        ang_vel_world = root[10:13].unsqueeze(0)
        lin_vel_body = quat_rotate_inverse(quat_xyzw.unsqueeze(0), lin_vel_world, w_last=True).squeeze(0)
        ang_vel_body = quat_rotate_inverse(quat_xyzw.unsqueeze(0), ang_vel_world, w_last=True).squeeze(0)

        pos = root[0:3].detach().cpu().tolist()
        quat = quat_xyzw.detach().cpu().tolist()
        lin = lin_vel_body.detach().cpu().tolist()
        ang = ang_vel_body.detach().cpu().tolist()
        return pos, quat, lin, ang, self.simulator.time()

    def close(self) -> None:
        """Stop the spin thread and tear down the ROS2 node (idempotent; safe from Phase.CLOSE)."""
        stop = getattr(self, "_spin_stop", None)
        if stop is not None:
            stop.set()
        thread = getattr(self, "_spin_thread", None)
        if thread is not None:
            thread.join(timeout=1.0)
            self._spin_thread = None
        node = getattr(self, "_node", None)
        if node is not None:
            node.destroy_node()
            self._node = None
