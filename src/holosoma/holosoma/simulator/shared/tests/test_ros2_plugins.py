"""Tests for the ROS2 example plugins.

rclpy is not installed in the no_sim environment, so these tests inject lightweight fake
``rclpy`` / ROS message modules into ``sys.modules`` to exercise plugin construction +
callback wiring. A separate test asserts the configs import and resolve with NO rclpy
present (the optional-dependency guarantee).
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch

from holosoma.config_types.plugin import (
    ClockPublishPluginConfig,
    GantryControlPluginConfig,
    PluginConfig,
    ROS2OdometryPluginConfig,
)
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase

pytestmark = pytest.mark.no_sim


def _build_plugin(sim: Any, cfg: PluginConfig) -> Any:
    """Construct a single plugin the way BaseSimulator.install_plugins does."""
    return cfg.get_cls()(cfg, cast("BaseSimulator", sim))


def test_configs_and_impl_import_without_rclpy() -> None:
    # The optional-dep guarantee: configs import and get_cls() resolves the impl module
    # without pulling rclpy (it is deferred into the plugin __init__/methods).
    assert "rclpy" not in sys.modules
    assert ClockPublishPluginConfig().get_cls().__name__ == "ClockPublishPlugin"
    assert GantryControlPluginConfig().get_cls().__name__ == "GantryControlPlugin"
    assert ROS2OdometryPluginConfig().get_cls().__name__ == "ROS2OdometryPlugin"
    assert "rclpy" not in sys.modules


# ----- Fake ROS2 stack ------------------------------------------------------------------


class _FakeNode:
    def __init__(self, name: str) -> None:
        self.name = name
        self.published: list[Any] = []
        self.published_by_topic: dict[str, list[Any]] = {}
        self.subscriptions: dict[str, Any] = {}
        self.publisher_types: dict[str, type] = {}
        self.subscription_types: dict[str, type] = {}
        self.destroyed = False

    def create_publisher(self, msg_cls: Any, topic: str, depth: Any) -> Any:
        node = self
        node.published_by_topic[topic] = []
        node.publisher_types[topic] = msg_cls

        class _Pub:
            def publish(self, msg: Any) -> None:
                assert isinstance(msg, msg_cls)
                node.published.append(msg)
                node.published_by_topic[topic].append(msg)

        return _Pub()

    def create_subscription(self, msg_cls: Any, topic: str, cb: Any, depth: int) -> None:
        self.subscriptions[topic] = cb
        self.subscription_types[topic] = msg_cls

    def destroy_node(self) -> None:
        self.destroyed = True


@pytest.fixture
def fake_ros2(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install fake rclpy + message modules; yield handles the tests can drive."""
    created_nodes: list[_FakeNode] = []

    rclpy = types.ModuleType("rclpy")

    def _init(*_args: Any, **_kwargs: Any) -> None:
        return None

    def _create_node(name: str, *, context: Any = None) -> _FakeNode:
        node = _FakeNode(name)
        created_nodes.append(node)
        return node

    def _spin_once(*_args: Any, **_kwargs: Any) -> None:
        return None

    rclpy.init = _init  # type: ignore[attr-defined]
    rclpy.create_node = _create_node  # type: ignore[attr-defined]
    rclpy.spin_once = _spin_once  # type: ignore[attr-defined]

    # Plugins now own a private SingleThreadedExecutor per spin thread.
    class _FakeExecutor:
        def __init__(self, *, context: Any = None) -> None:
            self.node: Any = None
            self.shutdown_called = False

        def add_node(self, _node: Any) -> None:
            self.node = _node

        def spin_once(self, timeout_sec: float | None = None) -> None:
            return None

        def shutdown(self, timeout_sec: float | None = None) -> bool:
            self.shutdown_called = True
            return True

        def remove_node(self, node: Any) -> None:
            assert node is self.node
            self.node = None

    rclpy_executors = types.ModuleType("rclpy.executors")
    rclpy_executors.SingleThreadedExecutor = _FakeExecutor  # type: ignore[attr-defined]

    class _FakeContext:
        def __init__(self) -> None:
            self._ok = True

        def ok(self) -> bool:
            return self._ok

        def shutdown(self) -> None:
            self._ok = False

    class _SignalHandlerOptions:
        NO = object()

    rclpy_context = types.ModuleType("rclpy.context")
    rclpy_context.Context = _FakeContext  # type: ignore[attr-defined]
    rclpy_signals = types.ModuleType("rclpy.signals")
    rclpy_signals.SignalHandlerOptions = _SignalHandlerOptions  # type: ignore[attr-defined]

    # rosgraph_msgs/msg/Clock has a nested builtin_interfaces/Time (sec + nanosec).
    class _Time:
        def __init__(self, sec: int = 0, nanosec: int = 0) -> None:
            self.sec = sec
            self.nanosec = nanosec

    class _Clock:
        def __init__(self) -> None:
            self.clock = _Time()

    rosgraph = types.ModuleType("rosgraph_msgs")
    rosgraph_msg = types.ModuleType("rosgraph_msgs.msg")
    rosgraph_msg.Clock = _Clock  # type: ignore[attr-defined]

    def _simple(name: str) -> type:
        # Bare message class; test callbacks set attributes (x/y/z, data) directly.
        return type(name, (), {})

    geo = types.ModuleType("geometry_msgs")
    geo_msg = types.ModuleType("geometry_msgs.msg")
    geo_msg.Point = _simple("Point")  # type: ignore[attr-defined]
    std = types.ModuleType("std_msgs")
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.Float64 = _simple("Float64")  # type: ignore[attr-defined]
    std_msg.Bool = _simple("Bool")  # type: ignore[attr-defined]

    class _QoSProfile:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _HistoryPolicy:
        KEEP_LAST = "keep_last"

    class _ReliabilityPolicy:
        RELIABLE = "reliable"
        BEST_EFFORT = "best_effort"

    rclpy_qos = types.ModuleType("rclpy.qos")
    rclpy_qos.QoSProfile = _QoSProfile  # type: ignore[attr-defined]
    rclpy_qos.HistoryPolicy = _HistoryPolicy  # type: ignore[attr-defined]
    rclpy_qos.ReliabilityPolicy = _ReliabilityPolicy  # type: ignore[attr-defined]

    class _Odometry:
        def __init__(self) -> None:
            self.header = types.SimpleNamespace(stamp=None, frame_id="")
            self.child_frame_id = ""
            self.pose = types.SimpleNamespace(
                pose=types.SimpleNamespace(
                    position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            )
            self.twist = types.SimpleNamespace(
                twist=types.SimpleNamespace(
                    linear=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    angular=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                )
            )

    nav = types.ModuleType("nav_msgs")
    nav_msg = types.ModuleType("nav_msgs.msg")
    nav_msg.Odometry = _Odometry  # type: ignore[attr-defined]
    builtin = types.ModuleType("builtin_interfaces")
    builtin_msg = types.ModuleType("builtin_interfaces.msg")
    builtin_msg.Time = _Time  # type: ignore[attr-defined]

    for name, mod in {
        "rclpy": rclpy,
        "rclpy.context": rclpy_context,
        "rclpy.executors": rclpy_executors,
        "rclpy.qos": rclpy_qos,
        "rclpy.signals": rclpy_signals,
        "rosgraph_msgs": rosgraph,
        "rosgraph_msgs.msg": rosgraph_msg,
        "geometry_msgs": geo,
        "geometry_msgs.msg": geo_msg,
        "std_msgs": std,
        "std_msgs.msg": std_msg,
        "nav_msgs": nav,
        "nav_msgs.msg": nav_msg,
        "builtin_interfaces": builtin,
        "builtin_interfaces.msg": builtin_msg,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return {
        "nodes": created_nodes,
        "Clock": _Clock,
        "Point": geo_msg.Point,
        "Float64": std_msg.Float64,
        "Bool": std_msg.Bool,
    }


# ----- Fakes for the simulator side -----------------------------------------------------


class _FakeGantry:
    def __init__(self) -> None:
        self.point: Any = (0.0, 0.0, 1.0)
        self.length: float = 0.2
        self.enabled: bool = False

    def set_enable(self, enable: bool) -> None:
        self.enabled = enable


class _FakeSim:
    """Minimal simulator stand-in whose HookRegistry is wired with per-phase base rates,
    mirroring BaseSimulator, so native ``every="20Hz"`` resolution works in tests."""

    def __init__(
        self,
        sim_time: float = 0.0,
        gantry: Any = None,
        fps: int = 200,
        control_decimation_steps: int = 4,
    ) -> None:
        control_hz = fps / control_decimation_steps
        base_rates = {
            Phase.PRE_STEP: float(fps),
            Phase.POST_STEP: float(fps),
            Phase.FRAME_BEGIN: control_hz,
            Phase.FRAME_END: control_hz,
        }
        self.hooks = HookRegistry(base_rates=base_rates)
        self.virtual_gantry = gantry
        self._t = sim_time
        self.body_names: list[str] = []
        self.rigid_body_pos_w: torch.Tensor
        self.rigid_body_quat_w: torch.Tensor
        self.rigid_body_lin_vel_w: torch.Tensor
        self.rigid_body_ang_vel_w: torch.Tensor
        self.robot_root_states: torch.Tensor

    def time(self) -> float:
        return self._t

    def find_rigid_body_indice(self, body_name: str) -> int:
        try:
            return self.body_names.index(body_name)
        except ValueError:
            return -1


@dataclass
class _Msg:
    pass


def _assert_bool_data(msg: Any, expected: bool) -> None:
    assert type(msg.data) is bool
    assert msg.data is expected


def test_clock_publish_plugin_publishes_sim_time_on_physics_phase(fake_ros2: dict[str, Any]) -> None:
    sim = _FakeSim(sim_time=2.5)
    plugin = _build_plugin(sim, ClockPublishPluginConfig(topic="/clock"))
    node = fake_ros2["nodes"][0]

    # It fires on POST_STEP (freshest sim time), not the control phase.
    sim.hooks.emit(Phase.FRAME_END)
    assert node.published == []
    sim.hooks.emit(Phase.POST_STEP)
    assert len(node.published) == 1
    msg = node.published[0]
    assert msg.clock.sec == 2
    assert msg.clock.nanosec == pytest.approx(0.5e9, abs=1)

    # CLOSE tears the node down.
    sim.hooks.emit(Phase.CLOSE)
    assert node.destroyed
    assert plugin is not None


def test_clock_publish_plugin_honors_publish_every(fake_ros2: dict[str, Any]) -> None:
    sim = _FakeSim(sim_time=1.0)
    _build_plugin(sim, ClockPublishPluginConfig(publish_every=3))
    node = fake_ros2["nodes"][0]
    for _ in range(6):
        sim.hooks.emit(Phase.POST_STEP)
    assert len(node.published) == 2  # steps 3 and 6


def test_clock_publish_plugin_resolves_frequency_string(fake_ros2: dict[str, Any]) -> None:
    # fps=200 physics rate; "20Hz" -> decimation 10 (publish every 10th physics step).
    sim = _FakeSim(sim_time=1.0, fps=200)
    _build_plugin(sim, ClockPublishPluginConfig(publish_every="20Hz"))
    node = fake_ros2["nodes"][0]
    for _ in range(20):
        sim.hooks.emit(Phase.POST_STEP)
    assert len(node.published) == 2  # steps 10 and 20


def test_gantry_control_applies_each_topic_independently(fake_ros2: dict[str, Any]) -> None:
    gantry = _FakeGantry()
    sim = _FakeSim(gantry=gantry)
    _build_plugin(sim, GantryControlPluginConfig())
    node = fake_ros2["nodes"][0]
    assert node.subscription_types == {
        "/gantry/position": fake_ros2["Point"],
        "/gantry/length": fake_ros2["Float64"],
        "/gantry/enabled": fake_ros2["Bool"],
    }
    assert node.publisher_types == {
        "/gantry/position/readback": fake_ros2["Point"],
        "/gantry/length/readback": fake_ros2["Float64"],
        "/gantry/enabled/readback": fake_ros2["Bool"],
    }

    # Commands apply on FRAME_BEGIN (before the gantry's own force step reads state).
    # A control-post emit must NOT apply anything.
    length_msg = _Msg()
    length_msg.data = 0.75  # type: ignore[attr-defined]
    node.subscriptions["/gantry/length"](length_msg)
    sim.hooks.emit(Phase.FRAME_END)
    assert gantry.length == 0.2  # not applied on the wrong phase
    assert node.published == []

    # Only command length: position and enabled stay untouched, and all default readbacks
    # report the resulting current state.
    sim.hooks.emit(Phase.FRAME_BEGIN)
    assert gantry.length == 0.75
    assert gantry.point == (0.0, 0.0, 1.0)  # not touched
    assert gantry.enabled is False  # not touched
    position = node.published_by_topic["/gantry/position/readback"][-1]
    length = node.published_by_topic["/gantry/length/readback"][-1]
    enabled = node.published_by_topic["/gantry/enabled/readback"][-1]
    assert (position.x, position.y, position.z) == (0.0, 0.0, 1.0)
    assert length.data == 0.75
    assert enabled.data is False

    # Now publish enabled only.
    enabled_msg = _Msg()
    enabled_msg.data = True  # type: ignore[attr-defined]
    node.subscriptions["/gantry/enabled"](enabled_msg)
    sim.hooks.emit(Phase.FRAME_BEGIN)
    assert gantry.enabled is True
    assert gantry.length == 0.75  # type: ignore[unreachable]  # unchanged (emit mutates state mypy can't track)
    assert node.published_by_topic["/gantry/enabled/readback"][-1].data is True

    # And position only.
    point_msg = _Msg()
    point_msg.x, point_msg.y, point_msg.z = 1.0, 2.0, 3.0
    node.subscriptions["/gantry/position"](point_msg)
    sim.hooks.emit(Phase.FRAME_BEGIN)
    assert tuple(gantry.point) == (1.0, 2.0, 3.0)
    position = node.published_by_topic["/gantry/position/readback"][-1]
    assert (position.x, position.y, position.z) == (1.0, 2.0, 3.0)

    sim.hooks.emit(Phase.CLOSE)
    assert node.destroyed


def test_gantry_control_without_commands_publishes_current_state(fake_ros2: dict[str, Any]) -> None:
    gantry = _FakeGantry()
    sim = _FakeSim(gantry=gantry)
    _build_plugin(sim, GantryControlPluginConfig())
    node = fake_ros2["nodes"][0]

    sim.hooks.emit(Phase.FRAME_BEGIN)  # no command; readback still publishes
    position_msgs = node.published_by_topic["/gantry/position/readback"]
    length_msgs = node.published_by_topic["/gantry/length/readback"]
    enabled_msgs = node.published_by_topic["/gantry/enabled/readback"]
    assert (position_msgs[-1].x, position_msgs[-1].y, position_msgs[-1].z) == (0.0, 0.0, 1.0)
    assert length_msgs[-1].data == 0.2
    _assert_bool_data(enabled_msgs[-1], False)

    # State can also change through simulator-side controls. The next frame must publish
    # fresh values rather than replaying the prior readback.
    gantry.point = (4.0, 5.0, 6.0)
    gantry.length = 1.5
    gantry.enabled = True
    sim.hooks.emit(Phase.FRAME_BEGIN)
    assert len(position_msgs) == len(length_msgs) == len(enabled_msgs) == 2
    assert (position_msgs[-1].x, position_msgs[-1].y, position_msgs[-1].z) == (4.0, 5.0, 6.0)
    assert length_msgs[-1].data == 1.5
    _assert_bool_data(enabled_msgs[-1], True)
    sim.hooks.emit(Phase.CLOSE)


_GANTRY_TOPIC_CASES = [
    ("position_topic", "subscription", "Point"),
    ("length_topic", "subscription", "Float64"),
    ("enabled_topic", "subscription", "Bool"),
    ("position_readback_topic", "publisher", "Point"),
    ("length_readback_topic", "publisher", "Float64"),
    ("enabled_readback_topic", "publisher", "Bool"),
]


@pytest.mark.parametrize(("field", "kind", "msg_name"), _GANTRY_TOPIC_CASES)
@pytest.mark.parametrize("topic", ["/custom/gantry/channel", None])
def test_gantry_control_configures_each_topic_independently(
    fake_ros2: dict[str, Any],
    field: str,
    kind: str,
    msg_name: str,
    topic: str | None,
) -> None:
    cfg = GantryControlPluginConfig(**cast("Any", {field: topic}))
    sim = _FakeSim(gantry=_FakeGantry())
    _build_plugin(sim, cfg)
    node = fake_ros2["nodes"][0]

    expected_subscription_types = {
        configured_topic: fake_ros2[configured_msg]
        for configured_topic, configured_msg in (
            (cfg.position_topic, "Point"),
            (cfg.length_topic, "Float64"),
            (cfg.enabled_topic, "Bool"),
        )
        if configured_topic is not None
    }
    expected_publisher_types = {
        configured_topic: fake_ros2[configured_msg]
        for configured_topic, configured_msg in (
            (cfg.position_readback_topic, "Point"),
            (cfg.length_readback_topic, "Float64"),
            (cfg.enabled_readback_topic, "Bool"),
        )
        if configured_topic is not None
    }
    assert node.subscription_types == expected_subscription_types
    assert node.publisher_types == expected_publisher_types

    endpoints = node.subscription_types if kind == "subscription" else node.publisher_types
    if topic is None:
        assert "/custom/gantry/channel" not in endpoints
    else:
        assert endpoints[topic] is fake_ros2[msg_name]
    sim.hooks.emit(Phase.CLOSE)


def test_gantry_control_custom_topics_and_disabled_channels(fake_ros2: dict[str, Any]) -> None:
    gantry = _FakeGantry()
    sim = _FakeSim(gantry=gantry)
    cfg = GantryControlPluginConfig(
        position_topic=None,
        length_topic="/custom/length/command",
        enabled_topic=None,
        position_readback_topic="/custom/position/state",
        length_readback_topic=None,
        enabled_readback_topic="/custom/enabled/state",
    )
    _build_plugin(sim, cfg)
    node = fake_ros2["nodes"][0]

    assert set(node.subscriptions) == {"/custom/length/command"}
    assert set(node.published_by_topic) == {"/custom/position/state", "/custom/enabled/state"}

    msg = _Msg()
    msg.data = 1.25  # type: ignore[attr-defined]
    node.subscriptions["/custom/length/command"](msg)
    sim.hooks.emit(Phase.FRAME_BEGIN)

    assert gantry.length == 1.25
    position = node.published_by_topic["/custom/position/state"]
    enabled = node.published_by_topic["/custom/enabled/state"]
    assert len(position) == len(enabled) == 1
    assert (position[0].x, position[0].y, position[0].z) == (0.0, 0.0, 1.0)
    assert enabled[0].data is False
    sim.hooks.emit(Phase.CLOSE)


def test_gantry_control_does_not_publish_without_virtual_gantry(fake_ros2: dict[str, Any]) -> None:
    sim = _FakeSim(gantry=None)
    _build_plugin(sim, GantryControlPluginConfig())
    node = fake_ros2["nodes"][0]

    sim.hooks.emit(Phase.FRAME_BEGIN)
    assert node.published == []

    msg = _Msg()
    msg.data = True  # type: ignore[attr-defined]
    node.subscriptions["/gantry/enabled"](msg)
    sim.hooks.emit(Phase.FRAME_BEGIN)
    assert node.published == []
    sim.hooks.emit(Phase.CLOSE)


def test_odometry_publishes_transformed_robot_body_frame(fake_ros2: dict[str, Any]) -> None:
    sim = _FakeSim(sim_time=3.25)
    sim.body_names = ["pelvis", "hand"]
    sim.rigid_body_pos_w = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]])
    # The hand is rotated +90 degrees around world Z.
    sqrt_half = 2**-0.5
    sim.rigid_body_quat_w = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, sqrt_half, sqrt_half]]])
    sim.rigid_body_lin_vel_w = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    sim.rigid_body_ang_vel_w = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])

    cfg = ROS2OdometryPluginConfig(
        body_name="hand",
        position=[1.0, 0.0, 0.0],
        # A +90-degree rotation around body X in wxyz order, with small input rounding drift.
        orientation=[1.000001 * sqrt_half, 1.000001 * sqrt_half, 0.0, 0.0],
        frame_id="odom",
        child_frame_id="tool",
    )
    _build_plugin(sim, cfg)
    node = fake_ros2["nodes"][0]

    sim.hooks.emit(Phase.FRAME_END)
    assert len(node.published) == 1
    msg = node.published[0]
    assert msg.header.frame_id == "odom"
    assert msg.child_frame_id == "tool"
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (3, 250_000_000)

    # Body offset [1,0,0] rotates to world [0,1,0], placing the frame at [1,3,3].
    assert (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z) == pytest.approx(
        (1.0, 3.0, 3.0)
    )
    assert (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w,
    ) == pytest.approx((0.5, 0.5, 0.5, 0.5))

    # v_frame_world = v_body_world + omega x offset = [1,0,0] + [-2,0,0].
    # Expressing [-1,0,0] and [0,0,2] in the composed child frame gives these twists.
    assert (msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z) == pytest.approx(
        (0.0, 0.0, -1.0), abs=1e-6
    )
    assert (msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z) == pytest.approx(
        (0.0, 2.0, 0.0), abs=1e-6
    )

    sim.hooks.emit(Phase.CLOSE)
    assert node.destroyed


def test_odometry_default_preserves_robot_root_behavior(fake_ros2: dict[str, Any]) -> None:
    sim = _FakeSim()
    sqrt_half = 2**-0.5
    sim.robot_root_states = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0, 0.0, sqrt_half, sqrt_half, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
    )
    _build_plugin(sim, ROS2OdometryPluginConfig())
    node = fake_ros2["nodes"][0]

    sim.hooks.emit(Phase.FRAME_END)
    assert len(node.published) == 1
    msg = node.published[0]
    assert (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z) == (1.0, 2.0, 3.0)
    assert (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w,
    ) == pytest.approx((0.0, 0.0, sqrt_half, sqrt_half))
    assert (msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z) == pytest.approx(
        (0.0, -1.0, 0.0), abs=1e-6
    )
    assert (msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-6
    )

    sim.hooks.emit(Phase.CLOSE)
    assert node.destroyed


def test_odometry_rejects_unknown_body(fake_ros2: dict[str, Any]) -> None:
    sim = _FakeSim()
    sim.body_names = ["pelvis", "hand"]
    _build_plugin(sim, ROS2OdometryPluginConfig(body_name="missing"))

    with pytest.raises(
        ValueError,
        match=r"body 'missing' was not found.*Available robot bodies: \['pelvis', 'hand'\]",
    ):
        sim.hooks.emit(Phase.FRAME_END)

    sim.hooks.emit(Phase.CLOSE)
