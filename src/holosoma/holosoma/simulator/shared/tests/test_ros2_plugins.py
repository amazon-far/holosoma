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

from holosoma.config_types.plugin import ClockPublishPluginConfig, GantryControlPluginConfig, PluginConfig
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase

pytestmark = pytest.mark.no_sim


def _build_plugin(sim: Any, cfg: PluginConfig) -> Any:
    """Construct a single plugin the way BaseSimulator.__init__ does: cls(cfg, sim)."""
    return cfg.get_cls()(cfg, cast("BaseSimulator", sim))


def test_configs_and_impl_import_without_rclpy() -> None:
    # The optional-dep guarantee: configs import and get_cls() resolves the impl module
    # without pulling rclpy (it is deferred into the plugin __init__/methods).
    assert "rclpy" not in sys.modules
    assert ClockPublishPluginConfig().get_cls().__name__ == "ClockPublishPlugin"
    assert GantryControlPluginConfig().get_cls().__name__ == "GantryControlPlugin"
    assert "rclpy" not in sys.modules


# ----- Fake ROS2 stack ------------------------------------------------------------------


class _FakeNode:
    def __init__(self, name: str) -> None:
        self.name = name
        self.published: list[Any] = []
        self.subscriptions: dict[str, Any] = {}
        self.destroyed = False

    def create_publisher(self, msg_cls: Any, topic: str, depth: int) -> Any:
        node = self

        class _Pub:
            def publish(self, msg: Any) -> None:
                node.published.append(msg)

        return _Pub()

    def create_subscription(self, msg_cls: Any, topic: str, cb: Any, depth: int) -> None:
        self.subscriptions[topic] = cb

    def destroy_node(self) -> None:
        self.destroyed = True


@pytest.fixture
def fake_ros2(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install fake rclpy + message modules; yield handles the tests can drive."""
    created_nodes: list[_FakeNode] = []

    rclpy = types.ModuleType("rclpy")

    def _init(*_args: Any, **_kwargs: Any) -> None:
        return None

    def _create_node(name: str) -> _FakeNode:
        node = _FakeNode(name)
        created_nodes.append(node)
        return node

    def _spin_once(*_args: Any, **_kwargs: Any) -> None:
        return None

    rclpy.init = _init  # type: ignore[attr-defined]
    rclpy.create_node = _create_node  # type: ignore[attr-defined]
    rclpy.spin_once = _spin_once  # type: ignore[attr-defined]

    # rosgraph_msgs/msg/Clock has a nested builtin_interfaces/Time (sec + nanosec).
    class _Time:
        def __init__(self) -> None:
            self.sec = 0
            self.nanosec = 0

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

    for name, mod in {
        "rclpy": rclpy,
        "rosgraph_msgs": rosgraph,
        "rosgraph_msgs.msg": rosgraph_msg,
        "geometry_msgs": geo,
        "geometry_msgs.msg": geo_msg,
        "std_msgs": std,
        "std_msgs.msg": std_msg,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return {"nodes": created_nodes, "Clock": _Clock}


# ----- Fakes for the simulator side -----------------------------------------------------


class _FakeGantry:
    def __init__(self) -> None:
        self.point: Any = None
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

    def time(self) -> float:
        return self._t


@dataclass
class _Msg:
    pass


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

    # Commands apply on FRAME_BEGIN (before the gantry's own force step reads state).
    # A control-post emit must NOT apply anything.
    length_msg = _Msg()
    length_msg.data = 0.75  # type: ignore[attr-defined]
    node.subscriptions["/gantry/length"](length_msg)
    sim.hooks.emit(Phase.FRAME_END)
    assert gantry.length == 0.2  # not applied on the wrong phase

    # Only publish length: position and enabled must be untouched.
    sim.hooks.emit(Phase.FRAME_BEGIN)
    assert gantry.length == 0.75
    assert gantry.point is None  # not touched
    assert gantry.enabled is False  # not touched

    # Now publish enabled only.
    enabled_msg = _Msg()
    enabled_msg.data = True  # type: ignore[attr-defined]
    node.subscriptions["/gantry/enabled"](enabled_msg)
    sim.hooks.emit(Phase.FRAME_BEGIN)
    assert gantry.enabled is True
    assert gantry.length == 0.75  # unchanged

    # And position only.
    point_msg = _Msg()
    point_msg.x, point_msg.y, point_msg.z = 1.0, 2.0, 3.0  # type: ignore[attr-defined]
    node.subscriptions["/gantry/position"](point_msg)
    sim.hooks.emit(Phase.FRAME_BEGIN)
    assert tuple(gantry.point) == (1.0, 2.0, 3.0)

    sim.hooks.emit(Phase.CLOSE)
    assert node.destroyed


def test_gantry_control_no_command_is_noop(fake_ros2: dict[str, Any]) -> None:
    gantry = _FakeGantry()
    sim = _FakeSim(gantry=gantry)
    _build_plugin(sim, GantryControlPluginConfig())
    sim.hooks.emit(Phase.FRAME_BEGIN)  # nothing published
    assert gantry.point is None
    assert gantry.length == 0.2
    assert gantry.enabled is False
    sim.hooks.emit(Phase.CLOSE)
