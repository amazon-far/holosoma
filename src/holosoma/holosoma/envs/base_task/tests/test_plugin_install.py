from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.config_types.plugin import NoOpPluginConfig, PluginConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase

pytestmark = pytest.mark.no_sim


class _FakeSimulator:
    """Minimal stand-in exposing the surface the plugins below touch."""

    def __init__(self) -> None:
        self.forces: list[tuple[str, list[float]]] = []
        self.logged: list[str] = []

    def apply_forces(self, actor: str, force: list[float]) -> None:
        self.forces.append((actor, force))


def _install(configs: dict[str, PluginConfig], simulator: Any = None, hooks: HookRegistry | None = None) -> BaseTask:
    """Run BaseTask's plugin installation on a task holding only what installation reads."""
    task = BaseTask.__new__(BaseTask)
    task.config = cast("Any", SimpleNamespace(plugin=configs))
    task.simulator = cast("BaseSimulator", simulator if simulator is not None else _FakeSimulator())
    task.hooks = hooks if hooks is not None else HookRegistry()
    task._install_plugins()
    return task


# A spec-shaped plugin: config declares knobs + get_cls(); the plugin (any class taking
# (cfg, task)) self-registers hooks on the task in __init__.
@dataclass(frozen=True)
class _ApplyForcePluginConfig(PluginConfig):
    ros_name: str = "foo"
    force_value: tuple[float, ...] = (1.0, 2.0, 3.0)

    def get_cls(self) -> Callable[..., Any]:
        return _ApplyForcePlugin


class _ApplyForcePlugin:
    def __init__(self, cfg: _ApplyForcePluginConfig, task: BaseTask) -> None:
        self.cfg = cfg
        self.task = task
        self.sim = cast("_FakeSimulator", task.simulator)
        task.hooks.add(Phase.PRE_STEP, self.set_forces)
        task.hooks.add(Phase.FRAME_END, self.log)

    def set_forces(self) -> None:
        self.sim.apply_forces("robot", list(self.cfg.force_value))

    def log(self) -> None:
        self.sim.logged.append(self.cfg.ros_name)


@dataclass(frozen=True)
class _RenderedSensorConsumerConfig(PluginConfig):
    def get_cls(self) -> Callable[..., Any]:
        return _RenderedSensorConsumer


class _RenderedSensorConsumer:
    def __init__(self, cfg: _RenderedSensorConsumerConfig, task: BaseTask) -> None:
        self.sim = cast("_InitializedFakeSim", task.simulator)
        task.hooks.add(Phase.FRAME_END, self.publish, name="consumer.publish")

    def publish(self) -> None:
        self.sim.events.append(f"publish:{self.sim.latest_sensor_value}")


class _InitializedFakeSim(BaseSimulator):
    """Minimal subclass that deliberately uses the real BaseSimulator initialization path."""

    def __init__(self, hooks: HookRegistry) -> None:
        self.events: list[str] = []
        self.latest_sensor_value = "stale"
        sim_cfg = SimpleNamespace(
            debug_viz=False,
            sim=SimpleNamespace(
                fps=100,
                control_decimation_steps=2,
                kinematic_playback=False,
            ),
        )
        config = SimpleNamespace(
            training=TrainingConfig(num_envs=1),
            simulator=sim_cfg,
            scene=SimpleNamespace(),
            sensors={},
            robot=SimpleNamespace(),
            logger=DisabledLoggerConfig(),
            experiment_dir=None,
        )
        super().__init__(cast("Any", config), cast("Any", None), "cpu", hooks)

    def render_sensors(self) -> None:
        self.latest_sensor_value = "fresh"
        self.events.append("render")


def test_simulator_renders_sensors_before_task_plugins_consume_them() -> None:
    hooks = HookRegistry()
    sim = _InitializedFakeSim(hooks)
    _install({"consumer": _RenderedSensorConsumerConfig()}, simulator=sim, hooks=hooks)
    hooks.emit(Phase.FRAME_END)
    assert sim.events == ["render", "publish:fresh"]


def test_install_constructs_plugins_with_the_task_in_key_order() -> None:
    task = _install(
        {
            "a": _ApplyForcePluginConfig(ros_name="a", force_value=(10.0, 0.0, 0.0)),
            "b": _ApplyForcePluginConfig(ros_name="b"),
        }
    )

    assert list(task.installed_plugins) == ["a", "b"]
    assert all(plugin.task is task for plugin in task.installed_plugins.values())

    task.hooks.emit(Phase.PRE_STEP)
    task.hooks.emit(Phase.FRAME_END)
    sim = cast("_FakeSimulator", task.simulator)
    assert sim.forces == [("robot", [10.0, 0.0, 0.0]), ("robot", [1.0, 2.0, 3.0])]
    assert sim.logged == ["a", "b"]


def test_noop_plugin_registers_nothing() -> None:
    # `plugin.<key>:none` resolves to NoOpPluginConfig -> NoOpPlugin, which registers no hooks:
    # the slot is disabled but construction still returns an instance.
    from holosoma.simulator.shared.builtin_plugins import NoOpPlugin

    task = _install({"off": NoOpPluginConfig(), "on": _ApplyForcePluginConfig()})
    assert set(task.installed_plugins) == {"off", "on"}
    assert type(task.installed_plugins["off"]) is NoOpPlugin
    task.hooks.emit(Phase.FRAME_END)
    assert cast("_FakeSimulator", task.simulator).logged == ["foo"]  # only the active plugin fired


def test_pluginconfig_get_cls_is_abstract() -> None:
    with pytest.raises(TypeError):
        PluginConfig()  # type: ignore[abstract]
