from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, cast

import pytest

from holosoma.config_types.plugin import NoOpPluginConfig, PluginConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.hooks import HookRef, HookRegistry, Phase
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

pytestmark = pytest.mark.no_sim


class _FakeSimulator:
    """Minimal stand-in exposing the surface the plugins below touch."""

    def __init__(self) -> None:
        self.forces: list[tuple[str, list[float]]] = []
        self.logged: list[str] = []

    def apply_forces(self, actor: str, force: list[float]) -> None:
        self.forces.append((actor, force))


def _install(configs: dict[str, PluginConfig]) -> BaseTask:
    """Run BaseTask's plugin installation on a task holding only what installation reads."""
    task = BaseTask.__new__(BaseTask)
    task.config = cast("Any", SimpleNamespace(plugin=configs))
    task.simulator = cast("BaseSimulator", _FakeSimulator())
    task.hooks = HookRegistry()
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
        task.hooks.add(Phase.PRE_STEP, self.set_forces, name=f"{cfg.ros_name}.set_forces")
        task.hooks.add(Phase.FRAME_END, self.log, name=f"{cfg.ros_name}.log")

    def set_forces(self) -> None:
        self.sim.apply_forces("robot", list(self.cfg.force_value))

    def log(self) -> None:
        self.sim.logged.append(self.cfg.ros_name)


class _ProbeHook(Enum):
    PUBLISH = "publish"


@dataclass(frozen=True)
class _PublisherConfig(PluginConfig):
    def get_cls(self) -> Callable[..., Any]:
        return _Publisher


class _Publisher:
    def __init__(self, cfg: _PublisherConfig, task: BaseTask) -> None:
        sim = cast("_FakeSimulator", task.simulator)
        task.hooks.add(Phase.FRAME_END, lambda: sim.logged.append("publish"), name="pub", key=_ProbeHook.PUBLISH)


@dataclass(frozen=True)
class _FollowerConfig(PluginConfig):
    follows: str = "pub"

    def get_cls(self) -> Callable[..., Any]:
        return _Follower


class _Follower:
    def __init__(self, cfg: _FollowerConfig, task: BaseTask) -> None:
        sim = cast("_FakeSimulator", task.simulator)
        task.hooks.add(
            Phase.FRAME_END,
            lambda: sim.logged.append("follow"),
            name="follow",
            after=HookRef(cfg.follows, _ProbeHook.PUBLISH),
        )


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


def test_install_scopes_each_plugin_to_its_slot_so_hookrefs_resolve() -> None:
    # The follower is configured first, so only slot-scoped publication orders it after the publisher.
    task = _install({"follower": _FollowerConfig(), "pub": _PublisherConfig()})
    task.hooks.validate()
    task.hooks.emit(Phase.FRAME_END)
    assert cast("_FakeSimulator", task.simulator).logged == ["publish", "follow"]


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
