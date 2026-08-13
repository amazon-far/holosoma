from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import pytest

from holosoma.config_types.plugin import NoOpPluginConfig, PluginConfig
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase

pytestmark = pytest.mark.no_sim


class _FakeSimulator:
    """Minimal stand-in exposing the surface a plugin touches.

    Cast to ``BaseSimulator`` where a plugin expects one; the plugins below hold a
    ``_FakeSimulator``-typed reference so their fake-only attribute access type-checks.
    """

    def __init__(self) -> None:
        self.hooks = HookRegistry()
        self.forces: list[tuple[str, list[float]]] = []
        self.logged: list[str] = []

    def apply_forces(self, actor: str, force: list[float]) -> None:
        self.forces.append((actor, force))


def _build_plugins(sim: _FakeSimulator, configs: dict[str, PluginConfig]) -> dict[str, Any]:
    """Mirror BaseSimulator.__init__'s plugin construction (cls(cfg, sim)) for the fake sim."""
    return {key: cfg.get_cls()(cfg, cast("BaseSimulator", sim)) for key, cfg in configs.items()}


# A spec-shaped plugin: config declares knobs + get_cls(); the plugin (any class taking
# (cfg, simulator)) self-registers hooks in __init__.
@dataclass(frozen=True)
class _ApplyForcePluginConfig(PluginConfig):
    ros_name: str = "foo"
    force_value: tuple[float, ...] = (1.0, 2.0, 3.0)

    def get_cls(self) -> Callable[..., Any]:
        return _ApplyForcePlugin


class _ApplyForcePlugin:
    def __init__(self, cfg: _ApplyForcePluginConfig, simulator: BaseSimulator) -> None:
        self.cfg = cfg
        self.sim = cast("_FakeSimulator", simulator)
        self.sim.hooks.add(Phase.PRE_STEP, self.set_forces)
        self.sim.hooks.add(Phase.FRAME_END, self.log)

    def set_forces(self) -> None:
        self.sim.apply_forces("robot", list(self.cfg.force_value))

    def log(self) -> None:
        self.sim.logged.append(self.cfg.ros_name)


def test_build_plugins_constructs_and_wires_hooks() -> None:
    sim = _FakeSimulator()
    installed = _build_plugins(sim, {"apply_force": _ApplyForcePluginConfig(force_value=(10.0, 0.0, 0.0))})

    assert set(installed) == {"apply_force"}
    assert isinstance(installed["apply_force"], _ApplyForcePlugin)

    # Emitting the phases the plugin registered on runs its callbacks.
    sim.hooks.emit(Phase.PRE_STEP)
    sim.hooks.emit(Phase.FRAME_END)
    assert sim.forces == [("robot", [10.0, 0.0, 0.0])]
    assert sim.logged == ["foo"]


def test_build_plugins_empty_is_noop() -> None:
    sim = _FakeSimulator()
    assert _build_plugins(sim, {}) == {}
    # No hooks registered, so emitting a phase does nothing.
    sim.hooks.emit(Phase.PRE_STEP)
    assert sim.forces == []


def test_noop_plugin_registers_nothing() -> None:
    # `plugin.<key>:none` resolves to NoOpPluginConfig -> NoOpPlugin, which registers no hooks:
    # the slot is disabled but construction still returns an instance.
    from holosoma.simulator.shared.builtin_plugins import NoOpPlugin

    sim = _FakeSimulator()
    installed = _build_plugins(sim, {"off": NoOpPluginConfig(), "on": _ApplyForcePluginConfig()})
    assert set(installed) == {"off", "on"}
    assert type(installed["off"]) is NoOpPlugin
    sim.hooks.emit(Phase.FRAME_END)
    assert sim.logged == ["foo"]  # only the active plugin fired


def test_build_plugins_preserves_key_order() -> None:
    sim = _FakeSimulator()
    configs: dict[str, PluginConfig] = {
        "a": _ApplyForcePluginConfig(ros_name="a"),
        "b": _ApplyForcePluginConfig(ros_name="b"),
    }
    installed = _build_plugins(sim, configs)
    assert list(installed) == ["a", "b"]
    sim.hooks.emit(Phase.FRAME_END)
    # Registration order (a before b) is preserved through emit.
    assert sim.logged == ["a", "b"]


def test_plugin_stores_cfg_and_simulator() -> None:
    sim = _FakeSimulator()
    cfg = _ApplyForcePluginConfig()
    plugin = _ApplyForcePlugin(cfg, cast("BaseSimulator", sim))
    assert plugin.cfg is cfg
    assert plugin.sim is sim


def test_pluginconfig_get_cls_is_abstract() -> None:
    with pytest.raises(TypeError):
        PluginConfig()  # type: ignore[abstract]
