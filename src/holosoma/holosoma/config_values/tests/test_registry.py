"""Tests for config preset registries."""

from __future__ import annotations

import dataclasses
import textwrap
from typing import Optional

import pytest
from loguru import logger
from typing_extensions import Annotated, TypeAlias

from holosoma.config_types.logger import DisabledLoggerConfig, LoggerConfig, WandbLoggerConfig
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_values.logger import LOGGER_REGISTRY
from holosoma.config_values.robot import g1_29dof
from holosoma.config_values.run_sim import RUN_SIM_REGISTRY
from holosoma.utils import config_registry as registry
from holosoma.utils.config_registry import ConfigRegistry, UseRegistry

# Ensure all real config_values registries are discovered before any test runs, so
# registry_for_value_type / parse_config see the full menu set regardless of test order.
registry.load_plugins()


class _FakeEP:
    """Stand-in for an importlib.metadata EntryPoint."""

    def __init__(self, name, value, loader):
        self.name = name
        self.value = value
        self._loader = loader

    def load(self):
        return self._loader()


@pytest.fixture(autouse=True)
def _isolate_registry_state():
    """Snapshot/restore module globals so test-created registries don't leak."""
    all_before = list(registry.ALL_REGISTRIES)
    loaded_before = set(registry._loaded_groups)
    files_before = set(registry._loaded_files)
    yield
    registry.ALL_REGISTRIES[:] = all_before
    registry._loaded_groups.clear()
    registry._loaded_groups.update(loaded_before)
    registry._loaded_files.clear()
    registry._loaded_files.update(files_before)


@pytest.fixture
def warnings():
    """Capture loguru WARNING messages (loguru does not feed pytest's caplog)."""
    msgs: list[str] = []
    sink_id = logger.add(lambda m: msgs.append(m.record["message"]), level="WARNING")
    yield msgs
    logger.remove(sink_id)


def _robot_registry() -> ConfigRegistry:
    return ConfigRegistry(RobotConfig, group="holosoma.config.robot")


def _patch_entry_points(monkeypatch, eps):
    monkeypatch.setattr(registry, "entry_points", lambda **_kwargs: eps)


# --- ConfigRegistry.add ------------------------------------------------------------------------


def test_add_returns_value_and_registers():
    reg = _robot_registry()
    value = dataclasses.replace(g1_29dof)
    assert reg.add("r", value) is value
    assert reg["r"] is value


def test_add_allows_none_sentinel():
    reg = ConfigRegistry(RobotConfig)
    assert reg.add("none", None) is None
    assert reg["none"] is None


def test_add_rejects_wrong_type():
    reg = _robot_registry()
    with pytest.raises(TypeError, match="expected RobotConfig"):
        reg.add("bad", object())


def test_registry_self_registers():
    before = len(registry.ALL_REGISTRIES)
    reg = _robot_registry()
    assert registry.ALL_REGISTRIES[-1] is reg
    assert len(registry.ALL_REGISTRIES) == before + 1


# --- entry-point loading -----------------------------------------------------------------------


def test_bad_entry_point_is_skipped_not_fatal(monkeypatch):
    def boom():
        raise RuntimeError("import blew up")

    good = dataclasses.replace(g1_29dof)
    _patch_entry_points(
        monkeypatch,
        [_FakeEP("bad", "pkg.mod:bad", boom), _FakeEP("good", "pkg.mod:good", lambda: good)],
    )
    reg = _robot_registry()
    registry.load_entrypoint_presets(reg)  # must not raise even though one ep errors
    assert "good" in reg
    assert "bad" not in reg


def test_failed_ep_is_retried_next_call(monkeypatch):
    state = {"fail": True}

    def maybe():
        if state["fail"]:
            raise RuntimeError("transient")
        return dataclasses.replace(g1_29dof)

    _patch_entry_points(monkeypatch, [_FakeEP("x", "p:x", maybe)])
    reg = _robot_registry()
    registry.load_entrypoint_presets(reg)
    assert "x" not in reg
    state["fail"] = False
    registry.load_entrypoint_presets(reg)
    assert "x" in reg  # retried, not skipped as "already loaded"


def test_entry_point_override_warns_and_wins(monkeypatch, warnings):
    override = dataclasses.replace(g1_29dof)
    _patch_entry_points(monkeypatch, [_FakeEP("g1_29dof", "ext.pkg:robot", lambda: override)])
    reg = _robot_registry()
    reg.add("g1_29dof", g1_29dof)
    registry.load_entrypoint_presets(reg)
    assert reg["g1_29dof"] is override  # last-writer-wins
    assert any("overrides existing" in m for m in warnings)


def test_entry_point_type_guard_skips_foreign(monkeypatch):
    # A value of the wrong type (shared group name from another package) is skipped, not raised.
    _patch_entry_points(monkeypatch, [_FakeEP("foreign", "other.pkg:thing", lambda: object())])
    reg = _robot_registry()
    registry.load_entrypoint_presets(reg)
    assert reg == {}


def test_idempotent_per_group_and_registry(monkeypatch):
    calls = {"n": 0}

    def make():
        calls["n"] += 1
        return dataclasses.replace(g1_29dof)

    _patch_entry_points(monkeypatch, [_FakeEP("x", "p:x", make)])
    reg = _robot_registry()
    registry.load_entrypoint_presets(reg)
    registry.load_entrypoint_presets(reg)
    assert calls["n"] == 1  # second call is a no-op


def test_no_group_registry_is_noop(monkeypatch):
    _patch_entry_points(monkeypatch, [_FakeEP("x", "p:x", lambda: dataclasses.replace(g1_29dof))])
    reg = ConfigRegistry(RobotConfig, group=None)
    registry.load_entrypoint_presets(reg)
    assert reg == {}


# --- --import-file loading ---------------------------------------------------------------------


def test_file_preset_adds_to_registry(tmp_path):
    # A file targets a registry explicitly and calls .add(); no type routing or namespace scan.
    f = tmp_path / "preset.py"
    f.write_text(
        textwrap.dedent(
            """
            from dataclasses import replace
            from holosoma.config_values.robot import ROBOT_REGISTRY, g1_29dof
            ROBOT_REGISTRY.add("my_robot", replace(g1_29dof, asset=replace(g1_29dof.asset, armature=0.05)))
            """
        )
    )
    from holosoma.config_values.robot import ROBOT_REGISTRY

    try:
        registry.load_file_presets([str(f)])
        assert ROBOT_REGISTRY["my_robot"].asset.armature == 0.05
        assert ROBOT_REGISTRY["g1_29dof"] is g1_29dof  # imported base was NOT re-registered
    finally:
        ROBOT_REGISTRY.pop("my_robot", None)


def test_file_preset_module_is_registered_during_execution(tmp_path):
    f = tmp_path / "dataclass_preset.py"
    f.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import sys
            from dataclasses import dataclass

            assert __name__ in sys.modules

            @dataclass(frozen=True)
            class LocalPreset:
                value: int = 7
            """
        )
    )

    module = registry._import_module_from_path(str(f))
    assert module.LocalPreset().value == 7


def test_file_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        registry.load_file_presets([str(tmp_path / "nope.py")])


# --- UseRegistry marker resolution ---------------------------------------------------------------


def test_registry_from_annotated_reads_marker():
    from typing_extensions import Annotated

    from holosoma.config_values.logger import LOGGER_REGISTRY
    from holosoma.utils.config_registry import UseRegistry, registry_from_annotated

    hint = Annotated[LoggerConfig, UseRegistry(LOGGER_REGISTRY)]
    assert registry_from_annotated(hint) is LOGGER_REGISTRY


def test_registry_from_annotated_unmarked_is_none():
    from typing_extensions import Annotated

    from holosoma.utils.config_registry import registry_from_annotated

    # Bare type, and Annotated without a UseRegistry marker, both resolve to None.
    assert registry_from_annotated(RobotConfig) is None
    assert registry_from_annotated(Annotated[RobotConfig, "some other metadata"]) is None


def test_registry_from_annotated_disambiguates_shared_type():
    # SimulatorConfig backs BOTH the simulator and run_sim registries; the marker picks one
    # explicitly, so there is no ambiguity to resolve by type.
    from typing_extensions import Annotated

    from holosoma.config_values.run_sim import RUN_SIM_REGISTRY
    from holosoma.config_values.simulator import SIMULATOR_REGISTRY
    from holosoma.utils.config_registry import UseRegistry, registry_from_annotated

    assert registry_from_annotated(Annotated[SimulatorConfig, UseRegistry(RUN_SIM_REGISTRY)]) is RUN_SIM_REGISTRY
    assert registry_from_annotated(Annotated[SimulatorConfig, UseRegistry(SIMULATOR_REGISTRY)]) is SIMULATOR_REGISTRY


# --- parse_config ------------------------------------------------------------------------------


# Registry-bound value types for the synthetic dynamic-dict configs below. Declared at module
# scope (not inside a test) so the string annotations resolve when tyro re-evaluates them under
# ``from __future__ import annotations``.
_LoggerField: TypeAlias = Annotated[LoggerConfig, UseRegistry(LOGGER_REGISTRY)]
_RunSimField: TypeAlias = Annotated[SimulatorConfig, UseRegistry(RUN_SIM_REGISTRY)]


@dataclasses.dataclass(frozen=True)
class _SynthConfig:
    loggers: dict[str, _LoggerField] = dataclasses.field(default_factory=dict)
    num: int = 1


@dataclasses.dataclass(frozen=True)
class _SimsConfig:
    sims: dict[str, _RunSimField] = dataclasses.field(default_factory=dict)


def test_parse_no_declarations():
    cfg = registry.parse_config(_SynthConfig, args=["--num=5"])
    assert isinstance(cfg, _SynthConfig)
    assert cfg.loggers == {}
    assert cfg.num == 5


def test_parse_dynamic_keys_resolve_variants():
    cfg = registry.parse_config(_SynthConfig, args=["loggers.main:disabled", "loggers.cloud:wandb", "--num=3"])
    assert isinstance(cfg.loggers["main"], DisabledLoggerConfig)
    assert isinstance(cfg.loggers["cloud"], WandbLoggerConfig)
    assert cfg.num == 3


def test_parse_dynamic_key_leaf_override():
    cfg = registry.parse_config(_SynthConfig, args=["loggers.cloud:wandb", "--loggers.cloud.base-dir=/tmp/runs"])
    assert cfg.loggers["cloud"].base_dir == "/tmp/runs"


def test_parse_unknown_variant_fails_loud():
    with pytest.raises(SystemExit, match="Unknown loggers variant 'bogus'"):
        registry.parse_config(_SynthConfig, args=["loggers.x:bogus"])


def test_parse_dynamic_dict_shared_type_disambiguated_by_marker():
    # SimulatorConfig backs both SIMULATOR_REGISTRY and RUN_SIM_REGISTRY. The _SimsConfig field
    # is marked with RUN_SIM_REGISTRY, so it resolves to exactly that registry — the marker
    # removes what used to be an unresolvable-by-type ambiguity. run_sim presets have the bridge
    # enabled; base simulator presets do not, so the resolved variant is observable.
    cfg = registry.parse_config(_SimsConfig, args=["sims.a:mujoco", "sims.b:mjwarp"])
    assert set(cfg.sims) == {"a", "b"}
    # run_sim preset -> bridge enabled (distinguishes it from the base simulator registry).
    assert cfg.sims["a"].config.bridge.enabled is True
    assert cfg.sims["b"].config.mujoco_backend.value == "warp"


def test_parse_factory_evaluated_after_plugins(monkeypatch):
    order: list[str] = []
    real_load = registry.load_plugins

    def spy_load(paths=None):
        order.append("load_plugins")
        return real_load(paths)

    def factory():
        order.append("factory")
        return _SynthConfig

    monkeypatch.setattr(registry, "load_plugins", spy_load)
    registry.parse_config(factory, args=["--num=1"])
    assert order == ["load_plugins", "factory"]


def test_parse_caller_default_is_merged():
    base = _SynthConfig(num=42)
    cfg = registry.parse_config(_SynthConfig, args=["loggers.a:disabled"], default=base)
    assert cfg.num == 42
    assert isinstance(cfg.loggers["a"], DisabledLoggerConfig)


def test_parse_required_field_needs_explicit_default():
    @dataclasses.dataclass(frozen=True)
    class _Req:
        name: str
        loggers: dict[str, _LoggerField] = dataclasses.field(default_factory=dict)

    with pytest.raises(TypeError, match="required fields.*explicit default"):
        registry.parse_config(_Req, args=["loggers.a:disabled"])

    cfg = registry.parse_config(_Req, args=["loggers.a:disabled"], default=_Req(name="x"))
    assert cfg.name == "x"
    assert isinstance(cfg.loggers["a"], DisabledLoggerConfig)


def test_parse_hyphenated_variant_matches():
    from holosoma.config_values.logger import LOGGER_REGISTRY

    LOGGER_REGISTRY.add("my-logger", WandbLoggerConfig())
    try:
        cfg = registry.parse_config(_SynthConfig, args=["loggers.x:my-logger"])
        assert isinstance(cfg.loggers["x"], WandbLoggerConfig)
    finally:
        LOGGER_REGISTRY.pop("my-logger", None)


def test_optional_dict_field_is_scanned():
    from holosoma.config_values.logger import LOGGER_REGISTRY
    from holosoma.utils.config_registry import registry_from_annotated
    from holosoma.utils.tyro_utils import find_dynamic_dict_fields

    @dataclasses.dataclass(frozen=True)
    class _Opt:
        loggers: Optional[dict[str, _LoggerField]] = None  # noqa: UP007 - covers typing.Union origin
        pipe: dict[str, _LoggerField] | None = None  # covers types.UnionType origin (PEP 604)

    found = find_dynamic_dict_fields(_Opt)
    assert set(found) == {"loggers", "pipe"}
    # find_dynamic_dict_fields keeps the Annotated metadata so the marker survives the Optional unwrap.
    assert registry_from_annotated(found["loggers"]) is LOGGER_REGISTRY
    assert registry_from_annotated(found["pipe"]) is LOGGER_REGISTRY


def test_deprecated_defaults_alias_warns():
    import holosoma.config_values.robot as robot_mod

    with pytest.warns(DeprecationWarning, match="DEFAULTS is deprecated"):
        assert robot_mod.DEFAULTS is robot_mod.ROBOT_REGISTRY
