"""Tests for inference config preset registries."""

from __future__ import annotations

import dataclasses
import textwrap

import pytest
from loguru import logger

from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.config.config_values.robot import ROBOT_REGISTRY, g1_29dof
from holosoma_inference.utils import config_registry as registry
from holosoma_inference.utils.config_registry import ConfigRegistry

# Discover all real registries once so resolution sees the full menu set regardless of order.
registry.load_plugins()


class _FakeEP:
    def __init__(self, name, value, loader):
        self.name = name
        self.value = value
        self._loader = loader

    def load(self):
        return self._loader()


@pytest.fixture(autouse=True)
def _isolate_registry_state():
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
    msgs: list[str] = []
    sink_id = logger.add(lambda m: msgs.append(m.record["message"]), level="WARNING")
    yield msgs
    logger.remove(sink_id)


def _robot_registry() -> ConfigRegistry:
    return ConfigRegistry(RobotConfig, group="holosoma.config.robot")


def _patch_entry_points(monkeypatch, eps):
    monkeypatch.setattr(registry, "entry_points", lambda **_kwargs: eps)


def test_add_returns_value_and_rejects_wrong_type():
    reg = _robot_registry()
    value = dataclasses.replace(g1_29dof)
    assert reg.add("r", value) is value
    with pytest.raises(TypeError, match="expected RobotConfig"):
        reg.add("bad", object())


def test_bad_entry_point_is_skipped_not_fatal(monkeypatch):
    def boom():
        raise RuntimeError("import blew up")

    _patch_entry_points(monkeypatch, [_FakeEP("bad", "p:bad", boom)])
    reg = _robot_registry()
    registry.load_entrypoint_presets(reg)
    assert reg == {}


def test_entry_point_type_guard_skips_foreign(monkeypatch):
    _patch_entry_points(monkeypatch, [_FakeEP("foreign", "other.pkg:thing", lambda: object())])
    reg = _robot_registry()
    registry.load_entrypoint_presets(reg)
    assert reg == {}


def test_file_preset_adds_to_registry(tmp_path):
    f = tmp_path / "preset.py"
    f.write_text(
        textwrap.dedent(
            """
            from dataclasses import replace
            from holosoma_inference.config.config_values.robot import ROBOT_REGISTRY, g1_29dof
            ROBOT_REGISTRY.add("my-robot", replace(g1_29dof))
            """
        )
    )
    try:
        registry.load_file_presets([str(f)])
        assert isinstance(ROBOT_REGISTRY["my-robot"], RobotConfig)
    finally:
        ROBOT_REGISTRY.pop("my-robot", None)


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


def test_registry_from_annotated_reads_marker():
    from typing_extensions import Annotated

    from holosoma_inference.utils.config_registry import UseRegistry, registry_from_annotated

    assert registry_from_annotated(Annotated[RobotConfig, UseRegistry(ROBOT_REGISTRY)]) is ROBOT_REGISTRY
    # Unmarked hints resolve to None (no type-based inference).
    assert registry_from_annotated(RobotConfig) is None
    assert registry_from_annotated(Annotated[RobotConfig, "unrelated"]) is None


def test_annotated_inference_config_includes_presets():
    from holosoma_inference.config.config_values.inference import (
        INFERENCE_REGISTRY,
        get_annotated_inference_config,
    )

    # Factory builds an Annotated alias; the registry holds the core presets.
    assert get_annotated_inference_config() is not None
    assert "g1-29dof-loco" in INFERENCE_REGISTRY


def test_deprecated_defaults_alias_warns():
    import holosoma_inference.config.config_values.robot as robot_mod

    with pytest.warns(DeprecationWarning, match="DEFAULTS is deprecated"):
        assert robot_mod.DEFAULTS is robot_mod.ROBOT_REGISTRY
