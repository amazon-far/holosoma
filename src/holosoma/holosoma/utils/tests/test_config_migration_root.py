"""Saved WBT configs keep their known former root body, without inventing a root tag."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from loguru import logger

from holosoma.config_values.experiment import EXPERIMENT_REGISTRY
from holosoma.utils.config_migration import migrate_legacy_experiment_config

pytestmark = pytest.mark.no_sim

_FUNC = "holosoma.managers.command.terms.wbt:MotionCommand"


def _motion(raw: dict[str, Any]) -> dict[str, Any]:
    motion: dict[str, Any] = raw["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]
    return motion


def _legacy() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(
        (Path(__file__).parent / "data/legacy_g1_29dof_experiment_config.yaml").read_text()
    )
    raw["command"] = {
        "setup_terms": {
            "motion_command": {
                "func": _FUNC,
                "params": {
                    "motion_config": {
                        "motion_file": "motion.npz",
                        "body_name_ref": ["torso_link"],
                        "body_names_to_track": ["pelvis", "torso_link"],
                    }
                },
            }
        },
        "reset_terms": {"motion_command": {"func": _FUNC, "params": {}}},
        "step_terms": {"motion_command": {"func": _FUNC, "params": {}}},
    }
    return raw


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink)


@pytest.mark.parametrize("backend", ["isaacgym.IsaacGym", "isaacsim.IsaacSim"])
@pytest.mark.parametrize("separator", [":", "."])
def test_legacy_known_body_order_becomes_exact_selector(backend: str, separator: str, warnings: list[str]) -> None:
    raw = _legacy()
    module, _ = backend.split(".")
    raw["simulator"]["_target_"] = f"holosoma.simulator.{module}.{backend}"
    raw["command"]["setup_terms"]["motion_command"]["func"] = _FUNC.replace(":", separator)
    original = copy.deepcopy(raw)

    migrated = migrate_legacy_experiment_config(raw)

    assert _motion(migrated)["root_body"] == "pelvis"
    assert "#root" not in migrated["robot"]["body_tags"]
    assert migrated["command"]["reset_terms"] == raw["command"]["reset_terms"]
    assert migrated["command"]["step_terms"] == raw["command"]["step_terms"]
    assert raw == original
    assert migrate_legacy_experiment_config(migrated) == migrated
    assert any("MotionConfig.root_body" in warning and "pelvis" in warning for warning in warnings)


@pytest.mark.parametrize(
    "target",
    [
        "holosoma.simulator.mujoco.mujoco.MuJoCo",
        "custom.simulator.Unknown",
        None,
    ],
)
def test_old_unproven_runtime_order_requires_explicit_choice(target: str | None) -> None:
    raw = _legacy()
    raw["simulator"]["_target_"] = target

    with pytest.raises(ValueError, match=r"Cannot infer MotionConfig.root_body.*pre-refactor simulator"):
        migrate_legacy_experiment_config(raw)


@pytest.mark.parametrize("selector", ["torso_link", "#root", "", "#missing"])
def test_explicit_selector_is_preserved_even_for_legacy_mujoco(selector: str) -> None:
    raw = _legacy()
    raw["simulator"]["_target_"] = "holosoma.simulator.mujoco.mujoco.MuJoCo"
    _motion(raw)["root_body"] = selector

    migrated = migrate_legacy_experiment_config(raw)

    # Migration must not repair or reinterpret an explicit current-schema selection, even an invalid one.
    assert _motion(migrated)["root_body"] == selector


def test_pre_selector_canonical_dump_preserves_first_body_despite_different_root_tag(warnings: list[str]) -> None:
    raw = EXPERIMENT_REGISTRY["g1_29dof_wbt"].to_serializable_dict()
    _motion(raw).pop("root_body")
    # Canonical order was already independent of backend order in this schema.
    raw["simulator"]["_target_"] = "holosoma.simulator.mujoco.mujoco.MuJoCo"
    raw["robot"]["body_names"] = list(reversed(raw["robot"]["body_names"]))
    expected = raw["robot"]["body_names"][0]
    original = copy.deepcopy(raw)

    migrated = migrate_legacy_experiment_config(raw)

    assert _motion(migrated)["root_body"] == expected
    assert migrated["robot"] == original["robot"]
    assert raw == original
    assert len(warnings) == 1 and "MotionConfig.root_body" in warnings[0]


def test_current_dump_is_unchanged_and_silent(warnings: list[str]) -> None:
    raw = EXPERIMENT_REGISTRY["g1_29dof_wbt"].to_serializable_dict()
    assert _motion(raw)["root_body"] is None

    assert migrate_legacy_experiment_config(raw) == raw
    assert warnings == []


def test_missing_saved_body_order_cannot_be_migrated() -> None:
    raw = _legacy()
    raw["robot"]["body_names"] = []

    with pytest.raises(ValueError, match="nonempty saved robot.body_names"):
        migrate_legacy_experiment_config(raw)
