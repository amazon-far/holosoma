"""Validate set-based mirror tags, explicit joint pairs, and saved symmetry configurations."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import pytest
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass

from holosoma.config_types.algo import DistillationConfig, FastSACAlgoConfig, JointSymmetryConfig, PPOAlgoConfig
from holosoma.config_types.experiment import ExperimentConfig, SupportsJointSymmetry
from holosoma.config_types.robot import RobotConfig
from holosoma.config_values.algo import distillation
from holosoma.config_values.experiment import EXPERIMENT_REGISTRY
from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.loco.t1.experiment import t1_29dof
from holosoma.config_values.robot import g1_29dof as g1_robot
from holosoma.utils.config_migration import migrate_legacy_experiment_config
from holosoma.utils.name_tags import expand_name_spec

pytestmark = pytest.mark.no_sim
_NAMES = ["left", "right", "other_left", "other_right", "center"]
_TAGS = {"#mirror_first": ["right", "left"], "#mirror_second": ["other_right", "other_left"]}


@dataclass(frozen=True)
class _CustomSymmetryConfig(DistillationConfig):
    use_symmetry: bool = True
    symmetry: JointSymmetryConfig = dataclasses.field(default_factory=JointSymmetryConfig)


@dataclass(frozen=True)
class _MissingSymmetryConfig(DistillationConfig):
    use_symmetry: bool = True


def _symmetry_config(exp: ExperimentConfig) -> SupportsJointSymmetry:
    assert isinstance(exp.algo.config, SupportsJointSymmetry)
    return exp.algo.config


def test_custom_algorithm_can_validate_symmetry() -> None:
    symmetry = JointSymmetryConfig(pairs=[("left_knee_joint", "right_knee_joint")], flip_sign=[])
    algo = dataclasses.replace(
        distillation, config=_CustomSymmetryConfig(module=distillation.config.module, symmetry=symmetry)
    )
    cfg = dataclasses.replace(g1_29dof, algo=algo)
    assert _symmetry_config(cfg).symmetry.resolve(cfg.robot.joint_tags, cfg.robot.dof_names) == (
        ["left_knee_joint"],
        ["right_knee_joint"],
        [],
    )


def test_custom_algorithm_invalid_symmetry_fails_at_construction() -> None:
    symmetry = JointSymmetryConfig(pairs=[("left_knee_joint", "missing_joint")], flip_sign=[])
    algo = dataclasses.replace(
        distillation, config=_CustomSymmetryConfig(module=distillation.config.module, symmetry=symmetry)
    )
    with pytest.raises(ValueError, match="unknown joint names.*missing_joint"):
        dataclasses.replace(g1_29dof, algo=algo)


def test_symmetry_enabled_algorithm_requires_symmetry_specification() -> None:
    algo = dataclasses.replace(distillation, config=_MissingSymmetryConfig(module=distillation.config.module))
    with pytest.raises(ValueError, match="use_symmetry=True requires.*symmetry"):
        dataclasses.replace(g1_29dof, algo=algo)


@pytest.mark.parametrize(
    ("pairs", "match"),
    [
        ([], "at least one"),
        ([("left", "left")], "distinct"),
        ([("#left", "right")], "exact joint names"),
        ([("", "right")], "exact joint names"),
        ([("left", "right"), ("left", "other_right")], "multiple mirror pairs"),
        ([("left", "right"), ("right", "left")], "multiple mirror pairs"),
        ([("left", "right", "center")], "too_long"),
        ([("left",)], "Field required"),
    ],
)
def test_invalid_explicit_pair_definitions_fail_before_robot_loading(pairs: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        JointSymmetryConfig(pairs=pairs)


@pytest.mark.parametrize("pattern", ["", "[", "*"])
def test_invalid_regex_fails_even_with_explicit_pairs(pattern: str) -> None:
    with pytest.raises(ValueError, match="mirror_tag_pattern"):
        JointSymmetryConfig(pairs=[("left", "right")], mirror_tag_pattern=pattern)


def test_old_ordered_fields_are_not_silently_ignored() -> None:
    with pytest.raises(ValueError, match="Unexpected keyword argument"):
        JointSymmetryConfig(left=["left"], right=["right"])  # type: ignore[call-arg]


@pytest.mark.parametrize("members", [["left"], ["left", "right", "center"], ["left", "left"]])
def test_matched_tags_require_two_distinct_joints(members: list[str]) -> None:
    with pytest.raises(ValueError, match="mirror tag.*two distinct joints"):
        JointSymmetryConfig().resolve({"#mirror_bad": members}, _NAMES)


def test_overlapping_discovered_pairs_name_both_tags() -> None:
    with pytest.raises(ValueError, match="left.*multiple mirror pairs.*#mirror_first.*#mirror_second"):
        JointSymmetryConfig().resolve(
            {"#mirror_first": ["left", "right"], "#mirror_second": ["left", "other_right"]}, _NAMES
        )


def test_custom_regex_matches_full_tag_names_without_the_prefix() -> None:
    tags = {"#bilateral_knee": ["right", "left"], "#not_bilateral_knee": ["center"]}
    assert JointSymmetryConfig(mirror_tag_pattern=r"bilateral_.*").resolve(tags, _NAMES) == (["left"], ["right"], [])
    with pytest.raises(ValueError, match="at least one"):
        JointSymmetryConfig(mirror_tag_pattern="bilateral").resolve(tags, _NAMES)


def test_explicit_pairs_override_tag_discovery() -> None:
    assert JointSymmetryConfig(pairs=[("right", "left")], flip_sign=[]).resolve(
        {"#mirror_unused_group": ["left", "right", "center"]}, _NAMES
    ) == (["left"], ["right"], [])


def test_explicit_pairs_reject_unknown_joints() -> None:
    with pytest.raises(ValueError, match="unknown joint names.*missing"):
        JointSymmetryConfig(pairs=[("left", "missing")]).resolve(_TAGS, _NAMES)


def test_tag_order_pair_order_and_repeated_members_do_not_change_symmetry() -> None:
    expected: tuple[list[str], list[str], list[str]] = (["left", "other_left"], ["right", "other_right"], [])
    assert JointSymmetryConfig().resolve(_TAGS, _NAMES) == expected
    tags = {"#mirror_second": ["other_left", "other_right", "other_left"], "#mirror_first": ["left", "right"]}
    assert JointSymmetryConfig().resolve(tags, _NAMES) == expected
    explicit = JointSymmetryConfig(pairs=[("other_right", "other_left"), ("right", "left")])
    assert explicit.resolve({}, _NAMES) == expected


def test_nested_pair_tags_use_unique_expanded_members() -> None:
    tags = {"#mirror_pair": ["#both", "left"], "#both": ["right", "left", "right"]}
    assert JointSymmetryConfig().resolve(tags, _NAMES) == (["left"], ["right"], [])


def test_flip_selection_is_a_set_and_can_include_unpaired_center_joints() -> None:
    tags = {**_TAGS, "#symmetry_flip_sign": ["right", "left", "center", "left"]}
    assert JointSymmetryConfig().resolve(tags, _NAMES)[2] == ["left", "right", "center"]
    assert JointSymmetryConfig(flip_sign=[]).resolve(tags, _NAMES)[2] == []
    with pytest.raises(ValueError, match="both or neither.*left.*right"):
        JointSymmetryConfig(flip_sign=["left"]).resolve(tags, _NAMES)


@pytest.mark.parametrize("exp", [g1_29dof, g1_29dof_fast_sac])
def test_enabled_symmetry_requires_mirror_pairs(exp: ExperimentConfig) -> None:
    tags = {tag: names for tag, names in exp.robot.joint_tags.items() if not tag.startswith("#mirror_")}
    robot = dataclasses.replace(exp.robot, joint_tags=tags)
    with pytest.raises(ValueError, match="at least one.*use_symmetry=False"):
        dataclasses.replace(exp, robot=robot)
    algo: PPOAlgoConfig | FastSACAlgoConfig
    if isinstance(exp.algo, PPOAlgoConfig):
        algo = dataclasses.replace(exp.algo, config=dataclasses.replace(exp.algo.config, use_symmetry=False))
    else:
        assert isinstance(exp.algo, FastSACAlgoConfig)
        algo = dataclasses.replace(exp.algo, config=dataclasses.replace(exp.algo.config, use_symmetry=False))
    disabled = dataclasses.replace(exp, algo=algo, robot=robot)
    assert _symmetry_config(disabled).use_symmetry is False


@pytest.mark.parametrize("name", sorted(EXPERIMENT_REGISTRY))
def test_every_registered_experiment_resolves_its_robot_tags(name: str) -> None:
    exp = EXPERIMENT_REGISTRY[name]
    robot = exp.robot
    if _symmetry_config(exp).use_symmetry:
        left, right, flip = _symmetry_config(exp).symmetry.resolve(robot.joint_tags, robot.dof_names)
        assert left and len(left) == len(right) and not set(left) & set(right)
        assert set(flip) <= set(robot.dof_names)
        assert len(set(left + right)) == 2 * len(left)
    contact = exp.termination.terms.get("contact")
    if contact is not None:
        bodies = expand_name_spec(contact.params["bodies"], robot.body_tags, robot.body_names, kind="body")
        assert bodies and set(bodies) <= set(robot.body_names)
        assert not set(bodies) & set(expand_name_spec(["#feet"], robot.body_tags, robot.body_names, kind="body"))


@pytest.mark.parametrize("exp", [g1_29dof, t1_29dof, g1_29dof_fast_sac])
def test_serialized_dump_round_trips_with_set_membership(exp: ExperimentConfig) -> None:
    raw = exp.to_serializable_dict()
    loaded = ExperimentConfig(**raw)
    assert loaded.robot.joint_tags == exp.robot.joint_tags
    assert _symmetry_config(loaded).symmetry == JointSymmetryConfig()
    assert loaded.to_serializable_dict() == raw
    assert all(members == sorted(set(members)) for members in raw["robot"]["joint_tags"].values())


def test_robot_stores_sets_and_serializes_arrays() -> None:
    raw = TypeAdapter[RobotConfig](RobotConfig).dump_python(g1_robot, mode="json")
    raw["joint_tags"]["#repeated"] = ["#knees", "#knees"]
    robot = RobotConfig(**raw)
    assert robot.joint_tags["#repeated"] == frozenset({"#knees"})
    assert isinstance(robot.joint_tags["#repeated"], set)
    assert expand_name_spec(["#repeated"], robot.joint_tags, robot.dof_names, kind="joint") == [
        "left_knee_joint",
        "right_knee_joint",
    ]
    encoded = TypeAdapter[RobotConfig](RobotConfig).dump_python(robot, mode="json")
    assert encoded["joint_tags"]["#repeated"] == ["#knees"]
    assert TypeAdapter[RobotConfig](RobotConfig).validate_python(encoded) == robot


def test_saved_ordered_groups_convert_before_set_normalization() -> None:
    raw = g1_29dof.to_serializable_dict()
    tags = raw["robot"]["joint_tags"]
    for tag in list(tags):
        if tag.startswith("#mirror_"):
            del tags[tag]
    left = ["left_knee_joint", "left_ankle_pitch_joint"]
    right = ["right_ankle_pitch_joint", "right_knee_joint"]
    tags["#mirror_left"] = left
    tags["#mirror_right"] = right
    tags["#mirror_flip_sign"] = tags.pop("#symmetry_flip_sign")
    raw["algo"]["config"]["symmetry"] = {"left": ["#mirror_left"], "right": ["#mirror_right"], "flip_sign": []}
    original = copy.deepcopy(raw)
    migrated = migrate_legacy_experiment_config(raw)
    loaded = ExperimentConfig(**migrated)
    assert _symmetry_config(loaded).symmetry.pairs == list(zip(left, right, strict=True))
    resolved = _symmetry_config(loaded).symmetry.resolve(loaded.robot.joint_tags, loaded.robot.dof_names)
    assert resolved == (left, right, [])
    assert raw == original
    assert migrate_legacy_experiment_config(migrated) == migrated


@pytest.mark.parametrize("conflicting", [False, True])
def test_saved_json_pairs_must_agree_with_ordered_sides(conflicting: bool) -> None:
    raw = g1_29dof.to_serializable_dict()
    left, right = "left_knee_joint", "right_knee_joint"
    raw["algo"]["config"]["symmetry"] = {
        "left": [left],
        "right": [right],
        "pairs": [[left, "right_ankle_pitch_joint" if conflicting else right]],
        "flip_sign": [],
    }
    if conflicting:
        with pytest.raises(ValueError, match="conflicting pairs"):
            migrate_legacy_experiment_config(raw)
    else:
        loaded = ExperimentConfig(**migrate_legacy_experiment_config(raw))
        assert _symmetry_config(loaded).symmetry.resolve(loaded.robot.joint_tags, loaded.robot.dof_names) == (
            [left],
            [right],
            [],
        )
