"""Config-layer tests for ``JointSymmetryConfig`` and the ``ExperimentConfig`` symmetry validator (no simulator).

Symmetry augmentation is an algorithm concern: the robot only declares ``joint_tags``, and
``algo.config.symmetry`` names them. ``ExperimentConfig`` resolves the spec against the robot at
construction whenever ``use_symmetry`` is on, so a mismatch fails before any simulator starts.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

import holosoma.utils.config_migration
from holosoma.config_types.algo import FastSACConfig, JointSymmetryConfig, PPOConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import EXPERIMENT_REGISTRY
from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.loco.t1.experiment import t1_29dof
from holosoma.config_values.robot import g1_29dof as g1_robot
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt
from holosoma.utils.config_migration import migrate_legacy_experiment_config
from holosoma.utils.name_tags import expand_name_spec

pytestmark = pytest.mark.no_sim

_LEGACY_DUMPS = Path(holosoma.utils.config_migration.__file__).parent / "tests" / "data"


def _with_algo_config(exp: ExperimentConfig, **overrides: Any) -> ExperimentConfig:
    algo = dataclasses.replace(exp.algo, config=dataclasses.replace(exp.algo.config, **overrides))  # type: ignore[arg-type]
    return dataclasses.replace(exp, algo=algo)


def _without_mirror_tags(joint_tags: dict[str, list[str]]) -> dict[str, list[str]]:
    """``joint_tags`` minus the ``mirror_*`` tags; the other tags stay because the PD gain keys reference them."""
    return {tag: names for tag, names in joint_tags.items() if not tag.startswith("#mirror_")}


def test_joint_symmetry_defaults_reference_mirror_tags() -> None:
    symmetry = JointSymmetryConfig()
    assert symmetry.left == ["#mirror_left"]
    assert symmetry.right == ["#mirror_right"]
    assert symmetry.flip_sign == ["#mirror_flip_sign"]


def test_algo_configs_carry_symmetry_default() -> None:
    assert PPOConfig(module_dict=g1_29dof.algo.config.module_dict).symmetry == JointSymmetryConfig()  # type: ignore[union-attr]
    assert FastSACConfig().symmetry == JointSymmetryConfig()
    assert g1_29dof.algo.config.symmetry == JointSymmetryConfig()
    assert g1_29dof_fast_sac.algo.config.symmetry == JointSymmetryConfig()


@pytest.mark.parametrize("exp", [g1_29dof, g1_29dof_fast_sac], ids=["ppo", "fast_sac"])
def test_use_symmetry_with_robot_lacking_mirror_tags_raises(exp: ExperimentConfig) -> None:
    assert exp.algo.config.use_symmetry is True
    robot = dataclasses.replace(g1_robot, joint_tags=_without_mirror_tags(g1_robot.joint_tags))
    with pytest.raises(ValueError, match=r"Unknown joint tag '#mirror_left'.*use_symmetry=False"):
        dataclasses.replace(exp, robot=robot)


def test_use_symmetry_false_with_robot_lacking_mirror_tags_constructs() -> None:
    assert g1_29dof_wbt.algo.config.use_symmetry is False
    robot = dataclasses.replace(g1_29dof_wbt.robot, joint_tags=_without_mirror_tags(g1_29dof_wbt.robot.joint_tags))
    cfg = dataclasses.replace(g1_29dof_wbt, robot=robot)
    assert not {"#mirror_left", "#mirror_right", "#mirror_flip_sign"} & set(cfg.robot.joint_tags)

    # The same robot passes a use_symmetry=True experiment once symmetry is switched off.
    cfg = dataclasses.replace(_with_algo_config(g1_29dof, use_symmetry=False), robot=robot)
    assert cfg.algo.config.use_symmetry is False


def test_unequal_left_right_lengths_raise() -> None:
    symmetry = JointSymmetryConfig(left=["#left_leg"], right=["#right_leg", "right_shoulder_pitch_joint"])
    with pytest.raises(ValueError, match=r"left \(6 joints\) and \.right \(7 joints\)"):
        _with_algo_config(g1_29dof, symmetry=symmetry)


def test_empty_left_right_raise() -> None:
    with pytest.raises(ValueError, match=r"left \(0 joints\) and \.right \(0 joints\)"):
        _with_algo_config(g1_29dof, symmetry=JointSymmetryConfig(left=[], right=[]))


def test_overlapping_left_right_raise() -> None:
    symmetry = JointSymmetryConfig(left=["#left_leg"], right=["#left_leg"])
    with pytest.raises(ValueError, match=r"share joints \['left_ankle_pitch_joint'"):
        _with_algo_config(g1_29dof, symmetry=symmetry)


def test_unknown_joint_name_in_spec_raises() -> None:
    symmetry = JointSymmetryConfig(left=["not_a_joint"], right=["right_knee_joint"])
    with pytest.raises(ValueError, match=r"Unknown joint name 'not_a_joint'"):
        _with_algo_config(g1_29dof, symmetry=symmetry)


def test_explicit_names_and_tags_mix_constructs() -> None:
    symmetry = JointSymmetryConfig(
        left=["#left_leg", "left_shoulder_pitch_joint"],
        right=["#right_leg", "right_shoulder_pitch_joint"],
        flip_sign=["waist_yaw_joint"],
    )
    cfg = _with_algo_config(g1_29dof, symmetry=symmetry)
    assert cfg.algo.config.symmetry == symmetry


@pytest.mark.parametrize("name", sorted(EXPERIMENT_REGISTRY))
def test_every_registered_experiment_resolves_its_robot_tag_references(name: str) -> None:
    """The specs the shipped terms hand to the env resolvers expand against the preset's own robot."""
    exp = EXPERIMENT_REGISTRY[name]
    robot = exp.robot
    if exp.algo.config.use_symmetry:
        symmetry = exp.algo.config.symmetry
        left = expand_name_spec(symmetry.left, robot.joint_tags, robot.dof_names, kind="joint")
        right = expand_name_spec(symmetry.right, robot.joint_tags, robot.dof_names, kind="joint")
        assert len(left) == len(right) > 0 and not set(left) & set(right)
        assert expand_name_spec(symmetry.flip_sign, robot.joint_tags, robot.dof_names, kind="joint")
    contact = exp.termination.terms.get("contact")
    if contact is not None:
        bodies = expand_name_spec(contact.params["bodies"], robot.body_tags, robot.body_names, kind="body")
        assert bodies and set(bodies) <= set(robot.body_names)
        feet = expand_name_spec(["#feet"], robot.body_tags, robot.body_names, kind="body")
        assert not set(bodies) & set(feet)  # a foot on the ground must not terminate the episode
    if exp.env_class.endswith("WholeBodyTrackingManager"):
        assert not robot.asset.fix_base_link  # MotionCommand writes the clip's root pose and refuses a fixed base


@pytest.mark.parametrize("exp", [g1_29dof, t1_29dof, g1_29dof_fast_sac], ids=["g1", "t1", "g1_fast_sac"])
def test_serialized_dump_round_trips_through_validator(exp: ExperimentConfig) -> None:
    """A dump whose robot declares the tags loads back via ``ExperimentConfig(**dict)`` (eval_utils path)."""
    assert exp.algo.config.use_symmetry is True
    loaded = ExperimentConfig(**exp.to_serializable_dict())
    assert loaded.algo.config.symmetry == JointSymmetryConfig()
    assert loaded.robot.joint_tags == exp.robot.joint_tags


def _mirror_pairs(
    symmetry: JointSymmetryConfig, joint_tags: dict[str, list[str]], dof_names: list[str]
) -> set[frozenset[str]]:
    left = expand_name_spec(symmetry.left, joint_tags, dof_names, kind="joint")
    right = expand_name_spec(symmetry.right, joint_tags, dof_names, kind="joint")
    return {frozenset(pair) for pair in zip(left, right)}


@pytest.mark.parametrize("exp", [g1_29dof, t1_29dof], ids=["g1", "t1"])
def test_migrated_legacy_dump_satisfies_symmetry_validator(exp: ExperimentConfig) -> None:
    """A pre-refactor checkpoint dump (``use_symmetry: true``, no ``joint_tags``) passes the validator once
    ``migrate_legacy_experiment_config`` has derived the mirror tags, and those tags reproduce the shipped preset."""
    robot_type = exp.robot.asset.robot_type
    raw = yaml.safe_load((_LEGACY_DUMPS / f"legacy_{robot_type}_experiment_config.yaml").read_text())
    assert raw["algo"]["config"]["use_symmetry"] is True
    assert "joint_tags" not in raw["robot"]
    gain_tags = _without_mirror_tags(exp.robot.joint_tags)
    with pytest.raises(ValueError, match=r"Unknown joint tag '#mirror_left'"):
        dataclasses.replace(exp, robot=dataclasses.replace(exp.robot, joint_tags=gain_tags))

    migrated_robot = migrate_legacy_experiment_config(raw)["robot"]
    assert migrated_robot["dof_names"] == list(exp.robot.dof_names)
    migrated_tags: dict[str, list[str]] = migrated_robot["joint_tags"]
    assert set(migrated_tags) == {"#mirror_left", "#mirror_right", "#mirror_flip_sign"}

    cfg = dataclasses.replace(exp, robot=dataclasses.replace(exp.robot, joint_tags={**gain_tags, **migrated_tags}))
    symmetry = cfg.algo.config.symmetry
    dof_names = list(exp.robot.dof_names)
    assert _mirror_pairs(symmetry, migrated_tags, dof_names) == _mirror_pairs(symmetry, exp.robot.joint_tags, dof_names)
    assert set(expand_name_spec(symmetry.flip_sign, migrated_tags, dof_names, kind="joint")) == set(
        expand_name_spec(symmetry.flip_sign, exp.robot.joint_tags, dof_names, kind="joint")
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (["left_knee_joint", "#left_leg", "#left_arm"], ["#right_leg", "#right_arm"]),  # duplicate on the left only
        (["#left_leg", "#left_arm"], ["#right_leg", "#right_arm", "right_knee_joint"]),  # on the right only
    ],
    ids=["left", "right"],
)
def test_duplicate_joint_on_one_side_raises(left: list[str], right: list[str]) -> None:
    # A joint appears once per side; tags dedupe on expansion, so the check counts before that collapse.
    with pytest.raises(ValueError, match=r"more than once"):
        _with_algo_config(g1_29dof, symmetry=JointSymmetryConfig(left=left, right=right))


def test_duplicate_joint_inside_a_composed_tag_raises() -> None:
    # The duplicate hides inside the robot's own composed mirror tag (the spec is just the default "#mirror_left").
    joint_tags = {**g1_29dof.robot.joint_tags, "#mirror_left": ["left_knee_joint", "#left_leg", "#left_arm"]}
    with pytest.raises(ValueError, match=r"left_knee_joint.*more than once.*#mirror_left"):
        dataclasses.replace(g1_29dof, robot=dataclasses.replace(g1_29dof.robot, joint_tags=joint_tags))


def test_sign_flips_must_preserve_double_mirroring() -> None:
    with pytest.raises(ValueError, match=r"flip_sign.*left_hip_yaw_joint.*right_hip_yaw_joint"):
        _with_algo_config(g1_29dof, symmetry=JointSymmetryConfig(flip_sign=["left_hip_yaw_joint"]))


def test_flip_sign_is_a_selection_not_an_ordered_correspondence() -> None:
    symmetry = JointSymmetryConfig(flip_sign=["#mirror_flip_sign", "left_hip_yaw_joint"])
    cfg = _with_algo_config(g1_29dof, symmetry=symmetry)
    left, right, flip = cfg.algo.config.symmetry.resolve(cfg.robot.joint_tags, cfg.robot.dof_names)
    assert left[0] == "left_hip_pitch_joint"
    assert right[0] == "right_hip_pitch_joint"
    assert flip.count("left_hip_yaw_joint") == 1
    assert cfg.to_serializable_dict()["algo"]["config"]["symmetry"]["flip_sign"] == symmetry.flip_sign
