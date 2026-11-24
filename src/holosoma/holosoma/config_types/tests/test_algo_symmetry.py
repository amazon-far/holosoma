"""Config-layer tests for ``JointSymmetryConfig`` and the ``ExperimentConfig`` symmetry validator (no simulator).

Symmetry augmentation is an algorithm concern: the robot only declares ``joint_tags``, and
``algo.config.symmetry`` names them. ``ExperimentConfig`` resolves the spec against the robot at
construction whenever ``use_symmetry`` is on, so a mismatch fails before any simulator starts.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from pydantic.dataclasses import dataclass

from holosoma.config_types.algo import DistillationConfig, FastSACConfig, JointSymmetryConfig, PPOConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.algo import distillation
from holosoma.config_values.experiment import EXPERIMENT_REGISTRY
from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.loco.t1.experiment import t1_29dof
from holosoma.config_values.robot import g1_29dof as g1_robot
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt
from holosoma.utils.name_tags import expand_name_spec

pytestmark = pytest.mark.no_sim


@dataclass(frozen=True)
class _CustomSymmetryConfig(DistillationConfig):
    use_symmetry: bool = True
    symmetry: JointSymmetryConfig = dataclasses.field(default_factory=JointSymmetryConfig)


@dataclass(frozen=True)
class _MissingSymmetryConfig(DistillationConfig):
    use_symmetry: bool = True


def test_custom_algorithm_can_validate_symmetry() -> None:
    symmetry = JointSymmetryConfig(left=["left_knee_joint"], right=["right_knee_joint"], flip_sign=[])
    algo = dataclasses.replace(
        distillation, config=_CustomSymmetryConfig(module=distillation.config.module, symmetry=symmetry)
    )
    cfg = dataclasses.replace(g1_29dof, algo=algo)
    assert cfg.algo.config.symmetry.resolve(cfg.robot.joint_tags, cfg.robot.dof_names) == (  # type: ignore[union-attr]
        ["left_knee_joint"],
        ["right_knee_joint"],
        [],
    )


def test_custom_algorithm_invalid_symmetry_fails_at_construction() -> None:
    symmetry = JointSymmetryConfig(left=["left_knee_joint"], right=["missing_joint"], flip_sign=[])
    algo = dataclasses.replace(
        distillation, config=_CustomSymmetryConfig(module=distillation.config.module, symmetry=symmetry)
    )
    with pytest.raises(ValueError, match="Unknown joint name 'missing_joint'"):
        dataclasses.replace(g1_29dof, algo=algo)


def test_symmetry_enabled_algorithm_requires_symmetry_specification() -> None:
    algo = dataclasses.replace(distillation, config=_MissingSymmetryConfig(module=distillation.config.module))
    with pytest.raises(ValueError, match="use_symmetry=True requires.*symmetry"):
        dataclasses.replace(g1_29dof, algo=algo)


def _symmetry_config(exp: ExperimentConfig) -> PPOConfig | FastSACConfig:
    assert isinstance(exp.algo.config, (PPOConfig, FastSACConfig))
    return exp.algo.config


def _with_algo_config(exp: ExperimentConfig, **overrides: Any) -> ExperimentConfig:
    algo = dataclasses.replace(exp.algo, config=dataclasses.replace(exp.algo.config, **overrides))  # type: ignore[arg-type]
    return dataclasses.replace(exp, algo=algo)


def _without_mirror_tags(joint_tags: dict[str, list[str]]) -> dict[str, list[str]]:
    """``joint_tags`` minus the ``mirror_*`` tags; the other tags stay because the PD gain keys reference them."""
    return {tag: names for tag, names in joint_tags.items() if not tag.startswith("#mirror_")}


@pytest.mark.parametrize("exp", [g1_29dof, g1_29dof_fast_sac], ids=["ppo", "fast_sac"])
def test_use_symmetry_with_robot_lacking_mirror_tags_raises(exp: ExperimentConfig) -> None:
    assert _symmetry_config(exp).use_symmetry is True
    robot = dataclasses.replace(g1_robot, joint_tags=_without_mirror_tags(g1_robot.joint_tags))
    with pytest.raises(ValueError, match=r"Unknown joint tag '#mirror_left'.*use_symmetry=False"):
        dataclasses.replace(exp, robot=robot)


def test_use_symmetry_false_with_robot_lacking_mirror_tags_constructs() -> None:
    assert _symmetry_config(g1_29dof_wbt).use_symmetry is False
    robot = dataclasses.replace(g1_29dof_wbt.robot, joint_tags=_without_mirror_tags(g1_29dof_wbt.robot.joint_tags))
    cfg = dataclasses.replace(g1_29dof_wbt, robot=robot)
    assert not {"#mirror_left", "#mirror_right", "#mirror_flip_sign"} & set(cfg.robot.joint_tags)

    # The same robot passes a use_symmetry=True experiment once symmetry is switched off.
    cfg = dataclasses.replace(_with_algo_config(g1_29dof, use_symmetry=False), robot=robot)
    assert _symmetry_config(cfg).use_symmetry is False


@pytest.mark.parametrize(
    ("symmetry", "match"),
    [
        (
            JointSymmetryConfig(left=["#left_leg"], right=["#right_leg", "right_shoulder_pitch_joint"]),
            r"left \(6 joints\) and \.right \(7 joints\)",
        ),
        (JointSymmetryConfig(left=[], right=[]), r"left \(0 joints\) and \.right \(0 joints\)"),
        (JointSymmetryConfig(left=["#left_leg"], right=["#left_leg"]), r"share joints \['left_ankle_pitch_joint'"),
        (JointSymmetryConfig(left=["not_a_joint"], right=["right_knee_joint"]), "Unknown joint name 'not_a_joint'"),
        (
            JointSymmetryConfig(left=["left_knee_joint", "#left_leg", "#left_arm"], right=["#right_leg", "#right_arm"]),
            "more than once",
        ),
        (
            JointSymmetryConfig(
                left=["#left_leg", "#left_arm"], right=["#right_leg", "#right_arm", "right_knee_joint"]
            ),
            "more than once",
        ),
        (
            JointSymmetryConfig(flip_sign=["left_hip_yaw_joint"]),
            r"flip_sign.*left_hip_yaw_joint.*right_hip_yaw_joint",
        ),
    ],
    ids=["unequal", "empty", "overlap", "unknown", "repeat-left", "repeat-right", "asymmetric-signs"],
)
def test_invalid_symmetry_correspondences_raise(symmetry: JointSymmetryConfig, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _with_algo_config(g1_29dof, symmetry=symmetry)


def test_explicit_names_and_tags_mix_constructs() -> None:
    symmetry = JointSymmetryConfig(
        left=["#left_leg", "left_shoulder_pitch_joint"],
        right=["#right_leg", "right_shoulder_pitch_joint"],
        flip_sign=["waist_yaw_joint"],
    )
    cfg = _with_algo_config(g1_29dof, symmetry=symmetry)
    assert _symmetry_config(cfg).symmetry.resolve(cfg.robot.joint_tags, cfg.robot.dof_names) == (
        [*g1_robot.joint_tags["#left_leg"], "left_shoulder_pitch_joint"],
        [*g1_robot.joint_tags["#right_leg"], "right_shoulder_pitch_joint"],
        ["waist_yaw_joint"],
    )


@pytest.mark.parametrize("name", sorted(EXPERIMENT_REGISTRY))
def test_every_registered_experiment_resolves_its_robot_tag_references(name: str) -> None:
    """The specs the shipped terms hand to the env resolvers expand against the preset's own robot."""
    exp = EXPERIMENT_REGISTRY[name]
    robot = exp.robot
    if _symmetry_config(exp).use_symmetry:
        symmetry = _symmetry_config(exp).symmetry
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
    assert _symmetry_config(exp).use_symmetry is True
    loaded = ExperimentConfig(**exp.to_serializable_dict())
    assert _symmetry_config(loaded).symmetry == JointSymmetryConfig()
    assert loaded.robot.joint_tags == exp.robot.joint_tags


def test_duplicate_joint_inside_a_composed_tag_raises() -> None:
    # The duplicate hides inside the robot's own composed mirror tag (the spec is just the default "#mirror_left").
    joint_tags = {**g1_29dof.robot.joint_tags, "#mirror_left": ["left_knee_joint", "#left_leg", "#left_arm"]}
    with pytest.raises(ValueError, match=r"left_knee_joint.*more than once.*#mirror_left"):
        dataclasses.replace(g1_29dof, robot=dataclasses.replace(g1_29dof.robot, joint_tags=joint_tags))


def test_flip_sign_is_a_selection_not_an_ordered_correspondence() -> None:
    symmetry = JointSymmetryConfig(flip_sign=["#mirror_flip_sign", "left_hip_yaw_joint"])
    cfg = _with_algo_config(g1_29dof, symmetry=symmetry)
    left, right, flip = _symmetry_config(cfg).symmetry.resolve(cfg.robot.joint_tags, cfg.robot.dof_names)
    assert left[0] == "left_hip_pitch_joint"
    assert right[0] == "right_hip_pitch_joint"
    assert flip.count("left_hip_yaw_joint") == 1
    assert cfg.to_serializable_dict()["algo"]["config"]["symmetry"]["flip_sign"] == symmetry.flip_sign
