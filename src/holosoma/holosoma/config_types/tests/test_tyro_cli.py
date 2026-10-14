from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import pytest
import tyro
from pydantic import ValidationError

from holosoma.config_types.distribution import DistributionSpec
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.randomization import MujocoMaterialDR
from holosoma.utils.tyro_utils import TYRO_CONIFG

pytestmark = pytest.mark.no_sim


def test_experiment_config() -> None:
    assert isinstance(tyro.cli(ExperimentConfig, args=(), config=TYRO_CONIFG), ExperimentConfig)


def test_distribution_like_field_is_a_cli_flag() -> None:
    # A DistributionLike range is ONE token holding a Python literal. Without the CLI constructor
    # on DistributionLike, tyro cannot build a parser for the union's Dict[str, Any] member and
    # errors as soon as a flag targets such a field ("struct-type values requires a default value").
    cfg = tyro.cli(MujocoMaterialDR, args=("--sliding-friction", "[0.3, 0.9]"), config=TYRO_CONIFG)
    assert cfg.sliding_friction == [0.3, 0.9]

    cfg = tyro.cli(
        MujocoMaterialDR,
        args=("--sliding-friction", "{'kind': 'gaussian', 'low': 0.1, 'high': 0.9}"),
        config=TYRO_CONIFG,
    )
    # pydantic validates the dict form into the union's DistributionSpec member.
    assert cfg.sliding_friction == DistributionSpec(kind="gaussian", low=0.1, high=0.9)

    cfg = tyro.cli(MujocoMaterialDR, args=("--solref", "0", "[0.02, 0.05]"), config=TYRO_CONIFG)
    assert cfg.solref == {0: [0.02, 0.05]}


def test_distribution_like_invalid_range_rejected_at_cli() -> None:
    with pytest.raises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        tyro.cli(MujocoMaterialDR, args=("--sliding-friction", "[0.9, 0.3]"), config=TYRO_CONIFG)


def test_randomization_configs_build_eager_parser() -> None:
    # Schema exporters and shell completion build the full parser tree up front, forcing every
    # union branch that lazy parsing skips.
    with pytest.warns(DeprecationWarning, match="get_parser"):
        tyro.extras.get_parser(MujocoMaterialDR)


def test_robot_tag_dicts_expose_per_key_flags() -> None:
    # A populated dict[str, list[str]] field gets one flag per key taking a Python-literal list.
    cfg = tyro.cli(ExperimentConfig, args=("--robot.body-tags.#feet", "['left_ankle_roll_link']"), config=TYRO_CONIFG)
    assert cfg.robot.body_tags["#feet"] == ["left_ankle_roll_link"]
    assert cfg.robot.body_tags["#base"] == ["torso_link"]  # other keys keep the preset's values


def test_action_controller_gain_dict_keys_are_cli_flags() -> None:
    # Exact-name and #tag gain keys each get a flag on stiffness and damping alike; the override reaches every
    # DOF the key covers.
    cfg = tyro.cli(
        ExperimentConfig,
        args=(
            "action:g1-29dof-joint-pos",
            *("--action.terms.joint-control.controller.stiffness.#hip-yaw", "50"),
            *("--action.terms.joint-control.controller.damping.waist-yaw-joint", "3"),
            *("--action.terms.joint-control.controller.damping.#knees", "7"),
        ),
        config=TYRO_CONIFG,
    )
    robot = cfg.robot
    assert cfg.action is not None
    controller = cfg.action.terms["joint_control"].controller
    assert controller is not None
    assert controller.stiffness["#hip_yaw"] == 50.0 and controller.damping["waist_yaw_joint"] == 3.0
    assert controller.damping["#knees"] == 7.0
    kp, kd = controller.resolve_gains(robot)
    assert [kp[robot.dof_names.index(n)] for n in ("left_hip_yaw_joint", "right_hip_yaw_joint")] == [50.0, 50.0]
    assert kd[robot.dof_names.index("waist_yaw_joint")] == 3.0
    assert [kd[robot.dof_names.index(n)] for n in ("left_knee_joint", "right_knee_joint")] == [7.0, 7.0]
    assert kp[robot.dof_names.index("left_knee_joint")] == 99.098427777  # other keys keep the preset's values
    assert kd[robot.dof_names.index("left_hip_yaw_joint")] == 2.557889765


def test_robot_joint_tag_dict_exposes_per_key_flags() -> None:
    cfg = tyro.cli(ExperimentConfig, args=("--robot.joint-tags.#legs", "['left_knee_joint']"), config=TYRO_CONIFG)
    assert cfg.robot.joint_tags["#legs"] == ["left_knee_joint"]
    assert cfg.robot.joint_tags["#knees"] == ["left_knee_joint", "right_knee_joint"]  # other keys keep the preset's


def test_robot_joint_tag_override_feeds_gain_coverage_at_cli() -> None:
    # The preset's #knees gain key expands through joint_tags at parse time: dropping a knee from the tag leaves
    # that DOF without gains under P control. Tyro may propagate a root dataclass's validation error
    # directly, or report it to stderr before exiting.
    err = io.StringIO()
    with pytest.raises((SystemExit, ValidationError)) as exc, redirect_stdout(io.StringIO()), redirect_stderr(err):
        tyro.cli(
            ExperimentConfig,
            args=("action:g1-29dof-joint-pos", "--robot.joint-tags.#knees", "['left_knee_joint']"),
            config=TYRO_CONIFG,
        )
    assert "joint controller stiffness has no key covering DOF(s) ['right_knee_joint']" in " ".join(
        (err.getvalue() + str(exc.value)).split()
    )


def test_robot_tag_with_unknown_body_rejected_at_cli() -> None:
    err = io.StringIO()
    with pytest.raises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(err):
        tyro.cli(ExperimentConfig, args=("--robot.body-tags.#feet", "['nope']"), config=TYRO_CONIFG)
    # The exit must come from the RobotConfig validator (its message reaches stderr), not from a CLI usage
    # error, which is also a SystemExit. tyro wraps its output, so compare with whitespace collapsed.
    assert "Unknown body name 'nope'" in " ".join(err.getvalue().split())


def test_robot_tag_dict_rejects_new_key_at_cli() -> None:
    # tyro exposes one flag per key already in the preset's dict; a new tag cannot be added from the CLI and has
    # to be declared in the preset (docs, 'Adding a robot'). The exit is a CLI usage error, not a validator one.
    err = io.StringIO()
    with pytest.raises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(err):
        tyro.cli(ExperimentConfig, args=("--robot.body-tags.#wings", "['pelvis']"), config=TYRO_CONIFG)
    text = " ".join(err.getvalue().split())
    assert "Unrecognized" in text and "--robot.body-tags.#wings" in text
    assert "Unknown body" not in text
