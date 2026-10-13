"""``JointPositionActionTerm`` resolves its controller against the selected robot (no simulator).

Built on a stub env carrying the attributes the constructor reads; gain tensors must follow the
simulator's DOF order, on the env device, as float32.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from holosoma.config_types.action import ActionTermCfg, JointControllerCfg
from holosoma.config_types.robot import RobotConfig
from holosoma.config_values.loco.g1.action import g1_29dof_joint_pos
from holosoma.config_values.loco.t1.action import t1_29dof_joint_pos
from holosoma.config_values.robot import g1_29dof, t1_29dof_waist_wrist
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm
from tests.simulators._robot_presets import TEST_ARM_CONTROLLER, test_arm_action, test_arm_fixed

pytestmark = pytest.mark.no_sim

NUM_ENVS = 2


def _cfg(controller: JointControllerCfg, *, scale: float = 0.25) -> ActionTermCfg:
    return ActionTermCfg(
        func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
        scale=scale,
        controller=controller,
    )


G1_CONTROLLER = cast("JointControllerCfg", g1_29dof_joint_pos.terms["joint_control"].controller)
T1_CONTROLLER = cast("JointControllerCfg", t1_29dof_joint_pos.terms["joint_control"].controller)


def _env(robot: RobotConfig, dof_names: list[str] | None = None, torque_limits: torch.Tensor | None = None) -> Any:
    names = list(robot.dof_names) if dof_names is None else dof_names
    return SimpleNamespace(
        num_dof=len(names),
        num_envs=NUM_ENVS,
        device="cpu",
        dof_names=names,
        robot_config=robot,
        torque_limits=torch.ones(len(names)) if torque_limits is None else torque_limits,
    )


@pytest.mark.parametrize(
    ("robot", "controller"),
    [
        (g1_29dof, G1_CONTROLLER),
        (t1_29dof_waist_wrist, T1_CONTROLLER),
        (test_arm_fixed, TEST_ARM_CONTROLLER),
    ],
    ids=["g1_29dof", "t1_29dof_waist_wrist", "test_arm_fixed"],
)
def test_gain_tensors_equal_controller_gains_in_dof_order(robot: RobotConfig, controller: JointControllerCfg) -> None:
    env = _env(robot)
    kp, kd = controller.resolve_gains(robot)

    term = JointPositionActionTerm(_cfg(controller), env)

    assert term.p_gains.shape == term.d_gains.shape == (len(robot.dof_names),)
    assert term.p_gains.dtype == term.d_gains.dtype == torch.float32
    assert term.p_gains.device.type == "cpu"
    assert term.p_gains.tolist() == pytest.approx(kp)
    assert term.d_gains.tolist() == pytest.approx(kd)
    assert torch.equal(term.i_gains, torch.zeros(len(robot.dof_names)))
    # The env aliases the term's tensors (randomization terms scale them in place).
    assert env.p_gains is term.p_gains and env.d_gains is term.d_gains


def test_tag_keys_reach_every_member_dof() -> None:
    # g1 keys its knees by tag: both knee DOFs read the '#knees' value, in whatever positions they sit.
    term = JointPositionActionTerm(_cfg(G1_CONTROLLER), _env(g1_29dof))
    knees = [g1_29dof.dof_names.index(n) for n in ("left_knee_joint", "right_knee_joint")]
    assert term.p_gains[knees].tolist() == pytest.approx([G1_CONTROLLER.stiffness["#knees"]] * 2)
    assert term.d_gains[knees].tolist() == pytest.approx([G1_CONTROLLER.damping["#knees"]] * 2)


def test_gains_follow_the_simulator_dof_order() -> None:
    # The term indexes by env.dof_names (the simulator's order); a reordered simulator list gets reordered gains.
    reversed_names = list(reversed(g1_29dof.dof_names))
    kp, kd = G1_CONTROLLER.resolve_gains(g1_29dof)

    term = JointPositionActionTerm(_cfg(G1_CONTROLLER), _env(g1_29dof, reversed_names))

    assert term.p_gains.tolist() == pytest.approx(list(reversed(kp)))
    assert term.d_gains.tolist() == pytest.approx(list(reversed(kd)))


def test_action_scales_come_from_the_action_term() -> None:
    cfg = test_arm_action.terms["joint_control"]
    assert cfg.controller is not None
    assert cfg.controller.action_scales_by_effort_limit_over_p_gain is False
    term = JointPositionActionTerm(cfg, _env(test_arm_fixed))
    assert torch.equal(term.action_scales, torch.full((3,), 0.25))


@pytest.mark.parametrize("invalid", [float("inf"), float("nan")], ids=["infinite", "nan"])
def test_effort_derived_action_scales_reject_nonfinite_effort(invalid: float) -> None:
    controller = dataclasses.replace(
        TEST_ARM_CONTROLLER,
        action_scales_by_effort_limit_over_p_gain=True,
    )
    env = _env(test_arm_fixed, torque_limits=torch.tensor([20.0, invalid, 50.0]))

    with pytest.raises(ValueError, match=r"unbounded DOFs: \['elbow_joint'\]"):
        JointPositionActionTerm(_cfg(controller), env)


@pytest.mark.parametrize("invalid", [float("inf"), float("nan")], ids=["infinite", "nan"])
def test_torque_rfi_rejects_nonfinite_effort(invalid: float) -> None:
    env = _env(test_arm_fixed, torque_limits=torch.tensor([20.0, invalid, 50.0]))
    term = JointPositionActionTerm(_cfg(TEST_ARM_CONTROLLER), env)

    with pytest.raises(ValueError, match=r"torque RFI.*unbounded DOFs: \['elbow_joint'\]"):
        term.configure_torque_rfi(enabled=True, rfi_lim=0.1)


def test_simulator_dof_missing_from_the_config_raises() -> None:
    with pytest.raises(ValueError, match="Simulator DOF 'ghost_joint' is not in robot_config.dof_names"):
        JointPositionActionTerm(
            _cfg(TEST_ARM_CONTROLLER),
            _env(test_arm_fixed, [*test_arm_fixed.dof_names, "ghost_joint"]),
        )
