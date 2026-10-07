"""Construction-time guards for the implicit (simulator-native PD) joint-position action term.

``JointPositionTargetActionTerm`` hands position targets to ``simulator.apply_position_targets_at_dof``,
which only IsaacSim implements, and those targets only move the joints when IsaacSim built implicit
actuators for them (``control_mode="implicit_position_target"``). Each mismatch used to surface as an
``AttributeError`` on the first ``env.step()`` or as silently ignored targets.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from holosoma.config_types.action import ActionTermCfg
from holosoma.config_values.robot import g1_29dof
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm, JointPositionTargetActionTerm

pytestmark = pytest.mark.no_sim

_TERM_CFG = ActionTermCfg(func="holosoma.managers.action.terms.joint_control:JointPositionTargetActionTerm")


class _TorqueOnlySimulator:
    """Like MuJoCo and IsaacGym: explicit torques only."""

    def apply_torques_at_dof(self, torques: torch.Tensor, dof_indices: list[int] | None = None) -> None:
        raise AssertionError("not reached")


class _PositionTargetSimulator(_TorqueOnlySimulator):
    """Like IsaacSim: also accepts simulator-native position targets."""

    def __init__(self) -> None:
        self.targets: list[torch.Tensor] = []

    def apply_position_targets_at_dof(self, targets: torch.Tensor) -> None:
        self.targets.append(targets.clone())


def _env(simulator: Any, control_mode: str) -> SimpleNamespace:
    robot_config = dataclasses.replace(
        g1_29dof, control=dataclasses.replace(g1_29dof.control, control_mode=control_mode)
    )
    num_dof = len(robot_config.dof_names)
    return SimpleNamespace(
        robot_config=robot_config,
        simulator=simulator,
        num_envs=2,
        num_dof=num_dof,
        device="cpu",
        default_dof_pos=torch.linspace(-0.5, 0.5, num_dof).repeat(2, 1),
        log_dict={},
    )


def test_target_term_rejects_a_simulator_without_position_targets() -> None:
    with pytest.raises(NotImplementedError, match="_TorqueOnlySimulator does not implement"):
        JointPositionTargetActionTerm(_TERM_CFG, _env(_TorqueOnlySimulator(), "implicit_position_target"))


def test_target_term_rejects_explicit_control_mode() -> None:
    """With explicit actuators IsaacSim ignores position targets, so the robot would go limp."""
    with pytest.raises(ValueError, match="control_mode='implicit_position_target'"):
        JointPositionTargetActionTerm(_TERM_CFG, _env(_PositionTargetSimulator(), "explicit_pd_torque"))


def test_target_term_sends_scaled_targets_around_the_default_pose() -> None:
    simulator = _PositionTargetSimulator()
    env = _env(simulator, "implicit_position_target")
    term = JointPositionTargetActionTerm(_TERM_CFG, env)

    actions = torch.full((env.num_envs, env.num_dof), 0.5)
    term.process_actions(actions)
    term.apply_actions()

    (targets,) = simulator.targets
    assert torch.allclose(targets, env.default_dof_pos + actions * term.action_scales)
    assert env.action_scales is term.action_scales


def test_explicit_term_rejects_implicit_control_mode() -> None:
    """IsaacSim's implicit drive would add its own PD on top of the torques this term computes."""
    with pytest.raises(ValueError, match="use JointPositionTargetActionTerm"):
        JointPositionActionTerm(_TERM_CFG, _env(_PositionTargetSimulator(), "implicit_position_target"))
