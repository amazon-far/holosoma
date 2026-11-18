"""Torque-saturation logging in ``JointPositionActionTerm`` (no sim)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.config_types.action import ActionTermCfg
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm

pytestmark = pytest.mark.no_sim

DOF_NAMES = ["hip_joint", "knee_joint", "ankle_joint"]
LIMITS = torch.tensor([10.0, 20.0, 5.0])
NUM_ENVS = 4
DECIMATION = 2


class _Simulator:
    def __init__(self) -> None:
        self.dof_pos = torch.zeros(NUM_ENVS, len(DOF_NAMES))
        self.dof_vel = torch.zeros(NUM_ENVS, len(DOF_NAMES))
        self.simulator_config = SimpleNamespace(sim=SimpleNamespace(control_decimation_steps=DECIMATION))
        self.applied: list[torch.Tensor] = []

    def apply_torques_at_dof(self, torques: torch.Tensor) -> None:
        self.applied.append(torques.clone())


def _term(*, per_joint: bool = False) -> JointPositionActionTerm:
    """Torque-controlled term (action == requested torque) built through the real setup()."""
    num_dof = len(DOF_NAMES)
    term = JointPositionActionTerm.__new__(JointPositionActionTerm)
    term.cfg = ActionTermCfg(
        func="holosoma.managers.action.terms:JointPositionActionTerm",
        params={"log_torque_sat_per_joint": per_joint},
    )
    term.env = SimpleNamespace(
        num_envs=NUM_ENVS,
        device="cpu",
        dof_names=DOF_NAMES,
        torque_limits=LIMITS,
        log_dict={},
        _pending_torque_rfi=(False, 0.0),
        simulator=_Simulator(),
        robot_config=SimpleNamespace(control=SimpleNamespace(control_type="T", clip_torques=True, clip_actions=False)),
    )
    term._action_dim = num_dof
    term.action_scales = torch.ones(num_dof)
    term._raw_actions = torch.zeros(NUM_ENVS, num_dof)
    term._processed_actions = torch.zeros(NUM_ENVS, num_dof)
    term._actions_after_delay = torch.zeros(NUM_ENVS, num_dof)
    term.torques = torch.zeros(NUM_ENVS, num_dof)
    term._prev_dof_vel = torch.zeros(NUM_ENVS, num_dof)
    term._substep_idx = 0
    term.setup()
    return term


def _step(term: JointPositionActionTerm, actions: torch.Tensor, substeps: int = DECIMATION) -> None:
    term.process_actions(actions)
    for _ in range(substeps):
        term.apply_actions()


# Envs 0-1 hit the hip limit (10), every env hits the ankle limit (5) on the negative side, and the knee
# request (19) stays below its limit (20).
SATURATING = torch.tensor([[12.0, 1.0, -6.0], [10.0, 1.0, -5.0], [1.0, 19.0, -9.0], [1.0, 19.0, -5.0]])


def test_step_logs_overall_and_worst_joint_fractions() -> None:
    term = _term()
    _step(term, SATURATING)

    log = term.env.log_dict
    assert float(log["torque_sat_frac"]) == pytest.approx(0.5)  # (0.5 + 0 + 1) / 3
    assert float(log["torque_sat_max"]) == pytest.approx(1.0)
    assert not any(key.startswith(("torque_sat/", "torque_util_p99/")) for key in log)


def test_clean_step_after_saturated_step_logs_zero() -> None:
    term = _term()
    _step(term, SATURATING)
    first = term.env.log_dict["torque_sat_frac"]
    _step(term, torch.zeros(NUM_ENVS, len(DOF_NAMES)))
    assert float(first) == pytest.approx(0.5)  # earlier step's logged value is not overwritten
    assert float(term.env.log_dict["torque_sat_frac"]) == pytest.approx(0.0)
    assert float(term.env.log_dict["torque_sat_max"]) == pytest.approx(0.0)


def test_per_joint_keys_behind_flag() -> None:
    term = _term(per_joint=True)
    _step(term, SATURATING)

    log = term.env.log_dict
    assert float(log["torque_sat/hip"]) == pytest.approx(0.5)
    assert float(log["torque_sat/knee"]) == pytest.approx(0.0)
    assert float(log["torque_sat/ankle"]) == pytest.approx(1.0)
    # p99 of |tau| / limit over envs x substeps.
    assert float(log["torque_util_p99/hip"]) == pytest.approx(1.0)
    assert float(log["torque_util_p99/knee"]) == pytest.approx(0.95)
    assert float(log["torque_util_p99/ankle"]) == pytest.approx(1.0)
