"""Exercise the function-based delay terms through action processing and simulator torque writes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from holosoma.config_types.action import ActionTermCfg, JointControllerCfg
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm
from holosoma.managers.randomization.terms.locomotion import randomize_action_delay, setup_action_delay_buffers
from holosoma.utils.sampler import STAGE_RESET, STAGE_SETUP, TermSampler
from tests.simulators._robot_presets import test_arm_fixed

pytestmark = pytest.mark.no_sim


def _environment() -> tuple[SimpleNamespace, list[torch.Tensor]]:
    writes = []
    sim = SimpleNamespace(
        dof_pos=torch.zeros(2, 3),
        dof_vel=torch.zeros(2, 3),
        simulator_config=SimpleNamespace(sim=SimpleNamespace(control_decimation_steps=1)),
        apply_torques_at_dof=lambda values: writes.append(values.clone()),
    )
    env = SimpleNamespace(
        num_envs=2,
        num_dof=3,
        dof_names=test_arm_fixed.dof_names,
        device="cpu",
        robot_config=test_arm_fixed,
        torque_limits=torch.full((3,), 100.0),
        simulator=sim,
        log_dict={},
        _pending_torque_rfi=(False, 0.0),
    )
    return env, writes


def _action(env: Any) -> JointPositionActionTerm:
    term = JointPositionActionTerm(
        ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            scale=1.0,
            controller=JointControllerCfg(control_type="T", clip_torques=False),
        ),
        env,
    )
    term.setup()
    return term


def _step(term: JointPositionActionTerm, value: float) -> None:
    term.process_actions(torch.full((2, 3), float(value)))
    term.apply_actions()


def test_absent_delay_configuration_is_disabled_without_a_manager_dependency() -> None:
    env, writes = _environment()
    term = _action(env)
    randomize_action_delay(env, None, sampler=TermSampler.bind(1, "randomize_action_delay", STAGE_RESET))
    _step(term, 3)
    torch.testing.assert_close(writes[-1], torch.full((2, 3), 3.0))
    assert term.action_queue is None
    assert not hasattr(env, "_randomize_ctrl_delay")
    term.reset()


def test_enabled_delay_delivers_the_previous_action() -> None:
    env, writes = _environment()
    setup_action_delay_buffers(
        env, TermSampler.bind(1, "setup_action_delay_buffers", STAGE_SETUP), ctrl_delay_step_range=[1, 1]
    )
    term = _action(env)
    _step(term, 3)
    _step(term, 7)
    torch.testing.assert_close(writes[0], torch.zeros(2, 3))
    torch.testing.assert_close(writes[1], torch.full((2, 3), 3.0))


@pytest.mark.parametrize("all_envs", [False, True])
def test_reset_clears_existing_queue_even_after_delay_is_disabled(all_envs: bool) -> None:
    env, writes = _environment()
    setup_action_delay_buffers(
        env, TermSampler.bind(1, "setup_action_delay_buffers", STAGE_SETUP), ctrl_delay_step_range=[1, 1]
    )
    term = _action(env)
    _step(term, 3)
    randomize_action_delay(env, None, sampler=TermSampler.bind(1, "randomize_action_delay", STAGE_RESET), enabled=False)
    term.reset(None if all_envs else torch.tensor([0]))
    randomize_action_delay(env, None, sampler=TermSampler.bind(1, "randomize_action_delay", STAGE_RESET), enabled=True)
    _step(term, 7)
    expected = torch.zeros(2, 3)
    if not all_envs:
        expected[1] = 3.0
    torch.testing.assert_close(writes[-1], expected)


def test_delay_functions_preserve_keyed_sampler_draws_and_reset_subset() -> None:
    env, _ = _environment()
    setup = TermSampler.bind(42, "setup_action_delay_buffers", STAGE_SETUP)
    setup_action_delay_buffers(env, setup, ctrl_delay_step_range=[0, 4])
    expected = setup.draw_int(0, 4, env_ids=torch.arange(2))
    torch.testing.assert_close(env.action_delay_idx, expected)
    before = env.action_delay_idx.clone()
    reset = TermSampler.bind(42, "randomize_action_delay", STAGE_RESET)
    randomize_action_delay(env, torch.tensor([1]), sampler=reset)
    assert env.action_delay_idx[0] == before[0]
    torch.testing.assert_close(env.action_delay_idx[1:], reset.draw_int(0, 4, env_ids=torch.tensor([1])))
