from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.randomization.terms.locomotion import (
    randomize_action_delay,
    setup_action_delay_buffers,
)
from holosoma.utils.sampler import STAGE_RESET, STAGE_SETUP, TermSampler

pytestmark = pytest.mark.no_sim


def test_action_delay_requires_setup_when_enablement_is_not_supplied() -> None:
    env = SimpleNamespace(num_envs=3, device="cpu")
    sampler = TermSampler.bind(base_seed=42, term_name="action_delay", stage=STAGE_RESET)
    with pytest.raises(AttributeError, match="_randomize_ctrl_delay"):
        randomize_action_delay(env, [1], sampler=sampler)


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_action_delay_setup_controls_reset_behavior(enabled: bool) -> None:
    env = SimpleNamespace(num_envs=3, device="cpu")
    setup_sampler = TermSampler.bind(base_seed=42, term_name="action_delay", stage=STAGE_SETUP)
    setup_action_delay_buffers(env, setup_sampler, ctrl_delay_step_range=[0, 0], enabled=enabled)
    if not enabled:
        randomize_action_delay(env, [1], sampler=setup_sampler)
        assert not hasattr(env, "action_delay_idx")
        return

    reset_sampler = TermSampler.bind(base_seed=42, term_name="action_delay", stage=STAGE_RESET)
    randomize_action_delay(env, [1], sampler=reset_sampler, ctrl_delay_step_range=[2, 2])
    torch.testing.assert_close(env.action_delay_idx, torch.tensor([0, 2, 0]))
