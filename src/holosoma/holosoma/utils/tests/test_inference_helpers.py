"""Pure tests for policy-export metadata accessors."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.inference_helpers import get_action_scales_from_env

pytestmark = pytest.mark.no_sim


def test_action_scales_accessor_returns_the_active_term_tensor() -> None:
    scales = torch.tensor([0.25, 0.5])
    env = cast("BaseTask", SimpleNamespace(action_scales=scales))
    assert get_action_scales_from_env(env) is scales


def test_action_scales_accessor_rejects_an_env_without_a_joint_action_term() -> None:
    env = cast("BaseTask", SimpleNamespace())
    with pytest.raises(ValueError, match="active action term exposing action_scales"):
        get_action_scales_from_env(env)
