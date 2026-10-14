"""FastSAC's action-range adapter rejects unbounded joints before constructing the actor."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from holosoma.agents.fast_sac.fast_sac_agent import FastSACEnv
from holosoma.envs.base_task.base_task import BaseTask

pytestmark = pytest.mark.no_sim


def _env(limits: list[list[float]], *, action_scale: float = 0.5) -> BaseTask:
    robot = SimpleNamespace(
        default_dof_positions=lambda: {"shoulder": 0.0, "elbow": 1.0},
        dof_names=["shoulder", "elbow"],
    )
    return cast(
        "BaseTask",
        SimpleNamespace(
            device="cpu",
            dof_names=["shoulder", "elbow"],
            dof_pos_limits=torch.tensor(limits),
            robot_config=robot,
            action_scales=torch.full((2,), action_scale),
        ),
    )


def test_finite_position_limits_define_per_joint_action_boundaries() -> None:
    wrapped = FastSACEnv(_env([[-1.0, 2.0], [-3.0, 5.0]]), (), ())

    torch.testing.assert_close(wrapped._action_boundaries, torch.tensor([4.0, 8.0]))


@pytest.mark.parametrize(
    "limits",
    [
        [[float("-inf"), 2.0], [-3.0, 5.0]],
        [[-1.0, 2.0], [-3.0, float("inf")]],
        [[float("nan"), 2.0], [-3.0, 5.0]],
    ],
)
def test_unbounded_position_limit_names_the_joint(limits: list[list[float]]) -> None:
    with pytest.raises(ValueError, match=r"unbounded for \['(shoulder|elbow)'\]"):
        FastSACEnv(_env(limits), (), ())


@pytest.mark.parametrize("action_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_action_scale_must_be_finite_and_positive(action_scale: float) -> None:
    with pytest.raises(ValueError, match="finite positive action scales"):
        FastSACEnv(_env([[-1.0, 2.0], [-3.0, 5.0]], action_scale=action_scale), (), ())


def test_missing_action_scales_has_a_clear_error() -> None:
    env = _env([[-1.0, 2.0], [-3.0, 5.0]])
    delattr(env, "action_scales")
    with pytest.raises(ValueError, match="active action term exposing action_scales"):
        FastSACEnv(env, (), ())
