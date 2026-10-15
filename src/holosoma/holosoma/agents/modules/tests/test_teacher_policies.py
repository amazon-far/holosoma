"""Teachers rebuilt from their own checkpoints, and from saved construction configs."""

from __future__ import annotations

import copy
from typing import Any, Callable

import pytest

from holosoma.agents.modules.teacher_policies import (
    MLPTeacher,
    PPOTeacher,
    build_teacher,
    teacher_from_checkpoint,
)
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("normalize", [False, True])
def test_ppo_teacher_reproduces_the_actor_at_inference(
    make_ppo_teacher_checkpoint: Callable[..., Any], make_fitted_normalizer: Callable[..., Any], normalize: bool
) -> None:
    normalizer = make_fitted_normalizer(3) if normalize else None
    checkpoint, actor = make_ppo_teacher_checkpoint(3, 2, [4], normalizer=normalizer)

    teacher = teacher_from_checkpoint(checkpoint, num_obs=3, num_actions=2)

    obs = torch.randn(5, 3) * 3.0
    actor_obs = normalizer(obs, update=False) if normalizer is not None else obs
    assert isinstance(teacher, PPOTeacher)
    torch.testing.assert_close(teacher(obs), actor.act_inference({"actor_obs": actor_obs}), rtol=0.0, atol=0.0)


def test_ppo_teacher_rejects_weights_its_actor_does_not_have(make_ppo_teacher_checkpoint: Callable[..., Any]) -> None:
    checkpoint, _ = make_ppo_teacher_checkpoint(3, 2, [4])
    # E.g. an encoder the reconstructed actor would silently go without.
    checkpoint["actor_model_state_dict"]["actor_module.encoder.0.weight"] = torch.zeros(4, 3)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        teacher_from_checkpoint(checkpoint, num_obs=3, num_actions=2)


def test_checkpoints_of_other_formats_are_rejected() -> None:
    rsl_rl_style = {"model_state_dict": {"actor.0.weight": torch.zeros(4, 3)}}

    with pytest.raises(ValueError, match="Unsupported teacher checkpoint"):
        teacher_from_checkpoint(rsl_rl_style, num_obs=3, num_actions=2)


def test_teacher_width_must_match_teacher_obs(make_ppo_teacher_checkpoint: Callable[..., Any]) -> None:
    checkpoint, _ = make_ppo_teacher_checkpoint(3, 2, [4])
    checkpoint["obs_dim_dict"] = {"actor_obs": 4}
    checkpoint["history_length_dict"] = {"actor_obs": 1}

    with pytest.raises(ValueError, match="reads 4 observations"):
        teacher_from_checkpoint(checkpoint, num_obs=3, num_actions=2)


def test_multi_group_actors_need_recorded_observation_dims(make_ppo_teacher_checkpoint: Callable[..., Any]) -> None:
    checkpoint, _ = make_ppo_teacher_checkpoint(3, 2, [4])
    checkpoint["experiment_config"]["algo"]["config"]["module_dict"]["actor"]["input_dim"] = ["actor_obs", "terrain"]

    with pytest.raises(ValueError, match="split of teacher_obs is unknown"):
        teacher_from_checkpoint(checkpoint, num_obs=3, num_actions=2)


def test_recorded_observation_dims_build_the_exact_actor_input(make_ppo_teacher_checkpoint: Callable[..., Any]) -> None:
    checkpoint, actor = make_ppo_teacher_checkpoint(3, 2, [4])
    checkpoint["obs_dim_dict"] = {"actor_obs": 3, "critic_obs": 9}
    checkpoint["history_length_dict"] = {"actor_obs": 1, "critic_obs": 1}

    teacher = teacher_from_checkpoint(checkpoint, num_obs=3, num_actions=2)

    obs = torch.randn(2, 3)
    torch.testing.assert_close(teacher(obs), actor.act_inference({"actor_obs": obs}), rtol=0.0, atol=0.0)


def test_normalizer_presence_must_match_the_training_setting(
    make_ppo_teacher_checkpoint: Callable[..., Any], make_fitted_normalizer: Callable[..., Any]
) -> None:
    checkpoint, _ = make_ppo_teacher_checkpoint(3, 2, [4], normalizer=make_fitted_normalizer(3))
    checkpoint["experiment_config"]["algo"]["config"]["empirical_normalization"] = False

    with pytest.raises(ValueError, match="empirical_normalization"):
        teacher_from_checkpoint(checkpoint, num_obs=3, num_actions=2)


@pytest.mark.parametrize("kind", ["ppo", "mlp"])
def test_construction_config_rebuilds_the_same_teacher(
    make_ppo_teacher_checkpoint: Callable[..., Any], make_fitted_normalizer: Callable[..., Any], kind: str
) -> None:
    if kind == "ppo":
        checkpoint, _ = make_ppo_teacher_checkpoint(3, 2, [4], normalizer=make_fitted_normalizer(3))
        source: Any = teacher_from_checkpoint(checkpoint, num_obs=3, num_actions=2)
    else:
        source = MLPTeacher(num_obs=3, hidden_dims=[4], num_actions=2, activation="ELU")
    config = copy.deepcopy(source.construction_config())

    rebuilt = build_teacher(config)
    rebuilt.load_state_dict(source.state_dict(), strict=True)

    obs = torch.randn(4, 3)
    assert type(rebuilt) is type(source)
    torch.testing.assert_close(rebuilt(obs), source(obs), rtol=0.0, atol=0.0)


def test_unknown_teacher_types_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown teacher type"):
        build_teacher({"type": "transformer"})


def test_building_teachers_leaves_the_global_rng_untouched(make_ppo_teacher_checkpoint: Callable[..., Any]) -> None:
    checkpoint, _ = make_ppo_teacher_checkpoint(3, 2, [4])
    state = torch.get_rng_state()

    teacher = teacher_from_checkpoint(checkpoint, num_obs=3, num_actions=2)
    build_teacher(teacher.construction_config())

    assert torch.equal(torch.get_rng_state(), state)
