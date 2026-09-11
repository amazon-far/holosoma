"""Tests for the FastSAC actor's action sampling (no sim).

``Actor.explore`` derives its deterministic output from ``Actor.forward``, which is also what the
inference policy and the ONNX export emit. These tests pin that equivalence across both the
tanh-squashed and unsquashed configurations, since a change to ``forward`` would otherwise silently
desynchronize training-time and deployment-time actions.
"""

from __future__ import annotations

import pytest
import torch

from holosoma.agents.fast_sac.fast_sac import Actor

pytestmark = pytest.mark.no_sim

_N_OBS = 6
_N_ACT = 4


def _make_actor(use_tanh: bool) -> Actor:
    torch.manual_seed(0)
    return Actor(
        obs_indices={"actor_obs": {"size": _N_OBS, "start": 0, "end": _N_OBS}},
        obs_keys=["actor_obs"],
        n_act=_N_ACT,
        num_envs=5,
        hidden_dim=64,
        log_std_max=2.0,
        log_std_min=-5.0,
        use_tanh=use_tanh,
        action_scale=torch.linspace(0.5, 2.0, _N_ACT),
    )


@pytest.mark.parametrize("use_tanh", [True, False])
def test_returned_mean_matches_the_deterministic_action(use_tanh):
    actor = _make_actor(use_tanh)
    obs = torch.randn(5, _N_OBS)

    _, mean_action = actor.explore(obs, return_mean=True)

    torch.testing.assert_close(mean_action, actor(obs)[0])
    torch.testing.assert_close(mean_action, actor.explore(obs, deterministic=True))


@pytest.mark.parametrize("use_tanh", [True, False])
def test_sample_differs_from_the_mean(use_tanh):
    actor = _make_actor(use_tanh)
    obs = torch.randn(5, _N_OBS)

    action, mean_action = actor.explore(obs, return_mean=True)

    assert not torch.allclose(action, mean_action)


@pytest.mark.parametrize("use_tanh", [True, False])
def test_bare_call_returns_a_single_tensor(use_tanh):
    actor = _make_actor(use_tanh)

    action = actor.explore(torch.randn(5, _N_OBS))

    assert isinstance(action, torch.Tensor)
    assert action.shape == (5, _N_ACT)


def test_unsquashed_deterministic_action_is_the_raw_gaussian_loc():
    """Without tanh there is no squash or scale, so the mean action is the network output itself."""
    actor = _make_actor(use_tanh=False)
    obs = torch.randn(5, _N_OBS)

    mean_action, loc, _ = actor(obs)

    torch.testing.assert_close(mean_action, loc)


def test_squashed_deterministic_action_is_scaled_within_bounds():
    actor = _make_actor(use_tanh=True)
    obs = torch.randn(5, _N_OBS) * 5.0

    mean_action, loc, _ = actor(obs)

    torch.testing.assert_close(mean_action, torch.tanh(loc) * actor.action_scale + actor.action_bias)
    assert torch.all(mean_action.abs() <= actor.action_scale + actor.action_bias.abs())
