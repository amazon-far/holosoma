"""Shared GAE returns and advantage normalization."""

from __future__ import annotations

import pytest

from holosoma.agents.modules.advantages import compute_gae_returns, normalize_advantages
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def test_gae_returns_match_a_hand_computed_rollout() -> None:
    # Two steps, one env; the episode ends after the first step.
    values = torch.tensor([[[1.0]], [[2.0]]])
    rewards = torch.tensor([[[1.0]], [[0.5]]])
    dones = torch.tensor([[[1.0]], [[0.0]]])
    last_values = torch.tensor([[3.0]])
    gamma, lam = 0.9, 0.5

    returns = compute_gae_returns(last_values, values, dones, rewards, gamma, lam)

    # Step 1 bootstraps from last_values; step 0 is terminal, so it neither bootstraps nor
    # carries step 1's advantage.
    advantage_1 = 0.5 + gamma * 3.0 - 2.0
    advantage_0 = 1.0 - 1.0
    torch.testing.assert_close(returns, torch.tensor([[[advantage_0 + 1.0]], [[advantage_1 + 2.0]]]))


def test_single_process_normalization_standardizes() -> None:
    advantages = torch.tensor([1.0, 2.0, 3.0, 6.0])

    normalized = normalize_advantages(advantages)

    torch.testing.assert_close(normalized, (advantages - advantages.mean()) / (advantages.std() + 1e-8))


def test_multi_gpu_normalization_pools_statistics_across_equal_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    local = torch.tensor([1.0, 2.0, 3.0])
    remote = torch.tensor([10.0, 20.0, 30.0])

    def fake_all_reduce(stats: torch.Tensor, op: object) -> None:
        del op
        stats += torch.stack([remote.mean(), (remote**2).mean()])

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    normalized = normalize_advantages(local, world_size=2)

    combined = torch.cat((local, remote))
    expected = (local - combined.mean()) / torch.sqrt(combined.var(unbiased=False) + 1e-8)
    torch.testing.assert_close(normalized, expected)
