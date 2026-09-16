from __future__ import annotations

import torch

from holosoma.agents.ppo.ppo import PPO, EmpiricalNormalization


def _population_moments(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return values.mean(dim=0, keepdim=True), values.var(dim=0, keepdim=True, unbiased=False)


def _packed_moments(normalizer: EmpiricalNormalization) -> torch.Tensor:
    second_moment = normalizer._var + normalizer._mean.square()
    return torch.cat([normalizer._mean.reshape(-1), second_moment.reshape(-1)])


def test_empirical_normalization_merges_local_batches_exactly() -> None:
    normalizer = EmpiricalNormalization(shape=2, device="cpu")
    batches = [
        torch.tensor([[0.0, -2.0], [2.0, 0.0]]),
        torch.tensor([[10.0, 4.0], [12.0, 6.0]]),
    ]

    for batch in batches:
        normalizer.update(batch)

    expected_mean, expected_var = _population_moments(torch.cat(batches))
    torch.testing.assert_close(normalizer._mean, expected_mean)
    torch.testing.assert_close(normalizer._var, expected_var)
    assert normalizer.count.item() == 4


def test_deferred_sync_preserves_between_rank_variance(monkeypatch) -> None:
    local = EmpiricalNormalization(shape=1, device="cpu")
    remote = EmpiricalNormalization(shape=1, device="cpu")
    remote_moments: list[torch.Tensor] = []
    collective_calls = 0

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(packed: torch.Tensor, **_) -> None:
        nonlocal collective_calls
        collective_calls += 1
        packed.add_(remote_moments.pop(0))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    first_local = torch.tensor([[0.0], [2.0]])
    first_remote = torch.tensor([[10.0], [12.0]])
    local.update(first_local)
    remote.update(first_remote)
    remote_moments.append(_packed_moments(remote))
    local.flush_deferred_sync()

    all_values = [first_local, first_remote]
    expected_mean, expected_var = _population_moments(torch.cat(all_values))
    torch.testing.assert_close(local._mean, expected_mean)
    torch.testing.assert_close(local._var, expected_var)

    remote.load_state_dict(local.state_dict())
    second_local = torch.tensor([[4.0], [8.0]])
    second_remote = torch.tensor([[14.0], [18.0]])
    local.update(second_local)
    remote.update(second_remote)
    remote_moments.append(_packed_moments(remote))
    local.flush_deferred_sync()

    all_values.extend([second_local, second_remote])
    expected_mean, expected_var = _population_moments(torch.cat(all_values))
    torch.testing.assert_close(local._mean, expected_mean)
    torch.testing.assert_close(local._var, expected_var)
    torch.testing.assert_close(local._std, expected_var.sqrt())
    assert local.count.item() == 4
    assert collective_calls == 2
    assert not remote_moments


def test_multi_gpu_advantage_variance_roundoff_is_clamped(monkeypatch) -> None:
    advantages = torch.ones((2, 1))

    def fake_all_reduce(stats: torch.Tensor, **_) -> None:
        epsilon = torch.finfo(stats.dtype).eps
        stats.copy_(stats.new_tensor([2.0, 2.0 - epsilon]))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    algorithm = PPO.__new__(PPO)
    algorithm.gpu_world_size = 2

    normalized = PPO._normalize_advantages_multi_gpu(algorithm, advantages)

    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalized, torch.zeros_like(normalized))
