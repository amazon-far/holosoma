from __future__ import annotations

import torch

from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization


def test_distributed_variance_roundoff_is_clamped(monkeypatch) -> None:
    normalizer = EmpiricalNormalization(shape=1, device="cpu")
    values = torch.ones((2, 1))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(stats: torch.Tensor, **_) -> None:
        epsilon = torch.finfo(stats.dtype).eps
        stats.copy_(stats.new_tensor([[4.0], [4.0 - epsilon]]))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    normalizer.update(values)

    assert torch.isfinite(normalizer._std).all()
    assert torch.all(normalizer._var >= 0.0)
