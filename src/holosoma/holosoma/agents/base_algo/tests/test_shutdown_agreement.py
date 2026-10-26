"""Training ranks agree on a task shutdown request before leaving their loops."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from holosoma.agents.base_algo.base_algo import BaseAlgo


def _algo(multi_gpu: bool) -> BaseAlgo:
    multi_gpu_cfg = {"global_rank": 1, "local_rank": 1, "world_size": 2} if multi_gpu else None
    return BaseAlgo(cast("Any", SimpleNamespace()), cast("Any", None), "cpu", multi_gpu_cfg)


def test_a_rank_stops_when_another_rank_requested_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    def all_reduce(tensor: torch.Tensor, op: Any) -> None:
        tensor.copy_(torch.maximum(tensor, torch.tensor(1.0)))  # the peer rank requested

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    assert _algo(multi_gpu=True)._any_rank_requested_shutdown(False)


def test_a_single_process_uses_its_own_request_without_a_collective(monkeypatch: pytest.MonkeyPatch) -> None:
    def all_reduce(tensor: torch.Tensor, op: Any) -> None:
        raise AssertionError("single-process training must not enter a collective")

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    algo = _algo(multi_gpu=False)
    assert algo._any_rank_requested_shutdown(True)
    assert not algo._any_rank_requested_shutdown(False)
