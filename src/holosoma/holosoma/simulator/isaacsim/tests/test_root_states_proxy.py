"""RootStatesProxy re-reads backend root states only when the backend state version advances."""

from __future__ import annotations

import pytest
import torch

from holosoma.simulator.isaacsim.proxy_utils import RootStatesProxy

pytestmark = pytest.mark.no_sim


class _Backend:
    """Stand-in for the articulation buffers: a wxyz root-state tensor and a state version."""

    def __init__(self, num_envs: int = 2) -> None:
        self.version = 0
        self.reads = 0
        self.root_state_w = torch.zeros(num_envs, 13)
        self.root_state_w[:, 3] = 1.0  # identity quaternion, wxyz

    def latest(self) -> torch.Tensor:
        self.reads += 1
        return self.root_state_w

    def step(self, x: float) -> None:
        """Advance the simulation: new buffer contents and a new version."""
        self.root_state_w = self.root_state_w.clone()
        self.root_state_w[:, 0] = x
        self.version += 1


def _proxy(backend: _Backend) -> RootStatesProxy:
    return RootStatesProxy(backend.root_state_w, version=lambda: backend.version, latest=backend.latest)


def test_reads_are_cached_until_the_version_advances() -> None:
    backend = _Backend()
    proxy = _proxy(backend)

    for _ in range(3):
        assert proxy[:, 0].tolist() == [0.0, 0.0]
    assert backend.reads == 0

    backend.step(1.5)
    assert proxy[:, 0].tolist() == [1.5, 1.5]
    assert proxy[:, 0].tolist() == [1.5, 1.5]
    assert backend.reads == 1


def test_writes_stay_authoritative_until_the_next_step() -> None:
    backend = _Backend()
    proxy = _proxy(backend)
    written = proxy[0].clone()
    written[0] = 7.0

    proxy[0] = written

    assert proxy[0, 0].item() == 7.0
    assert proxy._get_wxyz(torch.tensor([0]))[0, 0].item() == 7.0  # what a flush writes
    backend.step(2.0)
    assert proxy[0, 0].item() == 2.0


def test_flush_after_a_step_writes_current_state_not_the_pre_step_cache() -> None:
    backend = _Backend()
    proxy = _proxy(backend)
    proxy[:, 0]  # cache at version 0

    backend.step(3.0)

    assert proxy._get_wxyz()[:, 0].tolist() == [3.0, 3.0]


def test_invalidate_rereads_after_a_write_that_bypassed_the_proxy() -> None:
    backend = _Backend()
    proxy = _proxy(backend)
    proxy[:, 0]
    backend.root_state_w = backend.root_state_w.clone()
    backend.root_state_w[1, 0] = 4.0  # direct backend write, version unchanged

    assert proxy[1, 0].item() == 0.0
    proxy.invalidate()
    assert proxy[1, 0].item() == 4.0
