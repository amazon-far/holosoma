"""Unit tests for CameraRuntime's cross-device buffer cache (pure, no simulator, CPU-only).

``set_buffer`` is the single write path for a rendered frame and co-locates cache invalidation with
the mutation; ``buffer_on`` serves cross-device copies, caching the first transfer per
``(data_type, device)`` so several consumers reading the same frame on host cost one copy. These pin
the caching + invalidation contract without a GPU by counting ``.to`` calls on a wrapper tensor.
"""

from __future__ import annotations

import pytest

from holosoma.simulator.shared.camera_sensor import CameraRuntime
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


class _CountingTensor(torch.Tensor):
    """A Tensor that counts ``.to(device)`` calls, to assert cross-device copies happen once.

    ``.to("cpu")`` on a cpu tensor is normally a no-op that returns self; here we force it to mint a
    fresh (plain) tensor and bump a counter so a test in a CPU-only env can still prove the cache
    collapses N reads into one copy. The returned copy is a plain Tensor, so it is not itself counted.
    """

    to_calls: int

    @staticmethod
    def wrap(data: torch.Tensor) -> _CountingTensor:
        t = data.as_subclass(_CountingTensor)
        t.to_calls = 0
        return t

    def to(self, *args, **kwargs):  # type: ignore[override]
        self.to_calls += 1
        # Return a distinct plain tensor so identity checks distinguish "the copy" from "the buffer".
        return torch.Tensor.clone(torch.Tensor.as_subclass(self, torch.Tensor))


def _rt(buf: torch.Tensor) -> CameraRuntime:
    rt = CameraRuntime(name="c", config=None)  # type: ignore[arg-type]  # test stub: buffer-only runtime, no real config
    rt.set_buffer("rgb", buf)
    return rt


def test_device_none_returns_buffer_without_copy():
    buf = _CountingTensor.wrap(torch.zeros(1, 2, 2, 3, dtype=torch.uint8))
    rt = _rt(buf)
    assert rt.buffer_on("rgb", None) is buf
    assert buf.to_calls == 0
    assert rt._device_cache == {}


def test_cross_device_copy_is_cached_across_reads():
    # Force a "different device" path via a fake device str so .to() actually fires in a CPU env.
    buf = _CountingTensor.wrap(torch.zeros(1, 2, 2, 3, dtype=torch.uint8))
    rt = CameraRuntime(name="c", config=None)  # type: ignore[arg-type]  # test stub: buffer-only runtime, no real config
    rt.set_buffer("rgb", buf)
    a = rt.buffer_on("rgb", "meta")  # a non-cpu device string -> triggers .to
    b = rt.buffer_on("rgb", "meta")  # second read reuses the cache
    assert buf.to_calls == 1  # only the first read paid the copy
    assert a is b  # same cached object handed to both consumers


def test_set_buffer_invalidates_stale_cross_device_copies():
    buf1 = _CountingTensor.wrap(torch.zeros(1, 2, 2, 3, dtype=torch.uint8))
    rt = CameraRuntime(name="c", config=None)  # type: ignore[arg-type]  # test stub: buffer-only runtime, no real config
    rt.set_buffer("rgb", buf1)
    first = rt.buffer_on("rgb", "meta")
    assert buf1.to_calls == 1

    # A fresh render replaces the buffer -> the old host copy must be dropped, next read re-copies.
    buf2 = _CountingTensor.wrap(torch.ones(1, 2, 2, 3, dtype=torch.uint8))
    rt.set_buffer("rgb", buf2)
    assert rt._device_cache == {}  # invalidated at the write site
    second = rt.buffer_on("rgb", "meta")
    assert buf2.to_calls == 1  # copied from the NEW buffer
    assert second is not first


def test_set_buffer_only_invalidates_its_own_modality():
    # A depth render must not evict a cached rgb copy (different data_type keys).
    rgb = _CountingTensor.wrap(torch.zeros(1, 2, 2, 3, dtype=torch.uint8))
    depth = _CountingTensor.wrap(torch.zeros(1, 2, 2, 1, dtype=torch.float32))
    rt = CameraRuntime(name="c", config=None)  # type: ignore[arg-type]  # test stub: buffer-only runtime, no real config
    rt.set_buffer("rgb", rgb)
    rt.set_buffer("depth", depth)
    rt.buffer_on("rgb", "meta")
    rt.buffer_on("depth", "meta")
    assert set(rt._device_cache) == {("rgb", "meta"), ("depth", "meta")}

    # Re-render ONLY depth: rgb's cached copy survives, depth's is dropped.
    rt.set_buffer("depth", _CountingTensor.wrap(torch.ones(1, 2, 2, 1, dtype=torch.float32)))
    assert set(rt._device_cache) == {("rgb", "meta")}
