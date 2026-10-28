"""Tests for ``BatchedSdkCalls`` — pure, no bridge, no child process, no DDS."""

from __future__ import annotations

from typing import Any

import pytest

from holosoma.bridge.unitree.batched_calls import BatchedSdkCalls

pytestmark = pytest.mark.no_sim

_BATCH = "__batch__"


class _Transport:
    """Stands in for the bridge's ``_call``, recording what crossed and replying per method."""

    def __init__(self, replies: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._replies = replies or {}

    def __call__(self, method: str, *args: Any) -> Any:
        self.calls.append((method, args))
        if method == _BATCH:
            return [self._replies.get(op_method) for op_method, _ in args[0]]
        return self._replies.get(method)


def test_calls_in_a_scope_cross_as_one_batch_in_order() -> None:
    transport = _Transport({"read": "command"})
    batcher = BatchedSdkCalls(transport, _BATCH)

    with batcher:
        batcher.defer("publish_a", 1)
        batcher.defer("publish_b", 2)
        replies = batcher.flush(("read", ()))

    assert [method for method, _ in transport.calls] == [_BATCH]
    assert transport.calls[0][1][0] == [("publish_a", (1,)), ("publish_b", (2,)), ("read", ())]
    assert replies[-1] == "command"


def test_a_lone_call_skips_the_batch_wrapper() -> None:
    transport = _Transport({"read": "command"})
    batcher = BatchedSdkCalls(transport, _BATCH)

    with batcher:
        assert batcher.flush(("read", ()))[-1] == "command"

    assert transport.calls == [("read", ())]


def test_defer_outside_a_scope_sends_immediately() -> None:
    transport = _Transport()
    batcher = BatchedSdkCalls(transport, _BATCH)

    batcher.defer("publish_a", 1)

    # Nothing would flush it, so holding it would lose the call entirely.
    assert transport.calls == [("publish_a", (1,))]


def test_leaving_a_scope_flushes_what_was_never_read_back() -> None:
    transport = _Transport()
    batcher = BatchedSdkCalls(transport, _BATCH)

    with batcher:
        batcher.defer("publish_a", 1)
        assert transport.calls == []

    assert transport.calls == [("publish_a", (1,))]


def test_a_scope_that_raises_still_flushes_what_it_queued() -> None:
    transport = _Transport()
    batcher = BatchedSdkCalls(transport, _BATCH)

    def scope_that_fails() -> None:
        with batcher:
            batcher.defer("publish_a", 1)
            raise RuntimeError("publish failed")

    with pytest.raises(RuntimeError, match="publish failed"):
        scope_that_fails()

    assert transport.calls == [("publish_a", (1,))]


def test_a_failed_flush_does_not_leave_the_queue_to_be_resent() -> None:
    class _Failing(_Transport):
        def __call__(self, method: str, *args: Any) -> Any:
            super().__call__(method, *args)
            raise RuntimeError("child died")

    transport = _Failing()
    batcher = BatchedSdkCalls(transport, _BATCH)

    def scope_whose_flush_fails() -> None:
        with batcher:
            batcher.defer("publish_a", 1)

    with pytest.raises(RuntimeError, match="child died"):
        scope_whose_flush_fails()

    # One attempt, and the queue is empty rather than carrying into the next scope.
    assert transport.calls == [("publish_a", (1,))]
    with batcher:
        pass
    assert transport.calls == [("publish_a", (1,))]
