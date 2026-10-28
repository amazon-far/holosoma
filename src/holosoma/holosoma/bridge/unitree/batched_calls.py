"""Batching for the multiprocess Unitree bridge's calls to its SDK child."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast


class BatchedSdkCalls:
    """Collects SDK calls made inside a transport scope and sends them in one transport call.

    Each call to the SDK child blocks the simulator's step thread while the child is scheduled. At
    the default transport cadence of one exchange per physics step, that is a large share of the
    step budget, and a transport update makes three (state, odometry, command read).

    Deferral is active only inside a scope: with none in flight there is nothing to batch with and
    nothing that would flush it, so a deferred call would never reach the SDK. A caller driving the
    bridge's publishes directly, rather than through ``SimulatorBridge.transport_step``, therefore
    gets the same one-call-per-publish behaviour it has always had.
    """

    def __init__(self, call: Callable[..., Any], batch_method: str) -> None:
        self._call = call
        self._batch_method = batch_method
        self._pending: list[tuple[str, tuple[Any, ...]]] = []
        self._collecting = False

    def defer(self, method: str, *args: Any) -> None:
        """Queue one call for the next flush, or send it now if no scope is open."""
        if not self._collecting:
            self._call(method, *args)
            return
        self._pending.append((method, args))

    def flush(self, *trailing: tuple[str, tuple[Any, ...]]) -> list[Any]:
        """Send every queued call plus ``trailing`` in ONE transport call; return their payloads.

        The receiver runs them in queue order, so the calls reach the SDK in the order they were
        made. ``trailing`` carries the call whose reply the caller needs, so it is sent last.
        """
        operations = self._pending + list(trailing)
        self._pending = []
        if not operations:
            return []
        if len(operations) == 1:
            method, args = operations[0]
            return [self._call(method, *args)]
        return cast("list[Any]", self._call(self._batch_method, operations))

    def __enter__(self) -> None:
        self._collecting = True

    def __exit__(self, *exc_info: object) -> None:
        # Stop collecting before flushing, so the flush itself cannot be deferred. Flushing on the
        # way out covers a scope that queued a publish without a following reply-bearing call, and
        # one that raised part-way through -- which, before batching, had already sent it.
        self._collecting = False
        self.flush()
