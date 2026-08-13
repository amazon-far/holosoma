"""ROS-free drop-oldest publish worker for async sensor egress.

A single-consumer worker thread fed by a bounded, drop-oldest queue. The sim thread calls
:meth:`PublishWorker.submit` (non-blocking); the worker thread drains and runs the supplied
``publish_fn``. When the queue is full the OLDEST item is dropped (latest-frame-wins) and a
dropped counter is incremented — so a slow/stalled consumer never blocks the sim, and overruns are
visible rather than silent. Pure ``threading``/``collections.deque`` (no rclpy), so the
backpressure policy is unit-tested without a ROS environment.

One worker per route (so a slow large-depth stream cannot head-of-line-block a fast RGB one); the
egress owns a worker per route when ``async_publish`` is on, and bypasses it (inline) when off.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Generic, TypeVar

from loguru import logger

T = TypeVar("T")


class PublishWorker(Generic[T]):
    """Bounded drop-oldest queue + one consumer thread that runs ``publish_fn`` per item."""

    def __init__(self, publish_fn: Callable[[T], None], *, maxlen: int, name: str) -> None:
        self._publish_fn = publish_fn
        self._name = name
        self._queue: deque[T] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop = threading.Event()
        self._dropped = 0
        self._published = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    @property
    def dropped(self) -> int:
        """Frames dropped so far due to a full queue (overrun visibility)."""
        return self._dropped

    @property
    def published(self) -> int:
        return self._published

    def submit(self, item: T) -> None:
        """Enqueue an item without blocking. Drops the oldest if the queue is full."""
        with self._lock:
            # deque(maxlen) silently evicts the oldest on append; count it as a drop so overruns
            # are reported rather than hidden.
            if len(self._queue) == self._queue.maxlen:
                self._dropped += 1
            self._queue.append(item)
        self._wakeup.set()

    def _drain_one(self) -> tuple[bool, T | None]:
        with self._lock:
            if self._queue:
                return True, self._queue.popleft()
            self._wakeup.clear()
            return False, None

    def _run(self) -> None:
        while not self._stop.is_set():
            got, item = self._drain_one()
            if not got or item is None:
                self._wakeup.wait(timeout=0.5)
                continue
            try:
                self._publish_fn(item)
                self._published += 1
            except Exception as exc:
                logger.error(f"PublishWorker '{self._name}' publish_fn failed: {exc}")

    def stop(self, *, timeout: float = 2.0) -> None:
        """Signal stop and join the thread. Idempotent. Drains nothing further on the way out."""
        self._stop.set()
        self._wakeup.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self._dropped:
            logger.warning(f"PublishWorker '{self._name}' dropped {self._dropped} frame(s) under backpressure.")
