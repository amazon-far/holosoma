"""Liveness heartbeat for the depth shared-memory transport.

The shared-memory block carries only the frame, so a consumer cannot tell a live
producer from a dead one: the last frame stays readable after the producer stops.
Producers therefore also publish a small heartbeat on a zmq PUB socket after every
frame write, and the consumer measures how long the frame sequence has stopped
advancing.

Wire contract (version 1), shared with ``holosoma``'s ``DepthShmPlugin``:

- endpoint: ``ipc:///tmp/<shm_name>.heartbeat`` by default. The producer binds and
  the consumer connects; either may start first.
- message: one UTF-8 JSON object per frame write,
  ``{"v": 1, "seq": int, "t_capture": float, "state": "OK" | "RECONNECTING" | "FAILED"}``.
  ``seq`` increments on every frame write. While its camera is down a producer keeps
  the last frame in shared memory and keeps publishing its last ``seq`` with
  ``state="RECONNECTING"``; ``FAILED`` means it has given up. ``t_capture`` is the
  producer's ``time.time()`` at capture and is informational only: staleness is
  measured on the consumer's own clock, so the two clocks never need to agree.

A producer that does not publish a heartbeat keeps working: the consumer logs one
warning and leaves liveness unchecked.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

HEARTBEAT_VERSION = 1
STATE_OK = "OK"
STATE_RECONNECTING = "RECONNECTING"
STATE_FAILED = "FAILED"
STATES = (STATE_OK, STATE_RECONNECTING, STATE_FAILED)


def default_heartbeat_endpoint(shm_name: str) -> str:
    """The endpoint a producer and consumer agree on for the block ``shm_name``."""
    return f"ipc:///tmp/{shm_name}.heartbeat"


def encode_heartbeat(seq: int, state: str, t_capture: float) -> bytes:
    if state not in STATES:
        raise ValueError(f"Heartbeat state must be one of {STATES}, got {state!r}")
    return json.dumps({"v": HEARTBEAT_VERSION, "seq": int(seq), "t_capture": float(t_capture), "state": state}).encode()


def decode_heartbeat(message: bytes) -> dict[str, Any]:
    """Parse and validate one heartbeat; raises ValueError on anything malformed."""
    try:
        beat = json.loads(message.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Heartbeat is not UTF-8 JSON: {exc}") from None
    if not isinstance(beat, dict) or beat.get("v") != HEARTBEAT_VERSION:
        raise ValueError(f"Unsupported heartbeat (expected v={HEARTBEAT_VERSION}): {beat!r}")
    if not isinstance(beat.get("seq"), int) or beat.get("state") not in STATES:
        raise ValueError(f"Heartbeat needs an int seq and a state in {STATES}: {beat!r}")
    return beat


class DepthHeartbeatPublisher:
    """Producer side: publish one heartbeat per frame write.

    For camera daemons that write the depth block. Use it from a single thread
    (zmq sockets are not thread-safe). Publishing never blocks the producer.
    """

    def __init__(self, endpoint: str):
        import zmq

        self._zmq = zmq
        # The process-wide context: a private one blocks in term() (and at garbage collection)
        # until every socket on it is closed.
        self._socket = zmq.Context.instance().socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)
        self.endpoint = endpoint

    def publish(self, seq: int, state: str = STATE_OK, t_capture: float | None = None) -> None:
        message = encode_heartbeat(seq, state, time.time() if t_capture is None else t_capture)
        try:
            self._socket.send(message, self._zmq.NOBLOCK)
        except self._zmq.Again:
            pass  # no subscriber yet, or its queue is full: a dropped heartbeat is harmless

    def close(self) -> None:
        self._socket.close()


@dataclass(frozen=True)
class DepthLiveness:
    """What the consumer knows about the producer at one poll."""

    checked: bool
    """False until the first heartbeat arrives (a producer without one, or not started yet)."""
    state: str | None
    """The producer's last reported state."""
    stale_s: float
    """Seconds since ``seq`` last advanced, on the consumer's clock (0.0 when unchecked)."""
    seq: int | None
    restarted: bool
    """True on the poll where ``seq`` went backwards: the producer restarted."""


class DepthHeartbeatMonitor:
    """Consumer side: subscribe to a producer's heartbeat and track frame staleness."""

    def __init__(
        self,
        endpoint: str,
        clock: Callable[[], float] = time.monotonic,
        missing_warn_after_s: float = 2.0,
    ):
        import zmq

        self._zmq = zmq
        self._clock = clock
        self._missing_warn_after_s = missing_warn_after_s
        self._socket = zmq.Context.instance().socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.CONFLATE, 1)  # only the newest heartbeat matters
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)
        self.endpoint = endpoint

        self._created_at = clock()
        self._seq: int | None = None
        self._state: str | None = None
        self._advanced_at = self._created_at
        self._warned_missing = False
        self._warned_malformed = False

    def poll(self) -> DepthLiveness:
        now = self._clock()
        restarted = False
        while True:
            try:
                message = self._socket.recv(self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            try:
                beat = decode_heartbeat(message)
            except ValueError as exc:
                if not self._warned_malformed:
                    logger.warning(f"[DepthHeartbeat] ignoring malformed heartbeat on {self.endpoint}: {exc}")
                    self._warned_malformed = True
                continue
            if self._seq is None or beat["seq"] != self._seq:
                restarted = self._seq is not None and beat["seq"] < self._seq
                self._seq = beat["seq"]
                self._advanced_at = now
            self._state = beat["state"]

        if self._seq is None:
            if not self._warned_missing and now - self._created_at > self._missing_warn_after_s:
                logger.warning(
                    f"[DepthHeartbeat] no heartbeat on {self.endpoint}: the depth producer does not "
                    f"publish one, so a stalled producer will not be detected"
                )
                self._warned_missing = True
            return DepthLiveness(checked=False, state=None, stale_s=0.0, seq=None, restarted=False)
        return DepthLiveness(
            checked=True, state=self._state, stale_s=now - self._advanced_at, seq=self._seq, restarted=restarted
        )

    def close(self) -> None:
        self._socket.close()
