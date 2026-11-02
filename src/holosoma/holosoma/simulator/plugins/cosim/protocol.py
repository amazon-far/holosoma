"""Wire protocol between a co-simulation leader (physics) and follower (rendering).

Each message is one ZeroMQ multipart message: a UTF-8 JSON header, then one frame per array holding
its raw C-contiguous bytes. The header lists the arrays as ``[name, dtype, shape]``, so arrays cross
processes, and torch versions, without pickling. The leader sends requests on a REQ socket and the
follower answers on a REP socket, strictly alternating.

Message types (version 1):

- ``hello``: environment facts; the reply carries the follower's.
- ``step``: header ``cams`` (camera names to render) and ``rows`` (env ids, or null for all); arrays
  ``root`` ``[N, 7]`` (position, xyzw quaternion), ``dof_pos`` ``[N, D]`` and ``objects`` ``[K, N, 7]``,
  float32, with positions relative to each env's origin. The reply holds one ``"<camera>/<data type>"``
  array per rendered image, ``[n, H, W, C]``.
- ``bye``: the leader is closing; the reply is ``ok``.
- ``error``: header ``msg``; sent in place of a reply when the follower fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import zmq

PROTOCOL_VERSION = 1


class CoSimError(RuntimeError):
    """The peer reported an error, or sent a message this side cannot accept."""


@dataclass
class Message:
    type: str
    header: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


class _Socket:
    def __init__(self, endpoint: str, socket_type: int, bind: bool):
        # The process-wide context: a private one blocks in term() until every socket is closed.
        self._socket = zmq.Context.instance().socket(socket_type)
        self._socket.setsockopt(zmq.LINGER, 0)
        if bind:
            self._socket.bind(endpoint)
        else:
            self._socket.connect(endpoint)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

    def send(
        self, msg_type: str, header: dict[str, Any] | None = None, arrays: dict[str, np.ndarray] | None = None
    ) -> None:
        """Send one message. Arrays are sent without copying and must stay unchanged until the reply."""
        buffers = [np.ascontiguousarray(a) for a in (arrays or {}).values()]
        specs = [[name, a.dtype.str, list(a.shape)] for name, a in zip(arrays or {}, buffers, strict=True)]
        head = {**(header or {}), "v": PROTOCOL_VERSION, "type": msg_type, "arrays": specs}
        self._socket.send_multipart([json.dumps(head).encode(), *buffers], copy=False)

    def recv(self, timeout_s: float | None) -> Message:
        """Receive one message, as writable arrays.

        Raises
        ------
        TimeoutError
            Nothing arrived within ``timeout_s`` seconds (``None`` waits indefinitely).
        CoSimError
            The peer sent ``error``, or a message with a different protocol version.
        """
        timeout_ms = None if timeout_s is None else int(timeout_s * 1000)
        if not self._poller.poll(timeout_ms):
            raise TimeoutError(f"No co-simulation message within {timeout_s} s")
        frames = self._socket.recv_multipart(copy=False)
        head = json.loads(bytes(frames[0].buffer).decode())
        if head.get("v") != PROTOCOL_VERSION:
            raise CoSimError(f"Co-simulation protocol version {head.get('v')} != {PROTOCOL_VERSION}")
        specs = head.pop("arrays")
        msg_type = head.pop("type")
        del head["v"]
        if msg_type == "error":
            raise CoSimError(f"Co-simulation peer failed: {head.get('msg')}")
        arrays = {
            name: np.frombuffer(frame.buffer, dtype=np.dtype(dtype)).reshape(shape).copy()
            for (name, dtype, shape), frame in zip(specs, frames[1:], strict=True)
        }
        return Message(msg_type, head, arrays)

    def close(self) -> None:
        self._socket.close(linger=0)


class Requester(_Socket):
    """Leader side: connects and sends requests."""

    def __init__(self, endpoint: str):
        super().__init__(endpoint, zmq.REQ, bind=False)


class Replier(_Socket):
    """Follower side: binds and answers requests."""

    def __init__(self, endpoint: str):
        super().__init__(endpoint, zmq.REP, bind=True)
