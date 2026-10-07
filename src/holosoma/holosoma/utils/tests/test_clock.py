from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from holosoma.utils.clock import ClockPub

pytestmark = pytest.mark.no_sim


def test_clock_start_failure_releases_partial_resources_and_disables_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    context = MagicMock()
    socket = context.socket.return_value
    socket.bind.side_effect = RuntimeError("address in use")
    monkeypatch.setattr("holosoma.utils.clock.zmq.Context", lambda: context)
    clock = ClockPub(port=1234)

    clock.start()

    socket.close.assert_called_once_with()
    context.term.assert_called_once_with()
    assert clock.context is None
    assert clock.socket is None
    assert not clock.enabled


def test_clock_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    context = MagicMock()
    socket = context.socket.return_value
    monkeypatch.setattr("holosoma.utils.clock.zmq.Context", lambda: context)
    clock = ClockPub(port=1234)
    clock.start()

    clock.close()
    clock.close()

    socket.close.assert_called_once_with()
    context.term.assert_called_once_with()
    assert clock.socket is None
    assert clock.context is None
    assert clock.start_time is None
    assert not clock.enabled


def test_clock_close_attempts_context_after_socket_failure() -> None:
    socket = MagicMock()
    socket.close.side_effect = RuntimeError("socket failed")
    context = MagicMock()
    clock = ClockPub()
    clock.socket = socket
    clock.context = context
    clock.enabled = True

    with pytest.raises(RuntimeError, match="socket failed"):
        clock.close()

    context.term.assert_called_once_with()
    clock.close()
    socket.close.assert_called_once_with()
    context.term.assert_called_once_with()
