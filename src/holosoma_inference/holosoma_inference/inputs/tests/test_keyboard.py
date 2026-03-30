"""Tests for keyboard input providers (new impl API).

Note: Comprehensive keyboard tests are in test_providers.py.
This module contains additional per-concern tests for keyboard-specific behaviour.
"""

from collections import deque

from .conftest import _make_policy


class TestKeyboardListenerThread:
    """Tests for _KeyboardListenerThread (new impl)."""

    def test_start_is_idempotent_in_non_tty(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import _KeyboardListenerThread

        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        listener = _KeyboardListenerThread()
        result1 = listener.start()
        result2 = listener.start()
        assert result1 is False
        assert result2 is False

    def test_subscribe_returns_independent_queue(self):
        from holosoma_inference.inputs.impl.keyboard import _KeyboardListenerThread

        listener = _KeyboardListenerThread()
        q1 = listener.subscribe()
        q2 = listener.subscribe()
        assert q1 is not q2

    def test_broadcast_delivers_to_all_subscribers(self):
        from holosoma_inference.inputs.impl.keyboard import _KeyboardListenerThread

        listener = _KeyboardListenerThread()
        q1 = listener.subscribe()
        q2 = listener.subscribe()

        for q in listener._subscribers:
            q.append("w")

        assert "w" in q1
        assert "w" in q2

    def test_popleft_on_one_doesnt_affect_other(self):
        from holosoma_inference.inputs.impl.keyboard import _KeyboardListenerThread

        listener = _KeyboardListenerThread()
        q1 = listener.subscribe()
        q2 = listener.subscribe()

        for q in listener._subscribers:
            q.append("]")

        q1.popleft()
        assert len(q1) == 0
        assert len(q2) == 1


class TestEnsureKeyboardListener:
    """Tests for _ensure_keyboard_listener helper."""

    def test_skips_shared_hardware_source(self):
        import types

        from holosoma_inference.inputs.impl.keyboard import _ensure_keyboard_listener

        p = types.SimpleNamespace(_shared_hardware_source=True)
        _ensure_keyboard_listener(p)
        assert not hasattr(p, "_keyboard_listener")

    def test_creates_listener_if_absent(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import _ensure_keyboard_listener, _KeyboardListenerThread

        p = _make_policy()
        del p._shared_hardware_source
        del p._keyboard_listener
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        _ensure_keyboard_listener(p)
        assert isinstance(p._keyboard_listener, _KeyboardListenerThread)

    def test_reuses_existing_listener(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import _ensure_keyboard_listener

        p = _make_policy()
        del p._shared_hardware_source
        del p._keyboard_listener
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        _ensure_keyboard_listener(p)
        first = p._keyboard_listener
        _ensure_keyboard_listener(p)
        assert p._keyboard_listener is first


class TestKeyboardInputPollBehaviour:
    """Additional tests for KeyboardInput queue behaviour."""

    def _make(self, velocity_keys=None):
        from holosoma_inference.inputs.impl.keyboard import KEYBOARD_COMMANDS, KeyboardInput

        queue = deque()
        return KeyboardInput(KEYBOARD_COMMANDS, queue, velocity_keys)

    def test_no_velocity_mapping_returns_none(self):
        dev = self._make()
        dev._queue.append("w")
        assert dev.poll_velocity() is None
        assert len(dev._queue) == 0  # queue still drained

    def test_commands_buffered_even_without_velocity_mapping(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        dev = self._make()
        dev._queue.extend(["]", "o"])
        dev.poll_velocity()
        assert dev.poll_commands() == [StateCommand.START, StateCommand.STOP]

    def test_poll_commands_clears_buffer(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        dev = self._make()
        dev._queue.append("]")
        dev.poll_velocity()
        assert dev.poll_commands() == [StateCommand.START]
        assert dev.poll_commands() == []
