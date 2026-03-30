"""Tests for keyboard input providers (new impl API).

Note: Comprehensive keyboard tests are in test_providers.py.
This module contains additional per-concern tests for keyboard-specific behaviour.
"""

from collections import deque


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


class TestGetKeyboardListener:
    """Tests for get_keyboard_listener module-level singleton."""

    def test_returns_keyboard_listener_thread(self, monkeypatch):
        import holosoma_inference.inputs.impl.keyboard as kb_module
        from holosoma_inference.inputs.impl.keyboard import _KeyboardListenerThread, get_keyboard_listener

        monkeypatch.setattr(kb_module, "_listener", None)
        listener = get_keyboard_listener()
        assert isinstance(listener, _KeyboardListenerThread)

    def test_returns_same_instance_on_repeated_calls(self, monkeypatch):
        import holosoma_inference.inputs.impl.keyboard as kb_module
        from holosoma_inference.inputs.impl.keyboard import get_keyboard_listener

        monkeypatch.setattr(kb_module, "_listener", None)
        first = get_keyboard_listener()
        second = get_keyboard_listener()
        assert first is second

    def test_reuses_existing_listener(self, monkeypatch):
        import holosoma_inference.inputs.impl.keyboard as kb_module
        from holosoma_inference.inputs.impl.keyboard import _KeyboardListenerThread, get_keyboard_listener

        existing = _KeyboardListenerThread()
        monkeypatch.setattr(kb_module, "_listener", existing)
        result = get_keyboard_listener()
        assert result is existing


class TestKeyboardInputPollBehaviour:
    """Additional tests for KeyboardInput queue behaviour."""

    def _make(self, velocity_keys=None):
        from holosoma_inference.inputs.impl.keyboard import KeyboardInput

        queue = deque()
        return KeyboardInput(queue, velocity_keys)

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
