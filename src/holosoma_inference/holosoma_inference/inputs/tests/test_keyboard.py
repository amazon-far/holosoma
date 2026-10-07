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


class TestHoldToMoveDirections:
    """Momentary direction keys: held = move, released = stand.

    A latching direction is easy to run into a wall with — letting go of the keyboard does not stop
    the robot — so the depth-distillation policy opts into hold-to-move. That needs true key-up
    events, which ``sshkeyboard`` cannot provide (a terminal reports auto-repeat, not releases), so a
    separate pynput listener tracks held keys. These tests drive that listener directly; they never
    start pynput, so they run headless.
    """

    @staticmethod
    def _make():
        from collections import deque

        from holosoma_inference.inputs.impl.keyboard import (
            KEYBOARD_DIRECTION_COMMANDS,
            KeyboardInput,
            _HoldListener,
        )

        hold = _HoldListener(KEYBOARD_DIRECTION_COMMANDS)
        dev = KeyboardInput(deque(), direction_keys=KEYBOARD_DIRECTION_COMMANDS, hold_listener=hold)
        return dev, hold

    @staticmethod
    def _key(char):
        class _FakeKey:
            def __init__(self, c):
                self.char = c

        return _FakeKey(char)

    def test_press_emits_direction_and_release_returns_to_stand(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        dev, hold = self._make()
        hold._on_press(self._key("w"))
        assert dev.poll_commands() == [StateCommand.MOVE_FORWARD]

        hold._on_release(self._key("w"))
        assert dev.poll_commands() == [StateCommand.ZERO_VELOCITY]

    def test_holding_emits_once_not_every_cycle(self):
        """A held key must not re-emit: the control loop polls at 50 Hz."""
        from holosoma_inference.inputs.api.commands import StateCommand

        dev, hold = self._make()
        hold._on_press(self._key("w"))
        assert dev.poll_commands() == [StateCommand.MOVE_FORWARD]
        assert dev.poll_commands() == []
        assert dev.poll_commands() == []

    def test_auto_repeat_press_is_ignored(self):
        """pynput delivers auto-repeat as repeated on_press; only the first may count."""
        from holosoma_inference.inputs.api.commands import StateCommand

        dev, hold = self._make()
        hold._on_press(self._key("e"))
        assert dev.poll_commands() == [StateCommand.MOVE_RIGHT_90]
        hold._on_press(self._key("e"))
        assert dev.poll_commands() == []

    def test_second_direction_takes_over_then_falls_back_to_the_held_one(self):
        """Held keys form a stack, so releasing the newer one resumes the older."""
        from holosoma_inference.inputs.api.commands import StateCommand

        dev, hold = self._make()
        hold._on_press(self._key("w"))
        assert dev.poll_commands() == [StateCommand.MOVE_FORWARD]

        hold._on_press(self._key("a"))
        assert dev.poll_commands() == [StateCommand.MOVE_LEFT_45]

        hold._on_release(self._key("a"))
        assert dev.poll_commands() == [StateCommand.MOVE_FORWARD]

        hold._on_release(self._key("w"))
        assert dev.poll_commands() == [StateCommand.ZERO_VELOCITY]

    def test_queued_direction_press_is_ignored_under_hold_mode(self):
        """Otherwise the press queue would re-latch the direction after a release."""
        dev, _hold = self._make()
        dev._queue.append("w")
        assert dev.poll_commands() == []

    def test_non_direction_keys_still_flow_through_the_queue(self):
        """Hold tracking must only govern directions, not start/stop/record/etc."""
        from holosoma_inference.inputs.api.commands import StateCommand

        dev, _hold = self._make()
        dev._queue.extend(["]", "c", "o"])
        assert dev.poll_commands() == [
            StateCommand.START,
            StateCommand.TOGGLE_RECORDING,
            StateCommand.STOP,
        ]

    def test_without_a_hold_listener_directions_latch_as_before(self):
        """Headless fallback: no key-up events means the previous press-only behavior."""
        from collections import deque

        from holosoma_inference.inputs.api.commands import StateCommand
        from holosoma_inference.inputs.impl.keyboard import KEYBOARD_DIRECTION_COMMANDS, KeyboardInput

        dev = KeyboardInput(deque(["w", "s"]), direction_keys=KEYBOARD_DIRECTION_COMMANDS, hold_listener=None)
        assert dev.poll_commands() == [StateCommand.MOVE_FORWARD, StateCommand.MOVE_BACKWARD]

    def test_special_keys_do_not_crash_the_hold_listener(self):
        """pynput sends key objects with no ``.char`` (shift, arrows); they must be ignored."""
        dev, hold = self._make()

        class _SpecialKey:
            @property
            def char(self):
                raise AttributeError("special key")

        hold._on_press(_SpecialKey())
        hold._on_release(_SpecialKey())
        assert dev.poll_commands() == []
