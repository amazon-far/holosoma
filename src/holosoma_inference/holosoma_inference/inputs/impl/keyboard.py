"""Keyboard input providers and shared listener."""

from __future__ import annotations

import sys
import threading
from collections import deque

import numpy as np
from loguru import logger
from sshkeyboard import listen_keyboard

from holosoma_inference.inputs.api.base import InputProvider
from holosoma_inference.inputs.api.commands import StateCommand, VelCmd

# ---------------------------------------------------------------------------
# Keyboard command mappings (discrete commands)
# ---------------------------------------------------------------------------

KEYBOARD_COMMANDS: dict[str, StateCommand] = {
    "]": StateCommand.START,
    "o": StateCommand.STOP,
    "i": StateCommand.INIT,
    "v": StateCommand.KP_DOWN_FINE,
    "b": StateCommand.KP_UP_FINE,
    "f": StateCommand.KP_DOWN,
    "g": StateCommand.KP_UP,
    "r": StateCommand.KP_RESET,
    "=": StateCommand.STAND_TOGGLE,
    "z": StateCommand.ZERO_VELOCITY,
    "m": StateCommand.START_MOTION_CLIP,
    "c": StateCommand.TOGGLE_RECORDING,
    "x": StateCommand.SWITCH_MODE,
    **{str(n): StateCommand[f"SWITCH_POLICY_{n}"] for n in range(1, 10)},
}

# ---------------------------------------------------------------------------
# Keyboard velocity mappings (continuous velocity increments)
#
# Each entry maps a keycode to (array_index, column, delta):
#   array_index 0 = lin_vel, 1 = ang_vel
#   column = which element within that array
#   delta = increment per keypress
# ---------------------------------------------------------------------------

KEYBOARD_VELOCITY_LOCOMOTION: dict[str, tuple[int, int, float]] = {
    "w": (0, 0, +0.1),  # lin_vel[0, 0] += 0.1
    "s": (0, 0, -0.1),  # lin_vel[0, 0] -= 0.1
    "a": (0, 1, +0.1),  # lin_vel[0, 1] += 0.1
    "d": (0, 1, -0.1),  # lin_vel[0, 1] -= 0.1
    "q": (1, 0, -0.1),  # ang_vel[0, 0] -= 0.1
    "e": (1, 0, +0.1),  # ang_vel[0, 0] += 0.1
}

# ---------------------------------------------------------------------------
# Keyboard direction mappings (discrete, absolute headings)
#
# For policies trained on a direction *class* instead of a velocity vector.
# Each press selects a heading outright, so one tap responds immediately —
# unlike the velocity map above, where a key nudges an accumulator and a
# direction reversal takes several presses to cross zero.
# ---------------------------------------------------------------------------

KEYBOARD_DIRECTION_COMMANDS: dict[str, StateCommand] = {
    "w": StateCommand.MOVE_FORWARD,
    "s": StateCommand.MOVE_BACKWARD,
    "a": StateCommand.MOVE_LEFT_45,
    "d": StateCommand.MOVE_RIGHT_45,
    "q": StateCommand.MOVE_LEFT_90,
    "e": StateCommand.MOVE_RIGHT_90,
}


# Key-repeat delays for a held key, in seconds. sshkeyboard's own defaults
# (0.75 / 0.05) put a long stall before the first repeat, which feels like
# dropped input when steering a robot.
REPEAT_DELAY_FIRST = 0.15
REPEAT_DELAY_REST = 0.05


class _KeyboardListenerThread(threading.Thread):
    """Daemon thread that broadcasts keypresses to subscriber queues.

    ``start()`` is idempotent and returns whether the listener is active.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._subscribers: list[deque[str]] = []

    def subscribe(self) -> deque[str]:
        q: deque[str] = deque()
        self._subscribers.append(q)
        return q

    def start(self) -> bool:
        """Start the thread if not already running. Returns True if active."""
        if self.is_alive():
            return True
        if not sys.stdin.isatty():
            return False
        super().start()
        return True

    def run(self) -> None:
        def on_press(keycode):
            for q in self._subscribers:
                q.append(keycode)

        try:
            # sshkeyboard defaults to delay_second_char=0.75, so holding a key
            # emits one press then stalls for 3/4 of a second before repeating —
            # which reads as the robot ignoring input. Shorten both repeat delays
            # so a held key streams smoothly; discrete taps are unaffected.
            listener = listen_keyboard(
                on_press=on_press,
                delay_second_char=REPEAT_DELAY_FIRST,
                delay_other_chars=REPEAT_DELAY_REST,
            )
            listener.start()
            listener.join()
        except OSError:
            pass
        except TypeError:
            # Older sshkeyboard without the delay kwargs: fall back to defaults
            # rather than losing keyboard control entirely.
            listener = listen_keyboard(on_press=on_press)
            listener.start()
            listener.join()


# Module-level singleton — one listener thread shared across all KeyboardInput instances.
_listener: _KeyboardListenerThread | None = None


def get_keyboard_listener() -> _KeyboardListenerThread:
    """Return the module-level keyboard listener, creating it on first call."""
    global _listener  # noqa: PLW0603
    if _listener is None:
        _listener = _KeyboardListenerThread()
    return _listener


class _HoldListener:
    """Tracks which direction keys are physically held, via true key-down/key-up events.

    ``sshkeyboard`` (the listener above) reports presses only — a terminal delivers auto-repeat, not
    releases — so it cannot express "move while held". pynput reads X11 key events directly and does
    report releases, which is what makes hold-to-move possible.

    Held keys form a stack: pressing a second direction while the first is down switches to the new
    one, and releasing it falls back to the one still held rather than to stand. That matches how a
    gamepad d-pad behaves and is the reference deployment's behavior.

    Requires a DISPLAY. Without one (or without pynput) :meth:`start` returns False and the caller
    keeps the press-only path, where a direction latches until another key changes it.
    """

    def __init__(self, direction_keys: dict[str, StateCommand]) -> None:
        self._direction_keys = dict(direction_keys)
        self._lock = threading.Lock()
        self._held: list[str] = []
        self._listener = None

    def start(self) -> bool:
        """Begin listening. Returns False when key-up events are unavailable."""
        try:
            from pynput import keyboard as pynput_kb
        except Exception as exc:  # import itself can fail for want of a display
            logger.debug(f"pynput unavailable ({exc}); direction keys will latch instead of holding.")
            return False

        try:
            self._listener = pynput_kb.Listener(on_press=self._on_press, on_release=self._on_release)
            self._listener.start()
        except Exception as exc:  # no X11 display, insufficient permissions, ...
            logger.debug(f"pynput listener failed ({exc}); direction keys will latch instead of holding.")
            self._listener = None
            return False
        return True

    @staticmethod
    def _keycode(key) -> str | None:
        """Lowercase single-char keycode for a pynput event, or None for a special key."""
        try:
            return key.char.lower() if key.char else None
        except AttributeError:
            return None

    def _on_press(self, key) -> None:
        keycode = self._keycode(key)
        if keycode is None or keycode not in self._direction_keys:
            return
        with self._lock:
            # pynput delivers auto-repeat as repeated on_press; only the first counts.
            if keycode not in self._held:
                self._held.append(keycode)

    def _on_release(self, key) -> None:
        keycode = self._keycode(key)
        if keycode is None or keycode not in self._direction_keys:
            return
        with self._lock:
            if keycode in self._held:
                self._held.remove(keycode)

    def active_command(self) -> StateCommand | None:
        """The most recently pressed still-held direction, or None if none are held."""
        with self._lock:
            keycode = self._held[-1] if self._held else None
        return self._direction_keys[keycode] if keycode else None

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


class KeyboardInput(InputProvider):
    """Unified keyboard device implementing both velocity and command protocols.

    Subscribes to a single keyboard queue. ``poll_velocity()`` drains the queue,
    applies velocity key increments, and buffers any command matches.
    ``poll_commands()`` returns the buffered commands.

    If no velocity_keys mapping is provided, ``poll_velocity()`` returns None
    but still drains the queue and buffers commands.
    """

    def __init__(
        self,
        queue: deque[str],
        velocity_keys: dict[str, tuple[int, int, float]] | None = None,
        direction_keys: dict[str, StateCommand] | None = None,
        hold_listener: _HoldListener | None = None,
    ) -> None:
        self._mapping = dict(KEYBOARD_COMMANDS)
        # Direction keys shadow the shared command table so a policy can rebind
        # w/a/s/d/q/e without those keys also firing their default meaning.
        if direction_keys:
            self._mapping.update(direction_keys)
        self._queue = queue
        # A key cannot be both a velocity increment and a direction command; when
        # a policy asks for directions, they win and the accumulator is unused.
        self._velocity_keys = None if direction_keys else velocity_keys
        self._lin_vel = np.zeros((1, 2))
        self._ang_vel = np.zeros((1, 1))
        self._pending_commands: list[StateCommand] = []

        # Hold-to-move: when key-up events are available, the direction is driven by which key is
        # physically down rather than by the last one tapped. Direction keys are then dropped from
        # the press queue — the hold state is the single source of truth, and letting a press also
        # enqueue the same command would re-latch the direction after release.
        self._hold_listener = hold_listener
        self._direction_commands = set(direction_keys.values()) if direction_keys else set()
        self._held_command: StateCommand | None = None

    @classmethod
    def create(
        cls,
        velocity_keys: dict[str, tuple[int, int, float]] | None = None,
        direction_keys: dict[str, StateCommand] | None = None,
        hold_directions: bool = False,
    ) -> KeyboardInput:
        """Create a KeyboardInput subscribed to the module-level keyboard listener.

        ``hold_directions`` asks for hold-to-move direction keys (release returns to stand). It needs
        true key-up events, so it silently falls back to the latching press-only behavior when
        pynput or a display is unavailable — a headless run stays controllable either way.
        """
        listener = get_keyboard_listener()
        queue = listener.subscribe()

        hold_listener = None
        if hold_directions and direction_keys:
            candidate = _HoldListener(direction_keys)
            if candidate.start():
                hold_listener = candidate
                logger.info("Keyboard direction keys are hold-to-move (release returns to stand).")
            else:
                logger.warning(
                    "Hold-to-move needs key-up events (pynput + a DISPLAY); "
                    "direction keys will latch until another is pressed."
                )
        return cls(queue, velocity_keys, direction_keys, hold_listener)

    def start(self) -> None:
        pass  # Listener already started by factory / create()

    def _drain_queue(self) -> None:
        """Process all pending keypresses into velocity state and command buffer."""
        while True:
            try:
                keycode = self._queue.popleft()
            except IndexError:
                break
            action = self._velocity_keys.get(keycode) if self._velocity_keys else None
            if action is not None:
                array_idx, col, delta = action
                if array_idx == 0:
                    self._lin_vel[0, col] += delta
                else:
                    self._ang_vel[0, col] += delta
                continue
            cmd = self._mapping.get(keycode)
            if cmd is None:
                continue
            # Under hold-to-move the hold listener owns the direction; a queued press for the same
            # command would re-assert it after a release and defeat the fallback to stand.
            if self._hold_listener is not None and cmd in self._direction_commands:
                continue
            self._pending_commands.append(cmd)

    def _poll_hold_state(self) -> None:
        """Turn changes in which direction key is held into commands.

        Emitted on transitions only, so a held key does not spam the same command every cycle. When
        the last direction is released this emits ``ZERO_VELOCITY``, which every direction-commanded
        policy already maps to stand.
        """
        if self._hold_listener is None:
            return
        active = self._hold_listener.active_command()
        if active == self._held_command:
            return
        self._held_command = active
        self._pending_commands.append(active if active is not None else StateCommand.ZERO_VELOCITY)

    def poll_velocity(self) -> VelCmd | None:
        self._drain_queue()
        if not self._velocity_keys:
            return None
        return VelCmd(
            (float(self._lin_vel[0, 0]), float(self._lin_vel[0, 1])),
            float(self._ang_vel[0, 0]),
        )

    def zero(self) -> None:
        """Reset velocity state to zero."""
        self._lin_vel[:] = 0.0
        self._ang_vel[:] = 0.0

    def poll_commands(self) -> list[StateCommand]:
        self._drain_queue()
        # After draining, so a direction change from the hold state is applied last and wins over a
        # same-cycle queued command.
        self._poll_hold_state()
        commands = self._pending_commands
        self._pending_commands = []
        return commands
