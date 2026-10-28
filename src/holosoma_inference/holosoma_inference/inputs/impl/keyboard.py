"""Keyboard input providers and shared listener."""

from __future__ import annotations

import sys
import threading
from collections import deque

import numpy as np
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
            listener = listen_keyboard(on_press=on_press)
            listener.start()
            listener.join()
        except OSError:
            pass


# Module-level singleton — one listener thread shared across all KeyboardInput instances.
_listener: _KeyboardListenerThread | None = None


def get_keyboard_listener() -> _KeyboardListenerThread:
    """Return the module-level keyboard listener, creating it on first call."""
    global _listener  # noqa: PLW0603
    if _listener is None:
        _listener = _KeyboardListenerThread()
    return _listener


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
    ) -> None:
        self._mapping = dict(KEYBOARD_COMMANDS)
        self._queue = queue
        self._velocity_keys = velocity_keys
        self._lin_vel = np.zeros((1, 2))
        self._ang_vel = np.zeros((1, 1))
        self._pending_commands: list[StateCommand] = []

    @classmethod
    def create(
        cls,
        velocity_keys: dict[str, tuple[int, int, float]] | None = None,
    ) -> KeyboardInput:
        """Create a KeyboardInput subscribed to the module-level keyboard listener."""
        listener = get_keyboard_listener()
        queue = listener.subscribe()
        return cls(queue, velocity_keys)

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
            if cmd is not None:
                self._pending_commands.append(cmd)

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
        commands = self._pending_commands
        self._pending_commands = []
        return commands
