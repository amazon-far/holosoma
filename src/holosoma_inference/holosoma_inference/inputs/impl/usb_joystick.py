"""USB joystick input provider (evdev-based).

Reads a USB gamepad (Xbox / Logitech / similar) directly from
``/dev/input/event*`` via ``python-evdev``.  Bypasses the SDK
:class:`InterfaceInput` path so that SDKs without a built-in wireless
controller can still drive policies from a host-side controller.

Implements both :class:`VelCmdProvider` and :class:`StateCommandProvider`
in a single class — the policy factory assigns the same instance to both
slots when ``velocity_input == state_input == "joystick"``.
"""

from __future__ import annotations

import threading

import evdev
from loguru import logger

from holosoma_inference.inputs.api.base import InputProvider
from holosoma_inference.inputs.api.commands import StateCommand, VelCmd
from holosoma_inference.inputs.impl.joystick import JOYSTICK_COMMANDS

STICK_DEADZONE = 0.1
TRIGGER_THRESHOLD = 128  # 0-255 typical for analog triggers; >threshold counts as pressed.

# Match interface_wrapper.py's _default_wc_key_map bit layout so policies that
# inspect raw key codes still see consistent values.
_BIT_R1 = 1
_BIT_L1 = 2
_BIT_START = 4
_BIT_SELECT = 8
_BIT_R2 = 16
_BIT_L2 = 32
_BIT_A = 256
_BIT_B = 512
_BIT_X = 1024
_BIT_Y = 2048
_BIT_UP = 4096
_BIT_RIGHT = 8192
_BIT_DOWN = 16384
_BIT_LEFT = 32768

_BUTTON_BIT = {
    evdev.ecodes.BTN_A: _BIT_A,
    evdev.ecodes.BTN_B: _BIT_B,
    evdev.ecodes.BTN_X: _BIT_X,
    evdev.ecodes.BTN_Y: _BIT_Y,
    evdev.ecodes.BTN_TL: _BIT_L1,
    evdev.ecodes.BTN_TR: _BIT_R1,
    evdev.ecodes.BTN_TL2: _BIT_L2,
    evdev.ecodes.BTN_TR2: _BIT_R2,
    evdev.ecodes.BTN_START: _BIT_START,
    evdev.ecodes.BTN_SELECT: _BIT_SELECT,
}

# Subset of the Unitree wireless-controller bitmask map sufficient for the
# combinations referenced in JOYSTICK_COMMANDS.  Kept inline so this module
# does not depend on the SDK.
_KEY_LABEL = {
    _BIT_R1: "R1",
    _BIT_L1: "L1",
    _BIT_L1 | _BIT_R1: "L1+R1",
    _BIT_START: "start",
    _BIT_SELECT: "select",
    _BIT_R2: "R2",
    _BIT_L2: "L2",
    _BIT_A: "A",
    _BIT_SELECT | _BIT_A: "select+A",
    _BIT_B: "B",
    _BIT_SELECT | _BIT_B: "select+B",
    _BIT_X: "X",
    _BIT_SELECT | _BIT_X: "select+X",
    _BIT_Y: "Y",
    _BIT_SELECT | _BIT_Y: "select+Y",
    _BIT_UP: "up",
    _BIT_DOWN: "down",
    _BIT_LEFT: "left",
    _BIT_RIGHT: "right",
}


def _list_gamepads() -> list[evdev.InputDevice]:
    """Return evdev devices that look like gamepads (have sticks + buttons)."""
    candidates: list[evdev.InputDevice] = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except (PermissionError, OSError):
            continue
        caps = dev.capabilities()
        if evdev.ecodes.EV_ABS not in caps or evdev.ecodes.EV_KEY not in caps:
            dev.close()
            continue
        abs_codes = {code for code, _ in caps[evdev.ecodes.EV_ABS]}
        if {evdev.ecodes.ABS_X, evdev.ecodes.ABS_Y, evdev.ecodes.ABS_RX} <= abs_codes:
            candidates.append(dev)
        else:
            dev.close()
    return candidates


class UsbJoystickInput(InputProvider):
    """Reads stick + button state from a USB gamepad via evdev.

    A daemon thread continuously consumes events and updates internal state.
    :meth:`poll_velocity` returns the latest sticks; :meth:`poll_commands`
    edge-detects button transitions and emits :class:`StateCommand` values
    using :data:`JOYSTICK_COMMANDS`.
    """

    def __init__(self, device_index: int = 0):
        if device_index < 0:
            raise ValueError(f"joystick_device must be >= 0, got {device_index}")

        self._mapping = dict(JOYSTICK_COMMANDS)

        gamepads = _list_gamepads()
        if not gamepads:
            raise RuntimeError(
                "No USB joystick found via evdev. Is the controller plugged in "
                "and is /dev/input mounted into this container?"
            )
        if device_index >= len(gamepads):
            for d in gamepads:
                d.close()
            raise RuntimeError(f"joystick_device={device_index} but only {len(gamepads)} gamepad(s) detected")

        self._device = gamepads[device_index]
        for d in gamepads:
            if d is not self._device:
                d.close()

        abs_caps = dict(self._device.capabilities()[evdev.ecodes.EV_ABS])
        self._abs_info = {
            evdev.ecodes.ABS_X: abs_caps[evdev.ecodes.ABS_X],
            evdev.ecodes.ABS_Y: abs_caps[evdev.ecodes.ABS_Y],
            evdev.ecodes.ABS_RX: abs_caps[evdev.ecodes.ABS_RX],
        }

        self._lock = threading.Lock()
        self._lx = 0.0  # left stick X, normalized [-1, 1], left=+ after sign flip
        self._ly = 0.0  # left stick Y, normalized [-1, 1], up=+ after sign flip
        self._rx = 0.0  # right stick X, normalized [-1, 1], left=+ after sign flip
        self._key_bits = 0  # OR'd _BIT_* of currently-held buttons
        self._dpad_x = 0  # -1 / 0 / 1 from ABS_HAT0X
        self._dpad_y = 0  # -1 / 0 / 1 from ABS_HAT0Y

        self._last_label: str = ""

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="usb_joystick")
        self._thread.start()

    # -- Lifecycle --------------------------------------------------------

    def start(self) -> None:
        pass  # Thread already running from __init__.

    def close(self) -> None:
        self._stop.set()
        try:
            self._device.close()
        except OSError as e:
            logger.debug(f"evdev close raised {type(e).__name__}: {e}")

    # -- VelCmdProvider protocol -----------------------------------------

    def poll_velocity(self) -> VelCmd | None:
        with self._lock:
            keys = self._effective_key_bits_locked()
            lx, ly, rx = self._lx, self._ly, self._rx

        # Match InterfaceInput: suppress sticks while any button is held.
        if keys != 0:
            return None

        lin_x = ly if abs(ly) > STICK_DEADZONE else 0.0
        lin_y = -lx if abs(lx) > STICK_DEADZONE else 0.0
        ang_z = -rx if abs(rx) > STICK_DEADZONE else 0.0
        return VelCmd((lin_x, lin_y), ang_z)

    def zero(self) -> None:
        pass

    # -- StateCommandProvider protocol -----------------------------------

    def poll_commands(self) -> list[StateCommand]:
        with self._lock:
            keys = self._effective_key_bits_locked()
        label = _KEY_LABEL.get(keys, "")

        commands: list[StateCommand] = []
        if label and label != self._last_label:
            cmd = self._mapping.get(label)
            if cmd is not None:
                commands.append(cmd)
        self._last_label = label
        return commands

    # -- Read loop -------------------------------------------------------

    def _run(self) -> None:
        try:
            for event in self._device.read_loop():
                if self._stop.is_set():
                    return
                if event.type == evdev.ecodes.EV_ABS:
                    self._handle_abs(event.code, event.value)
                elif event.type == evdev.ecodes.EV_KEY:
                    self._handle_key(event.code, event.value)
        except OSError:
            return  # Device unplugged.

    def _handle_abs(self, code: int, value: int) -> None:
        if code in self._abs_info:
            info = self._abs_info[code]
            span = info.max - info.min
            if span <= 0:
                return
            normalized = (value - info.min) / span * 2.0 - 1.0  # → [-1, 1]
            with self._lock:
                if code == evdev.ecodes.ABS_X:
                    self._lx = normalized  # stick-left → -1 (matches SDK wireless-controller convention)
                elif code == evdev.ecodes.ABS_Y:
                    self._ly = -normalized  # stick-up (forward) → +1
                elif code == evdev.ecodes.ABS_RX:
                    self._rx = normalized
        elif code == evdev.ecodes.ABS_HAT0X:
            with self._lock:
                self._dpad_x = int(value)
        elif code == evdev.ecodes.ABS_HAT0Y:
            with self._lock:
                self._dpad_y = int(value)
        elif code == evdev.ecodes.ABS_Z:  # analog L2
            self._set_bit(_BIT_L2, value > TRIGGER_THRESHOLD)
        elif code == evdev.ecodes.ABS_RZ:  # analog R2
            self._set_bit(_BIT_R2, value > TRIGGER_THRESHOLD)

    def _handle_key(self, code: int, value: int) -> None:
        bit = _BUTTON_BIT.get(code)
        if bit is None:
            # Some controllers expose dpad as buttons rather than HAT axes.
            if code == evdev.ecodes.BTN_DPAD_UP:
                self._set_bit(_BIT_UP, value != 0)
            elif code == evdev.ecodes.BTN_DPAD_DOWN:
                self._set_bit(_BIT_DOWN, value != 0)
            elif code == evdev.ecodes.BTN_DPAD_LEFT:
                self._set_bit(_BIT_LEFT, value != 0)
            elif code == evdev.ecodes.BTN_DPAD_RIGHT:
                self._set_bit(_BIT_RIGHT, value != 0)
            return
        self._set_bit(bit, value != 0)

    def _set_bit(self, bit: int, pressed: bool) -> None:
        with self._lock:
            if pressed:
                self._key_bits |= bit
            else:
                self._key_bits &= ~bit

    def _effective_key_bits_locked(self) -> int:
        """Combine button bits with HAT-axis-derived dpad bits."""
        bits = self._key_bits
        if self._dpad_x < 0:
            bits |= _BIT_LEFT
        elif self._dpad_x > 0:
            bits |= _BIT_RIGHT
        if self._dpad_y < 0:
            bits |= _BIT_UP
        elif self._dpad_y > 0:
            bits |= _BIT_DOWN
        return bits
