"""Robot SDK wireless controller input device.

Implements both :class:`VelCmdProvider` and :class:`StateCommandProvider`
protocols in a single class.  Reads the joystick message once per cycle
in :meth:`poll_velocity`, caches button states, and :meth:`poll_commands`
does edge detection on the cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma_inference.inputs.api.commands import StateCommand, VelCmd

if TYPE_CHECKING:
    from holosoma_inference.sdk.base.base_interface import BaseInterface


STICK_DEADZONE = 0.1


class InterfaceInput:
    """Reads both velocity and commands from the robot SDK interface.

    Satisfies both ``VelCmdProvider`` and ``StateCommandProvider`` protocols.
    The policy factory assigns the same instance to both its velocity and
    command slots, eliminating the need for shared-state wiring.
    """

    def __init__(self, interface: BaseInterface, mapping: dict[str, StateCommand]):
        self.interface = interface
        self._mapping = mapping
        self.key_states: dict[str, bool] = {}
        self.last_key_states: dict[str, bool] = {}

    def start(self) -> None:
        pass  # Joystick hardware initialized by SDK

    # -- VelCmdProvider protocol -----------------------------------------------

    def poll_velocity(self) -> VelCmd | None:
        wc_msg = self.interface.get_joystick_msg()
        if wc_msg is None:
            return None

        # Cache button states for poll_commands() (always, even when
        # sticks are suppressed during button presses).
        self.last_key_states = self.key_states.copy()
        cur_key = self.interface.get_joystick_key(wc_msg)
        if cur_key:
            self.key_states[cur_key] = True
        else:
            self.key_states = dict.fromkeys(self.key_states.keys(), False)

        # Sticks are only read when no buttons are pressed, matching the
        # behaviour of BaseInterface.process_joystick_input.
        if getattr(wc_msg, "keys", 0) != 0:
            return None

        lx = getattr(wc_msg, "lx", 0.0)
        ly = getattr(wc_msg, "ly", 0.0)
        rx = getattr(wc_msg, "rx", 0.0)

        lin_x = ly if abs(ly) > STICK_DEADZONE else 0.0
        lin_y = -lx if abs(lx) > STICK_DEADZONE else 0.0
        ang_z = -rx if abs(rx) > STICK_DEADZONE else 0.0

        return VelCmd((lin_x, lin_y), ang_z)

    def zero(self) -> None:
        pass

    # -- StateCommandProvider protocol -----------------------------------------

    def poll_commands(self) -> list[StateCommand]:
        """Edge-detect rising edges from cached key_states."""
        commands: list[StateCommand] = []
        for key, is_pressed in self.key_states.items():
            if is_pressed and not self.last_key_states.get(key, False):
                cmd = self._mapping.get(key)
                if cmd is not None:
                    commands.append(cmd)
        return commands
