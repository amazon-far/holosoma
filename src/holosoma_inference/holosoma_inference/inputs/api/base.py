"""Protocols for input providers.

Devices implement one or both protocols:

- :class:`VelCmdProvider` — continuous velocity (joystick sticks, keyboard WASD, ROS2 twist)
- :class:`StateCommandProvider` — discrete commands (joystick buttons, keyboard keys, ROS2 strings)

A device that provides both (e.g. the SDK wireless controller) implements
both protocols in a single class.  The policy assigns the same object to
its velocity and command slots — no shared-state wiring needed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from holosoma_inference.inputs.api.commands import StateCommand, VelCmd


@runtime_checkable
class VelCmdProvider(Protocol):
    """Provides absolute velocity state each cycle."""

    def start(self) -> None:
        """Initialize the input source (start threads, subscribe to topics, etc.)."""
        ...

    def poll_velocity(self) -> VelCmd | None:
        """Return current velocity, or None if no update is available."""
        ...

    def zero(self) -> None:
        """Reset internal velocity state to zero."""
        ...


@runtime_checkable
class StateCommandProvider(Protocol):
    """Provides discrete state commands each cycle."""

    def start(self) -> None:
        """Initialize the input source."""
        ...

    def poll_commands(self) -> list[StateCommand]:
        """Return commands accumulated since last poll."""
        ...


class InputProvider(VelCmdProvider, StateCommandProvider, Protocol):
    """Combined protocol for devices that provide both velocity and commands."""
