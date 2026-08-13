"""Sensor ABC for the holosoma policy pipeline.

Implementations live in the transport layer (e.g. ``holosoma_service``).
Policies depend only on this interface; they never import a concrete sensor.

Usage in a policy subclass::

    depth = self._injected_sensors.get("depth")
    if depth is not None:
        frame = depth.get_latest()   # np.ndarray | None
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Sensor(ABC):
    """Abstract sensor polled by a policy at inference time.

    The service node constructs concrete implementations and injects the dict
    ``policy._injected_sensors: dict[str, Sensor]`` before the policy's
    ``__init__`` runs — mirroring the vel/state injection pattern.
    """

    @abstractmethod
    def start(self) -> None:
        """Start the sensor (no-op for subscription-based sensors)."""

    @abstractmethod
    def get_latest(self) -> Any:
        """Return the most recent reading, or ``None`` if unavailable/stale."""
