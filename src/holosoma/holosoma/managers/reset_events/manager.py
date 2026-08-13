"""Reset event manager implementation.

This module provides the ResetEventManager class that orchestrates multiple
reset events in a simulation environment.
"""

from __future__ import annotations

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.types import EnvIds
from holosoma.utils.helpers import instantiate

from . import ResetManagerConfig
from .base import ResetEvent


class ResetEventManager:
    """Generic manager that orchestrates multiple reset events.

    Manages a collection of reset events and executes them in sequence when
    a scene reset is requested. Events are instantiated from configuration
    using our custom instantiate mechanism.

    Parameters
    ----------
    config : ResetManagerConfig
        Configuration containing the list of reset events to manage.
    simulator : BaseSimulator
        The simulator instance to pass to reset events.
    device : str
        Device string (e.g., "cuda:0", "cpu") to pass to reset events.

    Raises
    ------
    TypeError
        If any configured event does not implement the ResetEvent protocol.
    """

    def __init__(self, config: ResetManagerConfig, simulator: BaseSimulator, device: str):
        """Initialize the reset event manager."""
        self.events: list[ResetEvent] = []

        # Instantiate events from config using our custom instantiate
        for event_config in config.events:
            event = instantiate(event_config, simulator=simulator, device=device)
            if not isinstance(event, ResetEvent):
                raise TypeError(f"Event {event} does not implement ResetEvent protocol")
            self.events.append(event)

    def reset_scene(self, env_ids: EnvIds) -> None:
        """Execute all configured reset events.

        Iterates through all registered reset events and executes them
        in sequence for the specified environment IDs.
        """
        for event in self.events:
            event.reset(env_ids)
