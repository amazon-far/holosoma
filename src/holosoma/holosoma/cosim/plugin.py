"""Simulator plugin that makes a simulation task a co-simulation source."""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.cosim.bridge import Publisher
from holosoma.cosim.cuda import CacheAllocator

if TYPE_CHECKING:
    from holosoma.config_types.plugin import CosimPublisherPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class CosimPublisherPlugin:
    """Share ``simulator``'s state caches with a renderer and open a :class:`Publisher` to it.

    Installs the shared cache allocator, so it must run before anything reads simulator state.
    Construction blocks until the renderer accepts the connection.
    """

    def __init__(self, cfg: CosimPublisherPluginConfig, simulator: BaseSimulator) -> None:
        if not cfg.endpoint:
            raise ValueError("Set the co-simulation renderer endpoint with --plugin.<key>.endpoint")
        state = simulator.state
        fields = tuple(state.writable_names() if cfg.fields is None else cfg.fields)
        self.allocator = CacheAllocator(fields)
        state.set_buffer_allocator(self.allocator)
        published = tuple(name for name in fields if state.get(name).numel())
        self.publisher = Publisher(simulator, self.allocator, endpoint=cfg.endpoint, fields=published)
