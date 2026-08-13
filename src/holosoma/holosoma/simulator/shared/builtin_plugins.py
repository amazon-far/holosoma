"""In-tree reference plugins.

These ship as concrete examples of the plugin pattern and are registered in
``holosoma.config_values.plugin.PLUGIN_REGISTRY``. A plugin is any class constructed as
``cls(cfg, simulator)`` that registers hooks on ``simulator.hooks`` — there is no
base class to inherit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holosoma.config_types.plugin import NoOpPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class NoOpPlugin:
    """The runtime class for the ``none`` preset: registers nothing."""

    def __init__(self, cfg: NoOpPluginConfig, simulator: BaseSimulator) -> None:
        self.cfg = cfg
        self.simulator = simulator
