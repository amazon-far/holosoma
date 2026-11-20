"""Run profiles, selected with ``run:<name>``."""

from __future__ import annotations

from holosoma.config_types.run import RunConfig
from holosoma.utils.config_registry import ConfigRegistry

RUN_REGISTRY = ConfigRegistry(RunConfig, group="holosoma.config.run")

batch = RUN_REGISTRY.add("batch", RunConfig())
"""Many headless environments stepping as fast as possible, for training."""

interactive = RUN_REGISTRY.add("interactive", RunConfig(num_envs=1, headless=False, time_scale=1.0))
"""One environment in the viewer, paced to real time."""
