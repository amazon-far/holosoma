"""Top-level inference configuration types for holosoma_inference."""

from __future__ import annotations

import tyro
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated

from .observation import ObservationConfig
from .robot import RobotConfig
from .task import TaskConfig


@dataclass(frozen=True)
class InferenceConfig:
    """Top-level configuration for policy inference.

    Combines robot, observation, and task configurations
    for running policies on real robots or in simulation.
    """

    robot: RobotConfig
    """Robot hardware and control configuration."""

    observation: ObservationConfig
    """Observation space configuration."""

    task: TaskConfig
    """Task execution configuration."""


def _make_secondary_constructor():
    """Build subcommand constructor for the secondary config selector.

    Lazily imports get_defaults() to avoid circular import (config_types <=> config_values).
    Extracts .primary from each DualModePolicyConfig so the secondary offers the same presets
    as the top-level selector. A config must be registered as a primary to be used as a secondary.
    """
    from holosoma_inference.config.config_values.inference import get_defaults

    return tyro.extras.subcommand_type_from_defaults(
        {k: v.primary for k, v in get_defaults().items()}
    )


@dataclass
class DualModePolicyConfig:
    """A primary inference config paired with an optional secondary for dual-mode."""

    primary: Annotated[InferenceConfig, tyro.conf.arg(name="")]
    """Primary inference config."""

    secondary: Annotated[
        InferenceConfig,
        tyro.conf.arg(constructor_factory=_make_secondary_constructor),
    ] | None = None
    """Secondary inference config for dual-mode (X-button switch)."""
