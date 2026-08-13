"""Configuration types for observation manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ObsTermCfg:
    """Configuration for a single observation term."""

    func: str
    """Import path to the observation function, e.g.
    ``holosoma.managers.observation.terms.locomotion:base_lin_vel``."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments forwarded to ``func``."""

    scale: float | tuple[float, ...] = 1.0
    """Scaling factor(s) applied after noise and clipping."""

    noise: float = 0.0
    """Noise magnitude applied before scaling when the group enables noise."""

    clip: tuple[float, float] | None = None
    """Optional min/max clip bounds applied after scaling."""


@dataclass(frozen=True)
class ObsGroupCfg:
    """Configuration for an observation group (e.g. actor or critic input)."""

    terms: dict[str, ObsTermCfg] = field(default_factory=dict)
    """Mapping of term name to its configuration."""

    concatenate: bool = True
    """Whether to concatenate term outputs into a single tensor."""

    enable_noise: bool = False
    """If ``True``, apply each term's noise setting before scaling."""

    history_length: int = 1
    """Number of timesteps to retain for history stacking (``1`` disables history)."""


@dataclass(frozen=True)
class ObservationManagerCfg:
    """Configuration for the observation manager."""

    groups: dict[str, ObsGroupCfg] = field(default_factory=dict)
    """Mapping of group name to its configuration."""

    clip_observations: float = 100.0
    """Global observation clipping threshold (applied to all non-image observations)."""
