"""Observation configuration types for holosoma_inference."""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ObservationConfig:
    """Observation space configuration.

    Defines which observations are used, their dimensions,
    scaling factors, history lengths, etc. for policy inference.
    """

    obs_dict: dict[str, list[str]]
    """Maps observation group names to lists of observation components.

    Each key represents an observation group (e.g., "actor_obs", "critic_obs"),
    and each value is a list of observation component names that belong to that group.

    Example:
        {"actor_obs": ["base_ang_vel", "projected_gravity", "dof_pos"]}
    """

    obs_dims: dict[str, int]
    """Dimension of each observation component.

    Maps each observation component name to its dimensionality.

    Example:
        {"base_ang_vel": 3, "dof_pos": 29, "actions": 29}
    """

    obs_scales: dict[str, float]
    """Scaling factor applied to each observation component.

    Maps each observation component name to a scaling factor that will be
    multiplied with the raw observation values during preprocessing.

    Example:
        {"base_ang_vel": 0.25, "dof_vel": 0.05, "projected_gravity": 1.0}
    """

    history_length_dict: dict[str, int]
    """Number of timesteps to keep in history for each observation group.

    Maps each observation group name to the number of historical timesteps
    to maintain. A value of 1 means only the current observation is used.

    Example:
        {"actor_obs": 1, "critic_obs": 3}
    """
