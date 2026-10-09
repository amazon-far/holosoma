"""Field requirement decorators for simulator backends.

This module provides lightweight decorators for declaring field requirements
across different simulator backends. It has ZERO dependencies and can be safely
imported in any environment (IsaacGym, IsaacSim, MuJoCo).
"""

from __future__ import annotations

from typing import Any, Callable

# Module-level constant for the attribute name
MUJOCO_FIELD_ATTR = "mujoco_field"


def mujoco_required_field(field: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a manager function as requiring a MuJoCo field for per-environment operations.

    This decorator attaches metadata to manager functions (randomization, observation,
    reward, etc.) indicating which MuJoCo field they need prepared before execution.
    The simulator uses this metadata to automatically prepare the required fields via
    prepare_manager_fields().

    Parameters
    ----------
    field : str
        The MuJoCo field name that this function operates on.
        Examples: "body_mass", "geom_friction", "body_ipos", "joint_pos"

    Returns
    -------
    Callable
        The decorated function with mujoco_field metadata attached.

    Examples
    --------
    >>> from holosoma.simulator import mujoco_required_field
    >>>
    >>> # Randomization term
    >>> @mujoco_required_field("body_mass")
    >>> def randomize_mass_startup(env, env_ids, **params):
    >>>     # Randomize body masses per environment
    >>>     pass
    >>>
    >>> # Observation term (hypothetical)
    >>> @mujoco_required_field("joint_pos")
    >>> def compute_joint_observation(env):
    >>>     # Compute observation from joint positions
    >>>     pass
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, MUJOCO_FIELD_ATTR, field)
        return func

    return decorator


__all__ = ["mujoco_required_field"]
