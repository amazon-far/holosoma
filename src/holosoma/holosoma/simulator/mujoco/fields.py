"""Field preparation utilities for MuJoCo per-environment operations.

This module provides decorators and utilities for declaring and preparing MuJoCo
fields that need per-environment expansion (e.g., body_mass, geom_friction).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from holosoma.config_types.simulator import MujocoBackend
from holosoma.simulator.shared.field_decorators import MUJOCO_FIELD_ATTR


def collect_required_fields(*managers: Any) -> list[str]:
    """Scan managers for functions decorated with @required_field and collect field names.

    This utility scans manager setup terms to find functions with mujoco_field metadata
    attached by the @required_field decorator. Only fields from configured functions
    are collected, avoiding preparation of unused fields.

    Parameters
    ----------
    *managers : Any
        Manager instances to scan (e.g., RandomizationManager, ObservationManager, etc.).
        Managers must have a _setup_funcs attribute containing resolved functions.

    Returns
    -------
    list[str]
        List of unique MuJoCo field names required by the configured functions.

    Examples
    --------
    >>> from holosoma.simulator.mujoco.fields import collect_required_fields
    >>> fields = collect_required_fields(randomization_manager, observation_manager)
    >>> # fields might be ["body_mass", "geom_friction", "body_ipos"]
    """
    fields = set()

    for manager in managers:
        if manager is None:
            continue

        # Check if manager has setup functions (manager-based pattern)
        if hasattr(manager, "_setup_funcs"):
            for func in manager._setup_funcs.values():
                if hasattr(func, MUJOCO_FIELD_ATTR):
                    fields.add(getattr(func, MUJOCO_FIELD_ATTR))

    return list(fields)


def prepare_manager_fields(simulator, **managers) -> None:
    """Scan managers for field requirements and prepare them.

    This function collects MuJoCo field requirements from all configured manager
    functions decorated with @mujoco.required_field, then prepares those fields
    for per-environment operations.

    Parameters
    ----------
    simulator : MuJoCo
        The MuJoCo simulator instance.
    **managers : Any
        Keyword arguments of manager instances to scan (e.g.,
        randomization_manager=..., observation_manager=..., reward_manager=...).
        Managers must have a _setup_funcs attribute for field discovery.

    Notes
    -----
    - Automatically discovers fields from @mujoco.required_field decorators
    - Only prepares fields from configured/active manager functions
    - Delegates actual field preparation to prepare_fields()

    Examples
    --------
    >>> prepare_manager_fields(
    ...     simulator,
    ...     randomization_manager=rand_mgr,
    ...     observation_manager=obs_mgr,
    ...     reward_manager=rew_mgr,
    ... )
    """
    fields = collect_required_fields(*managers.values())
    if fields:
        logger.info(f"Discovered {len(fields)} required fields from managers: {fields}")
        prepare_fields(simulator, fields)
    else:
        logger.info("Discovered ZERO MuJoCo fields required by managers")


def prepare_fields(simulator, field_names: list[str]) -> None:
    """Prepare MuJoCo model fields for per-environment operations.

    This function expands MuJoCo model fields to support per-environment physics
    parameters when using the Warp backend. It handles all backend-specific
    setup including cache invalidation.

    Parameters
    ----------
    simulator : MuJoCo
        The MuJoCo simulator instance.
    field_names : list[str]
        List of MuJoCo field names to expand for per-environment use.
        Examples: 'body_mass', 'body_ipos', 'geom_friction', etc.

    Notes
    -----
    - No-op for ClassicBackend (single environment)
    - For WarpBackend: expands fields and clears bridge cache
    - Safe to call with empty list (no-op)
    """
    if not field_names:
        return

    # Only needed for WarpBackend with multiple environments
    if simulator.simulator_config.mujoco_backend != MujocoBackend.WARP:
        return

    logger.info(f"Preparing {len(field_names)} fields for per-environment operations")

    # Expand model fields (internal implementation detail)
    from holosoma.simulator.mujoco.backends.randomization import expand_model_fields

    expand_model_fields(simulator.backend.mjw_model, nworld=simulator.num_envs, fields_to_expand=field_names)

    # Clear bridge cache to reflect expanded arrays (internal implementation detail)
    simulator.backend.warp_model_bridge.clear_cache()

    logger.info("Field expansion complete")
