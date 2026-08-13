"""Physics utilities for IsaacGym simulator.

This module provides utilities for applying physics properties to actors
in the IsaacGym simulation environment, including rigid shape properties
and mass configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaacgym import gymapi
from loguru import logger

if TYPE_CHECKING:
    from holosoma.config_types.scene import PhysicsConfig


def apply_physx_asset_options(target: Any, physics: PhysicsConfig | None) -> None:
    """Apply the load-time PhysX asset options (``density`` + the 4 ``physx`` solver knobs) from a
    ``PhysicsConfig`` onto ``target``.

    ``target`` is anything with the matching settable attributes: a ``gymapi.AssetOptions`` (robot
    path) or a ``urdf_scene_loader.AssetConfig`` (object path); both name the fields identically.
    This is the single source of the ``PhysicsConfig`` -> IsaacGym load-time mapping, so a robot link
    and a scene object map ``density``/``physx`` the same way. ``None`` on ``physics``/``physx`` (or
    an unset ``density``) leaves ``target``'s existing value untouched.

    Friction/restitution/mass are not here: they are post-create per-shape/per-body writes
    (``apply_rigid_shape_properties`` / ``apply_mass_from_config``), not load-time AssetOptions.
    """
    if physics is None:
        return
    if physics.density is not None:
        target.density = physics.density
    physx = physics.physx
    if physx is not None:
        target.linear_damping = physx.linear_damping
        target.angular_damping = physx.angular_damping
        target.max_linear_velocity = physx.max_linear_velocity
        target.max_angular_velocity = physx.max_angular_velocity


def apply_rigid_shape_properties(
    gym: gymapi.Gym, env_ptr: int, actor_handle: int, physics_config: PhysicsConfig, object_name: str
) -> None:
    """Apply rigid shape properties (friction, restitution, compliance).

    Applies IsaacGym-specific physics properties to the actor's rigid shapes,
    including friction coefficients, restitution, and compliance values.

    Parameters
    ----------
    gym : gymapi.Gym
        The IsaacGym API instance.
    env_ptr : int
        Environment handle for the IsaacGym environment.
    actor_handle : int
        Handle to the actor to modify.
    physics_config : PhysicsConfig
        The live per-object physics config; IsaacGym friction is read from its ``isaacgym``
        sub-config (``None`` => keep the asset's authored friction).
    object_name : str
        Name of the object for logging purposes.

    Raises
    ------
    RuntimeError
        If no rigid shape properties are found for the specified object.
    """
    # Get current rigid shape properties
    shape_props = gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)

    if not shape_props:
        raise RuntimeError(f"No rigid shape properties found for '{object_name}'")

    # Apply IsaacGym-specific friction properties if configured
    if hasattr(physics_config, "isaacgym") and physics_config.isaacgym is not None:
        gym_config = physics_config.isaacgym
        logger.debug(f"Applying IsaacGym friction properties to '{object_name}': {gym_config}")

        for i in range(len(shape_props)):
            # Apply 1:1 mapping to IsaacGym RigidShapeProperties
            shape_props[i].friction = gym_config.friction
            shape_props[i].rolling_friction = gym_config.rolling_friction
            shape_props[i].torsion_friction = gym_config.torsion_friction
            shape_props[i].restitution = gym_config.restitution
            shape_props[i].compliance = gym_config.compliance

        # Set the modified properties back to the actor
        gym.set_actor_rigid_shape_properties(env_ptr, actor_handle, shape_props)
        logger.debug(
            f"Applied IsaacGym friction properties to '{object_name}': "
            f"friction={gym_config.friction}, rolling={gym_config.rolling_friction}, "
            f"torsion={gym_config.torsion_friction}"
        )

    else:
        logger.debug(f"No IsaacGym-specific friction config for '{object_name}', using defaults")


def apply_mass_from_config(
    gym: gymapi.Gym, env_ptr: int, actor_handle: int, physics_config: PhysicsConfig, object_name: str
) -> None:
    """Apply mass or density from physics config.

    Applies mass properties to the actor based on the physics configuration,
    with priority given to explicit mass values over density calculations.

    Parameters
    ----------
    gym : gymapi.Gym
        The IsaacGym API instance.
    env_ptr : int
        Environment handle for the IsaacGym environment.
    actor_handle : int
        Handle to the actor to modify.
    physics_config : PhysicsConfig
        The live per-object physics config; ``mass`` (priority 1) and ``density`` (priority 2)
        are read from its core fields.
    object_name : str
        Name of the object for logging purposes.

    Raises
    ------
    RuntimeError
        If no rigid body properties are found for the specified object.
    """
    # Get current rigid body properties
    body_props = gym.get_actor_rigid_body_properties(env_ptr, actor_handle)
    if not body_props:
        raise RuntimeError(f"No rigid body properties found for '{object_name}', cannot apply mass config")

    # Priority 1: Direct mass override
    if physics_config.mass is not None:
        target_mass = physics_config.mass
        logger.debug(f"Setting explicit mass for '{object_name}': {target_mass}")

        for prop in body_props:
            prop.mass = target_mass

        gym.set_actor_rigid_body_properties(env_ptr, actor_handle, body_props, recomputeInertia=True)
        logger.debug(f"Applied explicit mass to '{object_name}': {target_mass}")
        return

    # Priority 2: Density-based calculation
    if physics_config.density is not None:
        current_mass = sum(prop.mass for prop in body_props)

        if current_mass < 1e-6:
            # URDF has very low mass, log warning but keep original values
            target_density = physics_config.density
            logger.warning(
                f"URDF has very low mass ({current_mass}) for '{object_name}' "
                f"with density config {target_density}. Keeping original URDF mass values."
            )
        else:
            # Density was applied during asset loading, use existing mass
            logger.debug(f"Using density-calculated mass for '{object_name}': {current_mass}")
        return

    # Priority 3: No mass/density config - use URDF values
    current_mass = sum(prop.mass for prop in body_props)
    logger.debug(f"Using URDF mass for '{object_name}': {current_mass}")
