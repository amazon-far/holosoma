"""Physics configuration converters for IsaacSim.

This module provides conversion functions between the unified physics configuration
system and IsaacLab's specific configuration types.
"""

import isaaclab.sim as sim_utils
from holosoma.config_types.scene import PhysicsConfig, PhysXPhysicsConfig


def physics_to_mass_props(physics: PhysicsConfig | None) -> sim_utils.MassPropertiesCfg | None:
    """Convert PhysicsConfig to IsaacLab MassPropertiesCfg.

    Parameters
    ----------
    physics : PhysicsConfig or None
        PhysicsConfig instance with unified physics properties.

    Returns
    -------
    sim_utils.MassPropertiesCfg or None
        MassPropertiesCfg instance for IsaacLab mass properties, or None if no mass/density specified.

    Raises
    ------
    TypeError
        If physics is not a PhysicsConfig instance or None.
    """
    # Handle None case
    if physics is None:
        return None

    # Strict type checking
    if not isinstance(physics, PhysicsConfig):
        raise TypeError(f"Expected PhysicsConfig instance, got {type(physics)}")

    # Check if we have mass or density to set
    if physics.mass is None and physics.density is None:
        return None

    # Create mass properties config
    mass_props = sim_utils.MassPropertiesCfg(mass=physics.mass, density=physics.density)

    return mass_props


def physics_to_collision_props(physics: PhysicsConfig | None) -> sim_utils.CollisionPropertiesCfg:
    """Convert PhysicsConfig to IsaacLab CollisionPropertiesCfg.

    Always returns a (non-None) cfg so the spawner stamps ``CollisionAPI`` on the asset. The
    contact/rest offsets come from the IsaacSim sub-config (``physxCollision:contactOffset``/
    ``restOffset``); ``None`` leaves them unset, which IsaacLab treats as "keep the asset's
    authored value". The contact >= rest invariant is validated on ``IsaacSimPhysicsConfig``.

    Parameters
    ----------
    physics : PhysicsConfig or None
        Unified physics config; its ``isaacsim`` sub-config carries the offsets (if any).

    Returns
    -------
    sim_utils.CollisionPropertiesCfg
        Collision props, with contact/rest offsets set iff the IsaacSim sub-config set them.
    """
    isaacsim = physics.isaacsim if physics is not None else None
    if isaacsim is None:
        return sim_utils.CollisionPropertiesCfg()
    return sim_utils.CollisionPropertiesCfg(
        contact_offset=isaacsim.contact_offset,
        rest_offset=isaacsim.rest_offset,
        torsional_patch_radius=isaacsim.torsional_patch_radius,
    )


def physics_to_rigid_body_props(physics: PhysicsConfig | None, fixed: bool) -> sim_utils.RigidBodyPropertiesCfg:
    """Convert PhysicsConfig to IsaacLab RigidBodyPropertiesCfg.

    Parameters
    ----------
    physics : PhysicsConfig or None
        PhysicsConfig instance with unified physics properties.
    fixed : bool
        Whether the body is static. Maps to ``kinematic_enabled`` — a fixed body is spawned as a
        PhysX kinematic rigid body (immovable under gravity/contact, pose-settable).

    Returns
    -------
    sim_utils.RigidBodyPropertiesCfg
        RigidBodyPropertiesCfg instance for IsaacLab.

    Raises
    ------
    TypeError
        If physics is not a PhysicsConfig instance or None.
    """
    # Handle None case
    if physics is None:
        physics = PhysicsConfig()

    # Strict type checking - should be PhysicsConfig by now
    if not isinstance(physics, PhysicsConfig):
        raise TypeError(
            f"Expected PhysicsConfig instance, got {type(physics)}. "
            f"Physics config should be converted to PhysicsConfig in __post_init__ methods."
        )

    # PhysX solver knobs (damping / velocity limits) come from the shared physx sub-config;
    # None uses its defaults.
    physx = physics.physx or PhysXPhysicsConfig()

    # disable_gravity, retain_accelerations, and max_depenetration_velocity are intentionally-fixed
    # tuning values, not currently exposed via PhysicsConfig. They deliberately override IsaacLab's
    # defaults (all None); max_depenetration_velocity=1.0 is a deliberate 1.0 m/s cap on the velocity
    # used to push overlapping bodies apart.
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=fixed,
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=physx.linear_damping,
        angular_damping=physx.angular_damping,
        max_linear_velocity=physx.max_linear_velocity,
        max_angular_velocity=physx.max_angular_velocity,
        max_depenetration_velocity=1.0,
        # Note: density is handled by MaterialPropertiesCfg, not RigidBodyPropertiesCfg
    )

    return rigid_props
