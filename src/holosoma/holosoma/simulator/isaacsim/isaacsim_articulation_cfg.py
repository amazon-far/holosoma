from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg

from holosoma.simulator.isaacsim.converters import (
    physics_to_collision_props,
    physics_to_mass_props,
    physics_to_rigid_body_props,
)
from holosoma.utils.path import resolve_asset_path, urdf_conversion_cache_dir
from .api import ISAACSIM5, IsaacLabApi

if TYPE_CHECKING:
    from holosoma.config_types.robot import RobotConfig


def _per_joint_map(joint_names: Sequence[str], values: Sequence[float], property_name: str) -> dict[str, float]:
    """Map a DOF property by exact joint name without relying on source or simulator ordering."""
    if len(joint_names) != len(values):
        raise ValueError(
            f"Robot DOF property '{property_name}' has {len(values)} values for {len(joint_names)} joint names."
        )
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("Robot DOF names must be unique to build per-joint Isaac Lab actuator properties.")
    return {joint_name: float(values[index]) for index, joint_name in enumerate(joint_names)}


def build_robot_articulation_cfg(robot_config: RobotConfig, *, api: IsaacLabApi = ISAACSIM5) -> ArticulationCfg:
    """Assemble the robot ``ArticulationCfg`` from ``robot_config``.

    Resolves the asset root, maps the shared ``link_physics`` onto IsaacLab rigid/collision/mass
    props (via the same converters scene objects use), builds the URDF-convert or USD spawn cfg,
    the initial-state cfg, and one ``IdealPDActuatorCfg`` covering every DOF, with the shared soft joint limit factor. Pure config -> cfg: the caller instantiates the ``Articulation``.
    """
    robot_asset_cfg = robot_config.asset

    # PhysX solver knobs (damping, velocity caps) come from the shared link_physics.physx via the
    # converter objects use (physics_to_rigid_body_props), so a robot link and a scene object map
    # the physx sub-config identically. fixed=False (the robot is a free-base articulation, never
    # a kinematic body); None link_physics gives the converter's PhysXPhysicsConfig defaults.
    # @apply_nested on the articulation, so it reaches every link (body_names='.*' semantics).
    robot_link_physics = robot_asset_cfg.link_physics
    robot_rigid_props = physics_to_rigid_body_props(robot_link_physics, fixed=False)

    # Collision offsets (contact/rest offset, torsional patch) from link_physics.isaacsim, if set,
    # via the converter objects use. @apply_nested over every link collider. Gated on the isaacsim
    # sub-config being present (not just link_physics): physics_to_collision_props always returns a
    # non-None cfg, and passing it would run modify_collision_properties (stamping an empty
    # PhysxCollisionAPI on every robot collider) even when no offset is configured. Gating keeps it
    # None, and thus byte-for-byte the prior spawn, unless an offset is set.
    robot_collision_props = (
        physics_to_collision_props(robot_link_physics)
        if robot_link_physics is not None and robot_link_physics.isaacsim is not None
        else None
    )

    # density (a shared core field) reaches the robot links the way it reaches objects:
    # physics_to_mass_props gives a MassPropertiesCfg, @apply_nested over every link.
    # link_physics.mass is validator-rejected (_validate_link_physics), so only density flows,
    # matching how MuJoCo (geom.density) honors it on a robot
    # link. physics_to_mass_props returns None when neither mass nor density is set, so this is
    # byte-for-byte the prior spawn for a robot with no link_physics density (e.g. g1).
    robot_mass_props = physics_to_mass_props(robot_link_physics)

    robot_articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=robot_asset_cfg.enable_self_collisions,
        # NOTE: (4, 0) -> (8, 4) necessary for reproducing FAR-tracking-implementation
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )

    spawn: sim_utils.UrdfFileCfg | sim_utils.UsdFileCfg
    if robot_asset_cfg.usd_file is None:
        # convert from urdf dynamically
        full_urdf_path = resolve_asset_path(robot_asset_cfg.urdf_file, robot_asset_cfg.asset_root)

        # Get local rank to avoid race conditions in multi-GPU setups
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        usd_conversion_dir = os.path.join(urdf_conversion_cache_dir(full_urdf_path), f"converted_rank{local_rank}")
        merge_fixed_joints = robot_asset_cfg.collapse_fixed_joints
        if merge_fixed_joints and not api.importer_preserves_fixed_joint_markers:
            # Sim 6.2's preprocessing still removes marked fixed joints. Preserve the
            # configured contact frames until its importer honors dont_collapse.
            merge_fixed_joints = not any(
                joint.get("type") == "fixed" and joint.get("dont_collapse") == "true"
                for joint in ET.parse(full_urdf_path).getroot().findall("joint")
            )

        spawn = sim_utils.UrdfFileCfg(
            usd_dir=usd_conversion_dir,
            asset_path=full_urdf_path,
            fix_base=robot_asset_cfg.fix_base_link,
            merge_fixed_joints=merge_fixed_joints,
            replace_cylinders_with_capsules=robot_asset_cfg.replace_cylinder_with_capsule,
            force_usd_conversion=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0,
                    damping=0,
                ),
                target_type="none",
            ),
            activate_contact_sensors=True,
            rigid_props=robot_rigid_props,
            collision_props=robot_collision_props,
            mass_props=robot_mass_props,
            articulation_props=robot_articulation_props,
        )
    else:
        spawn = sim_utils.UsdFileCfg(
            usd_path=resolve_asset_path(robot_asset_cfg.usd_file, robot_asset_cfg.asset_root),
            activate_contact_sensors=True,
            rigid_props=robot_rigid_props,
            collision_props=robot_collision_props,
            mass_props=robot_mass_props,
            articulation_props=robot_articulation_props,
        )

    default_joint_angles = copy.deepcopy(robot_config.init_state.default_joint_angles)
    init_state = ArticulationCfg.InitialStateCfg(
        pos=tuple(robot_config.init_state.pos),
        joint_pos={joint_name: joint_angle for joint_name, joint_angle in default_joint_angles.items()},
        joint_vel={".*": 0.0},
    )

    dof_names_list = copy.deepcopy(robot_config.dof_names)
    dof_effort_limit_list = robot_config.dof_effort_limit_list
    dof_vel_limit_list = robot_config.dof_vel_limit_list
    dof_armature_list = robot_config.dof_armature_list
    dof_joint_friction_list = robot_config.dof_joint_friction_list

    if dof_names_list:
        zero_gains = {joint_name: 0.0 for joint_name in dof_names_list}
        actuators = {
            "all_dofs": IdealPDActuatorCfg(
                joint_names_expr=dof_names_list,
                effort_limit=_per_joint_map(dof_names_list, dof_effort_limit_list, "effort_limit"),
                velocity_limit=_per_joint_map(dof_names_list, dof_vel_limit_list, "velocity_limit"),
                stiffness=zero_gains,
                damping=zero_gains.copy(),
                armature=_per_joint_map(dof_names_list, dof_armature_list, "armature"),
                friction=_per_joint_map(dof_names_list, dof_joint_friction_list, "friction"),
            )
        }
    else:
        actuators = {}

    return ArticulationCfg(
        prim_path=f"{api.env_regex}/Robot",
        spawn=spawn,
        init_state=init_state,
        actuators=actuators,
        soft_joint_pos_limit_factor=0.9,
    )


def build_contact_sensor_cfg(history_length: int, debug_vis: bool, *, api: IsaacLabApi = ISAACSIM5) -> ContactSensorCfg:
    """Build the robot ContactSensorCfg covering every robot body prim.

    ``debug_vis`` draws the red contact spheres; the caller passes ``SimulatorInitConfig.debug_viz``
    so they follow the same switch as the other debug overlays. Isaac Lab applies it in
    ``SensorBase.__init__`` and again in ``_initialize_impl`` whenever the marker handle is unset, so
    a ``set_debug_vis(False)`` issued between construction and scene initialization gets undone --
    sweep after the first reset if you need to toggle the markers at runtime.
    """
    return ContactSensorCfg(
        prim_path=f"{api.env_regex}/Robot/.*",
        history_length=history_length,
        update_period=0.005,
        # Holosoma consumes net_forces_w and net_forces_w_history only. Tracking the four
        # air/contact-duration tensors adds per-step work without serving a production consumer.
        track_air_time=False,
        force_threshold=10.0,
        debug_vis=debug_vis,
    )
