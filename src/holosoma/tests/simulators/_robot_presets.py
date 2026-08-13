"""Test-only robot presets, not shipped on the production CLI.

These drive the cross-backend test harnesses (``behavior_assert``). They are kept out of
``holosoma.config_values.robot`` so the production ``robot:`` subcommand menu lists only shipped
robots.

:data:`onelink_box` is a single-link, 0-DOF, floating-base robot structurally identical to the
``small_box`` scene object but loaded through the robot path. ``behavior_assert`` applies a robot
``link_physics`` to it and compares its physical outcome against an object twin carrying the same
``PhysicsConfig``.

``behavior_assert.main`` calls :func:`register` so ``--robot onelink-box`` resolves through the
tyro path. Core does not import this module; tests register into it.
"""

from __future__ import annotations

from holosoma.config_types.robot import (
    RobotAssetConfig,
    RobotConfig,
    RobotControlConfig,
    RobotInitState,
)
from holosoma.config_types.scene import PhysicsConfig

# --------------------------------------------------------------------------------------------
# One-link floating-base "robot": small_box loaded through the robot path. 0 DOF, no actuators.
# --------------------------------------------------------------------------------------------

# Empty control config: no actuated DOFs. The behavior harness drives the sim directly and does
# not build PD gains, so this is structural only.
_EMPTY_CONTROL = RobotControlConfig(
    control_type="none",
    stiffness={},
    damping={},
    action_scale=1.0,
    action_clip_value=100.0,
    clip_actions=False,
    clip_torques=False,
)

# Spawn the base at 0.6 m so a drop-from-rest scenario has clearance (matches the box scenarios).
_INIT_STATE = RobotInitState(
    pos=[0.0, 0.0, 0.6],
    rot=[1.0, 0.0, 0.0, 0.0],  # wxyz
    lin_vel=[0.0, 0.0, 0.0],
    ang_vel=[0.0, 0.0, 0.0],
    default_joint_angles={},
)


def _onelink_asset(link_physics: PhysicsConfig | None) -> RobotAssetConfig:
    """The one-link box asset, optionally carrying a robot ``link_physics``.

    ``usd_file=None`` so IsaacSim converts the same URDF the other backends load.
    """
    return RobotAssetConfig(
        asset_root="@holosoma/data/robots",
        collapse_fixed_joints=True,
        replace_cylinder_with_capsule=False,
        flip_visual_attachments=False,
        armature=0.0,
        thickness=0.01,
        urdf_file="onelink_box/onelink_box.urdf",
        usd_file=None,
        xml_file="onelink_box/onelink_box.xml",
        robot_type="onelink_box",
        enable_self_collisions=False,
        default_dof_drive_mode=3,
        fix_base_link=False,  # floating base
        link_physics=link_physics,
    )


def _onelink_robot(link_physics: PhysicsConfig | None) -> RobotConfig:
    """A minimal 0-DOF RobotConfig wrapping the one-link box asset.

    Every DOF-indexed list is empty (no actuated joints); ``body_names=["baseLink"]`` is the single
    link. Observation and action dims are 0 (the harness builds no policy), so these are structural
    only.
    """
    return RobotConfig(
        num_bodies=1,
        dof_obs_size=0,
        algo_obs_dim_dict={},
        actions_dim=0,
        policy_obs_dim=0,
        critic_obs_dim=0,
        contact_pairs_multiplier=1,
        key_bodies=["baseLink"],
        num_feet=0,
        foot_body_name="baseLink",
        foot_height_name="baseLink",
        knee_name="baseLink",
        torso_name="baseLink",
        dof_names=[],
        upper_dof_names=[],
        upper_left_arm_dof_names=[],
        upper_right_arm_dof_names=[],
        lower_dof_names=[],
        has_torso=False,
        has_upper_body_dof=False,
        left_ankle_dof_names=[],
        right_ankle_dof_names=[],
        knee_dof_names=[],
        hips_dof_names=[],
        dof_pos_lower_limit_list=[],
        dof_pos_upper_limit_list=[],
        dof_vel_limit_list=[],
        dof_effort_limit_list=[],
        dof_armature_list=[],
        dof_joint_friction_list=[],
        body_names=["baseLink"],
        terminate_after_contacts_on=[],
        penalize_contacts_on=[],
        init_state=_INIT_STATE,
        randomize_link_body_names=["baseLink"],
        control=_EMPTY_CONTROL,
        asset=_onelink_asset(link_physics),
    )


# Bare one-link robot, no link_physics (asset-authored physics). Used as the base for
# material-carrying variants.
onelink_box = _onelink_robot(link_physics=None)

# The scenario harness sets the material-carrying link_physics per scenario via dataclasses.replace,
# so the registry needs only the bare preset.
TEST_ROBOT_PRESETS = {
    "onelink-box": onelink_box,
}


def register() -> None:
    """Merge the test-only robot presets into ``holosoma.config_values.robot.DEFAULTS``.

    Idempotent. tyro reads ``robot.DEFAULTS`` lazily at ``tyro.cli()`` time, so a key registered
    here resolves through the production tyro path.
    """
    import holosoma.config_values.robot as robot_values

    robot_values.DEFAULTS.update(TEST_ROBOT_PRESETS)
