"""Test-only robot presets, not shipped on the production CLI.

These drive the cross-backend test harnesses (``behavior_assert``). They are kept out of
``holosoma.config_values.robot`` so the production ``robot:`` subcommand menu lists only shipped
robots.

:data:`onelink_box` is a single-link, 0-DOF, floating-base robot structurally identical to the
``small_box`` scene object but loaded through the robot path. ``behavior_assert`` applies a robot
``link_physics`` to it and compares its physical outcome against an object twin carrying the same
``PhysicsConfig``.

:data:`test_arm_fixed` is a 3-DOF fixed-base arm (``asset.fix_base_link=True``): hinge, hinge,
slide. ``fixed_base_robot_assert`` and the fixed-base e2e test load it to prove a welded robot
spawns, steps, resets and is observed like a static scene object on every backend.

``behavior_assert.main`` calls :func:`register` so ``--robot onelink-box`` resolves through the
tyro path. Core does not import this module; tests register into it.
"""

from __future__ import annotations

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg, JointControllerCfg
from holosoma.config_types.robot import (
    RobotAssetConfig,
    RobotConfig,
    RobotInitState,
)
from holosoma.config_types.scene import PhysicsConfig

# --------------------------------------------------------------------------------------------
# One-link floating-base "robot": small_box loaded through the robot path. 0 DOF, no actuators.
# --------------------------------------------------------------------------------------------

# Spawn the base at 0.6 m so a drop-from-rest scenario has clearance (matches the box scenarios).
_INIT_STATE = RobotInitState(
    position=[0.0, 0.0, 0.6],
    default_joint_positions={},
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
        dof_names=[],
        dof_pos_limits={},
        dof_vel_limits={},
        dof_effort_limits={},
        dof_armature={},
        dof_joint_friction={},
        body_names=["baseLink"],
        init_state=_INIT_STATE,
        asset=_onelink_asset(link_physics),
        body_tags={"#base": ["baseLink"]},
    )


# Bare one-link robot, no link_physics (asset-authored physics). Used as the base for
# material-carrying variants.
onelink_box = _onelink_robot(link_physics=None)

# --------------------------------------------------------------------------------------------
# Fixed-base 3-DOF test arm: base_link welded to the world (fix_base_link=True).
# --------------------------------------------------------------------------------------------

# Chain base_link -> shoulder_joint (hinge z) -> link1 -> elbow_joint (hinge y) -> link2 ->
# slide_joint (slide x) -> slider; see data/robots/test_arm. The limit/effort/velocity lists equal
# the URDF <limit> values (IsaacGym asserts them within 1e-5). Gain keys are the exact DOF names.
# The base is mounted at a 90 deg yaw (wxyz) so the spawn checks exercise the wxyz -> xyzw path on every
# backend; a yaw keeps gravity along -z, so the joints still droop and the tip still falls.
TEST_ARM_YAW_QUAT_WXYZ = [0.7071068, 0.0, 0.0, 0.7071068]
test_arm_fixed = RobotConfig(
    dof_names=["shoulder_joint", "elbow_joint", "slide_joint"],
    body_names=["base_link", "link1", "link2", "slider"],
    dof_pos_limits=dict(
        zip(["shoulder_joint", "elbow_joint", "slide_joint"], zip([-1.5, -1.5, 0.0], [1.5, 1.5, 0.05]))
    ),
    dof_vel_limits={"shoulder_joint": 10.0, "elbow_joint": 10.0, "slide_joint": 1.0},
    dof_effort_limits={"shoulder_joint": 20.0, "elbow_joint": 20.0, "slide_joint": 50.0},
    dof_armature={"shoulder_joint": 0.01, "elbow_joint": 0.01, "slide_joint": 0.001},
    dof_joint_friction={"shoulder_joint": 0.0, "elbow_joint": 0.0, "slide_joint": 0.0},
    init_state=RobotInitState(
        position=[0.0, 0.0, 0.5],
        orientation=TEST_ARM_YAW_QUAT_WXYZ,
        default_joint_positions={"shoulder_joint": 0.0, "elbow_joint": 0.0, "slide_joint": 0.0},
    ),
    asset=RobotAssetConfig(
        asset_root="@holosoma/data/robots",
        collapse_fixed_joints=False,
        replace_cylinder_with_capsule=False,
        flip_visual_attachments=False,
        thickness=0.01,
        urdf_file="test_arm/test_arm.urdf",
        usd_file=None,
        xml_file="test_arm/test_arm.xml",
        robot_type="test_arm",
        enable_self_collisions=False,
        default_dof_drive_mode=3,
        fix_base_link=True,
    ),
    body_tags={"#base": ["base_link"]},
)

TEST_ARM_CONTROLLER = JointControllerCfg(
    stiffness={"shoulder_joint": 20.0, "elbow_joint": 20.0, "slide_joint": 200.0},
    damping={"shoulder_joint": 1.0, "elbow_joint": 1.0, "slide_joint": 5.0},
)
test_arm_action = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            scale=0.25,
            clip=(-100.0, 100.0),
            controller=TEST_ARM_CONTROLLER,
        )
    }
)

# The scenario harness sets the material-carrying link_physics per scenario via dataclasses.replace,
# so the registry needs only the bare preset.
TEST_ROBOT_PRESETS = {
    "onelink-box": onelink_box,
    "test-arm-fixed": test_arm_fixed,
}


def register() -> None:
    """Merge the test-only robot presets into ``holosoma.config_values.robot.DEFAULTS``.

    Idempotent. tyro reads ``robot.DEFAULTS`` lazily at ``tyro.cli()`` time, so a key registered
    here resolves through the production tyro path.
    """
    import holosoma.config_values.robot as robot_values

    robot_values.DEFAULTS.update(TEST_ROBOT_PRESETS)
