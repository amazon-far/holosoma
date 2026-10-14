"""Config-layer unit tests for ``RobotConfig`` construction-time validation (pure, no simulator).

Every per-DOF physical table on ``RobotConfig`` is indexed by ``dof_names``; these tests pin those
validators, the ``#tag`` tables, and the ``RobotInitState`` shape checks. Controller behavior is
tested with the action terms that own it.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pydantic
import pytest
import torch

from holosoma.config_types.robot import (
    RobotAssetConfig,
    RobotConfig,
    RobotInitState,
)
from holosoma.config_values.robot import ROBOT_REGISTRY, g1_29dof, t1_29dof_waist_wrist
from holosoma.utils.name_tags import expand_joint_value_map, expand_name_spec
from holosoma.utils.rotations import quat_rotate

pytestmark = pytest.mark.no_sim


def _replace_robot(robot: RobotConfig, **overrides: Any) -> RobotConfig:
    return dataclasses.replace(robot, **overrides)


def _minimal_robot(**overrides: Any) -> RobotConfig:
    """A 0-DOF, one-body robot; the smallest config the validators accept."""
    kwargs: dict[str, Any] = {
        "dof_names": [],
        "body_names": ["base"],
        "init_state": RobotInitState(position=[0.0, 0.0, 0.5]),
        "asset": RobotAssetConfig(
            asset_root="x",
            collapse_fixed_joints=True,
            replace_cylinder_with_capsule=False,
            flip_visual_attachments=False,
            thickness=0.01,
            urdf_file="r.urdf",
            usd_file=None,
            xml_file="r.xml",
            robot_type="r",
            enable_self_collisions=False,
            default_dof_drive_mode=3,
            fix_base_link=False,
        ),
    }
    kwargs.update(overrides)
    return RobotConfig(**kwargs)


def _synthetic_robot(dof_names: list[str], **overrides: Any) -> RobotConfig:
    """A robot with complete physical-property maps for the given DOF names."""
    kwargs: dict[str, Any] = {
        "dof_names": dof_names,
        "dof_pos_limits": dict.fromkeys(dof_names, (-1.0, 1.0)),
        "dof_vel_limits": dict.fromkeys(dof_names, 10.0),
        "dof_effort_limits": dict.fromkeys(dof_names, 10.0),
        "dof_armature": dict.fromkeys(dof_names, 0.0),
        "dof_joint_friction": dict.fromkeys(dof_names, 0.0),
        "init_state": RobotInitState(position=[0.0, 0.0, 0.5], default_joint_positions=dict.fromkeys(dof_names, 0.0)),
    }
    kwargs.update(overrides)
    return _minimal_robot(**kwargs)


# --------------------------------------------------------------------------------------------
# Shipped presets and the basic surface
# --------------------------------------------------------------------------------------------


def _mirror(name: str) -> str:
    return name.replace("left", "right").replace("Left", "Right")


_HUMANOIDS = sorted(name for name in ROBOT_REGISTRY if "#feet" in ROBOT_REGISTRY[name].body_tags)


@pytest.mark.parametrize("name", _HUMANOIDS)
def test_every_shipped_humanoid_declares_the_tags_its_consumers_reference(name: str) -> None:
    """The body/joint tags the shipped loco/WBT terms and symmetry augmentation resolve."""
    robot = ROBOT_REGISTRY[name]
    assert {"#base", "#feet", "#feet_height"} <= set(robot.body_tags)
    assert len(expand_name_spec(["#base"], robot.body_tags, robot.body_names, kind="body")) == 1
    feet = expand_name_spec(["#feet"], robot.body_tags, robot.body_names, kind="body")
    feet_height = expand_name_spec(["#feet_height"], robot.body_tags, robot.body_names, kind="body")
    # One gait-phase column per foot (feet_phase), one left and one right; the raycast points are separate bodies.
    assert len(feet) == len(feet_height) == 2
    assert [_mirror(n) for n in (feet[0], feet_height[0])] == [feet[1], feet_height[1]]
    assert not set(feet) & set(feet_height)
    assert {"#mirror_left", "#mirror_right", "#mirror_flip_sign"} <= set(robot.joint_tags)
    left = expand_name_spec(["#mirror_left"], robot.joint_tags, robot.dof_names, kind="joint")
    right = expand_name_spec(["#mirror_right"], robot.joint_tags, robot.dof_names, kind="joint")
    assert len(left) == len(right) > 0
    assert [_mirror(n) for n in left] == right  # index-corresponding sagittal pairs
    flip = expand_name_spec(["#mirror_flip_sign"], robot.joint_tags, robot.dof_names, kind="joint")
    assert flip and set(flip) <= set(robot.dof_names)


def test_shipped_humanoids_are_g1_and_t1() -> None:
    assert _HUMANOIDS == ["g1_29dof", "g1_29dof_w_object", "t1_29dof_waist_wrist"]


def test_yam_is_the_shipped_fixed_base_arm() -> None:
    """8 DOF (6 hinges + 2 finger slides), welded base, and DOF-keyed physical tables."""
    robot = ROBOT_REGISTRY["yam"]
    assert robot.asset.fix_base_link
    assert len(robot.dof_names) == 8 and robot.body_names[0] == "base"
    assert expand_name_spec(["#end_effector"], robot.body_tags, robot.body_names, kind="body") == ["gripper"]
    assert expand_name_spec(["#gripper"], robot.joint_tags, robot.dof_names, kind="joint") == ["joint7", "joint8"]
    defaults = robot.default_dof_positions()
    assert list(defaults.values())[6:] == [-0.003, -0.003]  # the tag-keyed finger default, in metres
    assert defaults["joint7"] == defaults["joint8"] == -0.003


def test_zero_dof_robot_constructs() -> None:
    robot = _minimal_robot()
    assert robot.dof_names == []
    assert not robot.asset.fix_base_link


def test_fix_base_link_is_the_only_base_mode_field() -> None:
    fixed = _replace_robot(g1_29dof, asset=dataclasses.replace(g1_29dof.asset, fix_base_link=True))
    assert not g1_29dof.asset.fix_base_link
    assert fixed.asset.fix_base_link
    assert "floating_base" not in dataclasses.asdict(g1_29dof)


def test_root_state_list_is_13_wide_pos_rot_linvel_angvel() -> None:
    init = g1_29dof.init_state
    assert init.root_state_list() == [*init.position, *init.orientation_xyzw, *init.lin_vel, *init.ang_vel]
    assert init.orientation == [1.0, 0.0, 0.0, 0.0] and init.orientation_xyzw == [0.0, 0.0, 0.0, 1.0]
    assert len(init.root_state_list()) == 13


def test_non_identity_orientation_moves_w_from_first_to_last() -> None:
    # A rotation about y (identity is invariant under any reorder that keeps w last, so it cannot pin this).
    init = RobotInitState(position=[0.0, 0.0, 0.5], orientation=[0.8, 0.0, 0.6, 0.0], lin_vel=[1.0, 2.0, 3.0])
    assert init.orientation_xyzw == [0.0, 0.6, 0.0, 0.8]
    assert init.root_state_list() == [0.0, 0.0, 0.5, 0.0, 0.6, 0.0, 0.8, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    # Both orders describe the same rotation: x rotated about y lands at [cos, 0, -sin] of the double angle.
    x_axis = torch.tensor([[1.0, 0.0, 0.0]])
    rotated = quat_rotate(torch.tensor([init.orientation_xyzw]), x_axis, w_last=True)
    assert torch.allclose(rotated, torch.tensor([[0.28, 0.0, -0.96]]), atol=1e-6)
    assert torch.allclose(rotated, quat_rotate(torch.tensor([init.orientation]), x_axis, w_last=False), atol=1e-6)


def test_duplicate_dof_names_rejected() -> None:
    names = list(g1_29dof.dof_names)
    names[1] = names[0]
    with pytest.raises(pydantic.ValidationError, match="dof_names has duplicate entries"):
        _replace_robot(g1_29dof, dof_names=names)


def test_empty_or_duplicate_body_names_rejected() -> None:
    with pytest.raises(pydantic.ValidationError, match="body_names must not be empty"):
        _replace_robot(g1_29dof, body_names=[])
    with pytest.raises(pydantic.ValidationError, match="body_names has duplicate entries"):
        _replace_robot(g1_29dof, body_names=[*g1_29dof.body_names[:-1], g1_29dof.body_names[0]])


@pytest.mark.parametrize(
    "field", ["dof_pos_limits", "dof_vel_limits", "dof_effort_limits", "dof_armature", "dof_joint_friction"]
)
def test_physical_property_maps_reject_overlapping_keys(field: str) -> None:
    value = (-1.0, 1.0) if field == "dof_pos_limits" else 1.0
    with pytest.raises(
        pydantic.ValidationError, match="keys '#knees' and 'left_knee_joint' both cover DOF 'left_knee_joint'"
    ):
        _replace_robot(g1_29dof, **{field: {"#knees": value, "left_knee_joint": value}})


def test_lower_limit_above_upper_rejected() -> None:
    with pytest.raises(pydantic.ValidationError, match="DOF 'left_knee_joint' has invalid range"):
        _replace_robot(g1_29dof, dof_pos_limits={"left_knee_joint": (2.0, 1.0)})


def test_position_bounds_are_a_pair() -> None:
    with pytest.raises(pydantic.ValidationError):
        _replace_robot(g1_29dof, dof_pos_limits={"left_knee_joint": (-1.0,)})


def test_zero_width_limit_is_allowed() -> None:
    _synthetic_robot(["knee"], dof_pos_limits={"knee": (0.0, 0.0)})


def test_default_positions_must_fall_within_configured_position_limits() -> None:
    with pytest.raises(pydantic.ValidationError, match=r"DOF 'knee' default 1\.5 is outside configured position range"):
        _synthetic_robot(
            ["knee"],
            dof_pos_limits={"knee": (-1.0, 1.0)},
            init_state=RobotInitState(position=[0.0, 0.0, 0.5], default_joint_positions={"knee": 1.5}),
        )


def test_implicit_zero_default_must_fall_within_configured_position_limits() -> None:
    with pytest.raises(
        pydantic.ValidationError, match=r"DOF 'slide' default 0\.0 is outside configured position range"
    ):
        _synthetic_robot(
            ["slide"],
            dof_pos_limits={"slide": (0.1, 1.0)},
            init_state=RobotInitState(position=[0.0, 0.0, 0.5]),
        )


def test_default_position_at_configured_limit_is_allowed() -> None:
    _synthetic_robot(
        ["knee"],
        dof_pos_limits={"knee": (-1.0, 1.0)},
        init_state=RobotInitState(position=[0.0, 0.0, 0.5], default_joint_positions={"knee": 1.0}),
    )


def test_default_position_is_not_checked_against_unknown_asset_limits() -> None:
    robot = _minimal_robot(
        dof_names=["slide"],
        init_state=RobotInitState(position=[0.0, 0.0, 0.5], default_joint_positions={"slide": 100.0}),
    )
    assert robot.default_dof_positions() == {"slide": 100.0}


def test_default_joint_positions_fill_missing_dofs_with_zero() -> None:
    robot = _replace_robot(
        g1_29dof, init_state=dataclasses.replace(g1_29dof.init_state, default_joint_positions={"#knees": 0.5})
    )
    defaults = robot.default_dof_positions()
    assert list(defaults) == robot.dof_names
    assert defaults == {
        name: 0.5 if name in ("left_knee_joint", "right_knee_joint") else 0.0 for name in robot.dof_names
    }
    assert robot.init_state.default_joint_positions == {"#knees": 0.5}
    with pytest.raises(pydantic.ValidationError, match="Unknown joint name 'ghost_joint'"):
        _replace_robot(
            g1_29dof, init_state=dataclasses.replace(g1_29dof.init_state, default_joint_positions={"ghost_joint": 0.0})
        )


def test_default_joint_positions_accept_tags() -> None:
    angles = {"#all": 0.0, "#knees": 0.5}
    with pytest.raises(pydantic.ValidationError, match="both cover DOF 'left_knee_joint'"):
        _replace_robot(g1_29dof, init_state=dataclasses.replace(g1_29dof.init_state, default_joint_positions=angles))
    tagged = {"#legs": 0.1, "#waist": 0.0, "#arms": 0.2}
    robot = _replace_robot(
        g1_29dof, init_state=dataclasses.replace(g1_29dof.init_state, default_joint_positions=tagged)
    )
    by_dof = robot.default_dof_positions()
    assert by_dof["left_knee_joint"] == 0.1 and by_dof["waist_yaw_joint"] == 0.0 and by_dof["left_elbow_joint"] == 0.2


def test_default_positions_map_preserves_serialized_config_and_dof_order() -> None:
    robot = _synthetic_robot(
        ["r_knee", "torso", "l_knee"],
        joint_tags={"#knees": ["#left", "r_knee"], "#left": ["l_knee"]},
        init_state=RobotInitState(position=[0.0, 0.0, 0.5], default_joint_positions={"torso": -0.1, "#knees": 0.3}),
    )
    serialized = json.dumps(dataclasses.asdict(robot))
    result = robot.default_dof_positions()
    assert list(result.items()) == [("r_knee", 0.3), ("torso", -0.1), ("l_knee", 0.3)]
    assert list(result.values()) == [0.3, -0.1, 0.3]

    # A deployment reader can resolve the same pose from saved fields alone.
    saved = json.loads(serialized)
    assert saved["init_state"]["default_joint_positions"] == {"torso": -0.1, "#knees": 0.3}
    assert (
        expand_joint_value_map(saved["init_state"]["default_joint_positions"], saved["joint_tags"], saved["dof_names"])
        == result
    )

    result["r_knee"] = 99.0
    assert robot.default_dof_positions()["r_knee"] == 0.3
    assert json.dumps(dataclasses.asdict(robot)) == serialized


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_default_joint_positions_must_be_finite(value: float) -> None:
    with pytest.raises(pydantic.ValidationError, match="default_joint_positions value for '#all' must be finite"):
        _synthetic_robot(
            ["knee"],
            init_state=RobotInitState(position=[0.0, 0.0, 0.5], default_joint_positions={"#all": value}),
        )


def test_default_positions_map_revalidates_mutable_default_entries() -> None:
    robot = _synthetic_robot(["knee"])
    robot.init_state.default_joint_positions.clear()
    assert robot.default_dof_positions() == {"knee": 0.0}
    robot.init_state.default_joint_positions["knee"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        robot.default_dof_positions()


def test_bridge_dof_names_must_exist() -> None:
    with pytest.raises(pydantic.ValidationError, match="bridge names \\['nope'\\] are not in dof_names"):
        _replace_robot(g1_29dof, bridge=dataclasses.replace(g1_29dof.bridge, excluded_dof_names=("nope",)))


def test_bridge_motor_order_must_not_repeat_a_dof() -> None:
    with pytest.raises(pydantic.ValidationError, match="bridge.controlled_dof_names has duplicate entries"):
        _replace_robot(
            g1_29dof,
            bridge=dataclasses.replace(
                g1_29dof.bridge, controlled_dof_names=("left_hip_pitch_joint", "left_hip_pitch_joint")
            ),
        )


@pytest.mark.parametrize(
    ("robot", "first_motor", "last_motor"),
    [
        (g1_29dof, "left_hip_pitch_joint", "right_wrist_yaw_joint"),
        (t1_29dof_waist_wrist, "AAHead_yaw", "Right_Ankle_Roll"),
    ],
)
def test_preset_sdk_order_survives_canonical_dof_reordering(
    robot: RobotConfig, first_motor: str, last_motor: str
) -> None:
    reordered = _replace_robot(robot, dof_names=list(reversed(robot.dof_names)))

    assert reordered.bridge.controlled_dof_names == robot.bridge.controlled_dof_names
    assert len(reordered.bridge.controlled_dof_names) == 29
    assert reordered.bridge.controlled_dof_names[0] == first_motor
    assert reordered.bridge.controlled_dof_names[-1] == last_motor
    assert reordered.dof_names[0] == last_motor
    assert reordered.dof_names[-1] == first_motor


def test_body_and_joint_tags_are_validated_against_names() -> None:
    ok = _replace_robot(
        g1_29dof,
        body_tags={"#feet": ["left_ankle_roll_link", "right_ankle_roll_link"], "#contacts": ["#feet", "pelvis"]},
        # The gain keys reference g1's pair tags, so the table is extended rather than replaced.
        joint_tags={**g1_29dof.joint_tags, "#knee_pair": ["#knees"]},
    )
    assert ok.body_tags["#contacts"] == ["#feet", "pelvis"]
    assert ok.joint_tags["#knee_pair"] == ["#knees"]
    with pytest.raises(pydantic.ValidationError, match="Unknown body name 'left_foot'"):
        _replace_robot(g1_29dof, body_tags={"#feet": ["left_foot"]})
    with pytest.raises(pydantic.ValidationError, match="Unknown joint name 'pelvis'"):
        _replace_robot(g1_29dof, joint_tags={"#base": ["pelvis"]})
    with pytest.raises(pydantic.ValidationError, match="body tag cycle"):
        _replace_robot(g1_29dof, body_tags={"#a": ["#b"], "#b": ["#a"]})
    with pytest.raises(pydantic.ValidationError, match="leading '#'"):
        _replace_robot(g1_29dof, body_tags={"feet": ["pelvis"]})


def test_tag_entries_must_be_lists_not_bare_strings() -> None:
    # The likeliest preset typo ('feet': 'left_ankle_roll_link') fails at construction, before any resolver.
    with pytest.raises(pydantic.ValidationError, match=r"body_tags\.#feet\s+Input should be a valid list"):
        _replace_robot(g1_29dof, body_tags={"#feet": "left_ankle_roll_link"})
    with pytest.raises(pydantic.ValidationError, match=r"joint_tags\.#knees\s+Input should be a valid list"):
        _replace_robot(g1_29dof, joint_tags={**g1_29dof.joint_tags, "#knees": "left_knee_joint"})


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("position", [0.0, 0.0], "position must have 3 entries"),
        ("orientation", [0.0, 0.0, 1.0], "List should have at least 4 items"),
        ("orientation", [2.0, 0.0, 0.0, 0.0], "Quaternion norm must be within"),
        ("lin_vel", [0.0], "lin_vel/ang_vel must have 3 entries"),
    ],
)
def test_init_state_shapes(field_name: str, value: list[float], match: str) -> None:
    overrides: dict[str, Any] = {field_name: value}
    with pytest.raises(pydantic.ValidationError, match=match):
        dataclasses.replace(g1_29dof.init_state, **overrides)


def test_isaacsim_synthetic_robot_config_constructs() -> None:
    """The IsaacSim articulation-cfg harness derives a 33-DOF synthetic robot from g1; it must validate."""
    from tests.simulators.isaacsim_articulation_cfg_assert import _synthetic_robot_config

    robot = _synthetic_robot_config()
    assert len(robot.dof_names) == 33


# --------------------------------------------------------------------------------------------
# extra="forbid": removed or misspelled fields fail loud on every Robot* class
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "kwarg"),
    [
        (g1_29dof, "num_bodies"),
        (g1_29dof, "foot_body_name"),
        (g1_29dof.init_state, "rot"),
        (g1_29dof.init_state, "default_joint_angles"),
        (g1_29dof.asset, "armature"),
        (g1_29dof.bridge, "motor_count"),
    ],
)
def test_unknown_field_is_rejected(target: Any, kwarg: str) -> None:
    with pytest.raises(pydantic.ValidationError, match=kwarg):
        dataclasses.replace(target, **{kwarg: 1})


@pytest.mark.parametrize("field", ["dof_vel_limits", "dof_effort_limits", "dof_armature", "dof_joint_friction"])
@pytest.mark.parametrize("value", [-0.1, float("nan")])
def test_invalid_physical_values_are_rejected(field: str, value: float) -> None:
    with pytest.raises(pydantic.ValidationError, match="DOF 'left_knee_joint'"):
        _replace_robot(g1_29dof, **{field: {"left_knee_joint": value}})


def test_sparse_tagged_properties_preserve_authored_maps_and_order_independence() -> None:
    from holosoma.utils.robot_properties import resolve_dof_properties

    robot = _replace_robot(g1_29dof, dof_effort_limits={"#knees": 12.0}, dof_pos_limits=None)
    before = dataclasses.asdict(robot)
    resolved = resolve_dof_properties(robot)
    assert resolved.effort == {"left_knee_joint": 12.0, "right_knee_joint": 12.0}
    assert resolved.position == {}
    assert dataclasses.asdict(robot) == before
    reordered = _replace_robot(robot, dof_names=list(reversed(robot.dof_names)))
    assert resolve_dof_properties(reordered).effort == resolved.effort
