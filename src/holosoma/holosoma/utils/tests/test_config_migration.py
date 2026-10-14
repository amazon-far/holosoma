"""Pre-refactor ``ExperimentConfig`` dumps migrate to the current schema (pure, no simulator).

The fixtures under ``data/`` were dumped by ``ExperimentConfig.to_serializable_dict`` on the schema
that preceded the RobotConfig refactor (humanoid name fields, ``init_state.pos``/``rot`` in xyzw,
symmetry lists on the robot, substring PD gain keys, ``contact_pairs_multiplier`` ...).
``migrate_legacy_experiment_config`` must turn them into configs that construct and reproduce the
shipped presets' behaviour.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pydantic
import pytest
import yaml
from loguru import logger

from holosoma.config_types.action import ActionTermCfg
from holosoma.config_types.command import NoiseToInitialPoseConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import EXPERIMENT_REGISTRY
from holosoma.utils.config_migration import migrate_legacy_experiment_config
from holosoma.utils.name_tags import expand_name_spec

pytestmark = pytest.mark.no_sim

_DATA = Path(__file__).parent / "data"
_ROBOTS = ["g1_29dof", "t1_29dof"]
# (kp, kd) the old rule gives these DOFs from the fixtures' key-sorted gain dicts.
_OLD_RULE_SPOT_GAINS = {
    "g1_29dof": {
        "left_hip_roll_joint": (99.098427777, 6.308801854),
        "waist_yaw_joint": (40.179238471, 2.557889765),
        "right_wrist_pitch_joint": (16.778327481, 1.068141502),
    },
    "t1_29dof": {"AAHead_yaw": (5.0, 0.5), "Left_Knee_Pitch": (200.0, 5.0), "Waist": (200.0, 5.0)},
}


def _legacy(name: str) -> dict[str, Any]:
    return yaml.safe_load((_DATA / f"legacy_{name}_experiment_config.yaml").read_text())  # type: ignore[no-any-return]


def _preset_gains(name: str) -> tuple[dict[str, float], dict[str, float]]:
    """The shipped preset's per-DOF ``(kp, kd)`` keyed by DOF name: the pre-refactor values ``test_robot_config``
    pins, so the expected migration output is not a re-implementation of the substring rule."""
    cfg = EXPERIMENT_REGISTRY[name]
    controller = _joint_term(cfg).controller
    assert controller is not None
    kp, kd = controller.resolve_gains(cfg.robot)
    robot = cfg.robot
    return dict(zip(robot.dof_names, kp)), dict(zip(robot.dof_names, kd))


def _joint_term(cfg: ExperimentConfig) -> ActionTermCfg:
    assert cfg.action is not None
    return cfg.action.terms["joint_control"]


def _joint_controller_dict(raw: dict[str, Any]) -> dict[str, Any]:
    return raw["action"]["terms"]["joint_control"]["controller"]  # type: ignore[no-any-return]


def _synthetic_joint_action() -> dict[str, Any]:
    return {
        "terms": {
            "joint_control": {
                "func": "holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
                "params": {},
                "scale": 1.0,
                "clip": None,
            }
        }
    }


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    msgs: list[str] = []
    sink_id = logger.add(lambda m: msgs.append(m.record["message"]), level="WARNING")
    yield msgs
    logger.remove(sink_id)


@pytest.mark.parametrize("name", _ROBOTS)
def test_raw_legacy_dump_is_rejected_loudly(name: str) -> None:
    with pytest.raises(pydantic.ValidationError, match="num_bodies"):
        ExperimentConfig(**_legacy(name))


@pytest.mark.parametrize("name", _ROBOTS)
def test_migrated_dump_constructs_and_matches_the_shipped_experiment(name: str) -> None:
    migrated = migrate_legacy_experiment_config(_legacy(name))
    cfg = ExperimentConfig(**migrated)
    current = EXPERIMENT_REGISTRY[name]
    robot, cur_robot = cfg.robot, current.robot

    # Articulation and spawn state (pos/rot xyzw -> position/orientation wxyz) are unchanged.
    assert robot.dof_names == cur_robot.dof_names
    assert robot.body_names == cur_robot.body_names
    assert robot.init_state == cur_robot.init_state
    assert robot.asset == cur_robot.asset
    term, cur_term = _joint_term(cfg), _joint_term(current)
    assert term.scale == cur_term.scale and term.clip == cur_term.clip
    assert term.controller is not None and cur_term.controller is not None
    assert (
        dataclasses.replace(
            term.controller,
            stiffness=cur_term.controller.stiffness,
            damping=cur_term.controller.damping,
        )
        == cur_term.controller
    )
    assert term.controller.resolve_gains(robot) == cur_term.controller.resolve_gains(cur_robot)

    # Morphology facts became tags with exactly the shipped preset's members.
    for tag in ("feet", "feet_height", "base"):
        assert expand_name_spec([f"#{tag}"], robot.body_tags, robot.body_names) == expand_name_spec(
            [f"#{tag}"], cur_robot.body_tags, cur_robot.body_names
        )

    # The dump's symmetry dict was key-sorted by yaml, so compare the pairing, not the order.
    def _pairs(r: Any) -> set[tuple[str, str]]:
        left = expand_name_spec(["#mirror_left"], r.joint_tags, r.dof_names, kind="joint")
        right = expand_name_spec(["#mirror_right"], r.joint_tags, r.dof_names, kind="joint")
        return set(zip(left, right))

    assert _pairs(robot) == _pairs(cur_robot) and len(_pairs(robot)) == 13
    assert set(expand_name_spec(["#mirror_flip_sign"], robot.joint_tags, robot.dof_names, kind="joint")) == set(
        expand_name_spec(["#mirror_flip_sign"], cur_robot.joint_tags, cur_robot.dof_names, kind="joint")
    )

    # Task decisions moved into the consuming terms' params with the same resolved bodies.
    assert cfg.termination is not None and current.termination is not None
    contact = cfg.termination.terms["contact"].params
    assert "contact_indices_attr" not in contact
    # Contact termination is order-free; the shipped preset now names the set by tag.
    assert set(contact["bodies"]) == set(
        expand_name_spec(
            current.termination.terms["contact"].params["bodies"], cur_robot.body_tags, cur_robot.body_names
        )
    )
    assert cfg.randomization is not None and current.randomization is not None
    mass = cfg.randomization.setup_terms["mass_randomizer"].params
    # The shipped preset names the DR links by tag; the migrated dump lists them literally (order-free set).
    assert set(mass["link_bodies"]) == set(
        expand_name_spec(
            current.randomization.setup_terms["mass_randomizer"].params["link_bodies"],
            cur_robot.body_tags,
            cur_robot.body_names,
        )
    )
    assert mass["base_body"] == "#base"
    assert cfg.randomization.setup_terms["randomize_base_com_startup"].params["body"] == "#base"

    # Relocated knob (16 is also the PhysxConfig default; test_contact_pairs_multiplier_relocates_by_value pins
    # the relocation itself).
    assert cfg.simulator.config.sim.physx.max_gpu_contact_pairs == 16 * 1024 * 1024


def test_init_state_rot_xyzw_becomes_orientation_wxyz() -> None:
    # The fixtures spawn upright (identity), which any reorder that keeps w last reproduces; use a real rotation.
    raw = _legacy("g1_29dof")
    raw["robot"]["init_state"]["rot"] = [0.0, 0.6, 0.0, 0.8]

    migrated = migrate_legacy_experiment_config(raw)

    init_state = migrated["robot"]["init_state"]
    assert "rot" not in init_state and "pos" not in init_state
    assert init_state["orientation"] == [0.8, 0.0, 0.6, 0.0]
    assert init_state["position"] == raw["robot"]["init_state"]["pos"]
    assert ExperimentConfig(**migrated).robot.init_state.orientation_xyzw == [0.0, 0.6, 0.0, 0.8]


def test_legacy_default_positions_keep_tags_and_do_not_add_a_legacy_dataclass_field(warnings: list[str]) -> None:
    raw = EXPERIMENT_REGISTRY["g1_29dof"].to_serializable_dict()
    defaults = {"#legs": 0.1, "#waist": -0.2, "#arms": 0.3}
    init_state = raw["robot"]["init_state"]
    init_state.pop("default_joint_positions")
    init_state["default_joint_angles"] = defaults
    snapshot = copy.deepcopy(raw)

    migrated = migrate_legacy_experiment_config(raw)
    cfg = ExperimentConfig(**migrated)
    saved = cfg.to_serializable_dict()
    assert saved["robot"]["init_state"]["default_joint_positions"] == defaults
    assert "default_joint_angles" not in saved["robot"]["init_state"]
    assert "default_joint_angles" not in {field.name for field in dataclasses.fields(cfg.robot.init_state)}
    assert cfg.robot.default_dof_positions()["left_knee_joint"] == 0.1
    assert cfg.robot.default_dof_positions()["waist_yaw_joint"] == -0.2
    assert cfg.robot.default_dof_positions()["left_elbow_joint"] == 0.3
    assert raw == snapshot
    assert migrate_legacy_experiment_config(migrated) == migrated
    assert any("init_state.default_joint_angles -> init_state.default_joint_positions" in msg for msg in warnings)


@pytest.mark.parametrize("name", _ROBOTS)
def test_legacy_superset_default_positions_drop_only_non_dofs(name: str, warnings: list[str]) -> None:
    raw = _legacy(name)
    expected = dict(raw["robot"]["init_state"]["default_joint_angles"])
    raw["robot"]["init_state"]["default_joint_angles"].update({"imu_in_torso_joint": 0.0, "unused_tool_joint": 1.25})
    snapshot = copy.deepcopy(raw)

    migrated = migrate_legacy_experiment_config(raw)

    assert migrated["robot"]["init_state"]["default_joint_positions"] == expected
    robot = ExperimentConfig(**migrated).robot
    assert robot.default_dof_positions() == expected
    assert list(robot.default_dof_positions()) == raw["robot"]["dof_names"]
    assert raw == snapshot
    drop_warnings = [msg for msg in warnings if "default_joint_angles entries" in msg]
    assert len(drop_warnings) == 1
    assert "imu_in_torso_joint" in drop_warnings[0] and "unused_tool_joint" in drop_warnings[0]
    warnings.clear()
    assert migrate_legacy_experiment_config(migrated) == migrated
    assert warnings == []


@pytest.mark.parametrize("invalid", ["missing", "nonfinite"])
def test_legacy_superset_defaults_still_validate_active_dofs(invalid: str) -> None:
    raw = _legacy("g1_29dof")
    defaults = raw["robot"]["init_state"]["default_joint_angles"]
    defaults["imu_in_torso_joint"] = 0.0
    joint = raw["robot"]["dof_names"][0]
    if invalid == "missing":
        defaults.pop(joint)
        message = "has no key covering DOF"
    else:
        defaults[joint] = float("nan")
        message = "must be finite"

    migrated = migrate_legacy_experiment_config(raw)
    if invalid == "missing":
        assert ExperimentConfig(**migrated).robot.default_dof_positions()[joint] == 0.0
    else:
        with pytest.raises(pydantic.ValidationError, match=message):
            ExperimentConfig(**migrated)


@pytest.mark.parametrize("tag_signal", ["defaults", "joint_tags", "empty_joint_tags"])
def test_tag_aware_defaults_with_old_schema_markers_stay_strict(tag_signal: str) -> None:
    raw = _legacy("g1_29dof")
    defaults = raw["robot"]["init_state"]["default_joint_angles"]
    if tag_signal == "defaults":
        defaults = {"#all": 0.1}
        raw["robot"]["init_state"]["default_joint_angles"] = defaults
    elif tag_signal == "joint_tags":
        raw["robot"]["joint_tags"] = {"#all": raw["robot"]["dof_names"]}
    elif tag_signal == "empty_joint_tags":
        raw["robot"]["joint_tags"] = {}
    defaults["unused_joint"] = 0.0

    migrated = migrate_legacy_experiment_config(raw)

    assert migrated["robot"]["init_state"]["default_joint_positions"] == defaults
    with pytest.raises(pydantic.ValidationError, match="Unknown joint name 'unused_joint'"):
        ExperimentConfig(**migrated)


@pytest.mark.parametrize("keep_legacy_key", [False, True])
def test_current_default_positions_with_old_schema_markers_stay_strict(keep_legacy_key: bool) -> None:
    raw = _legacy("g1_29dof")
    init_state = raw["robot"]["init_state"]
    defaults = init_state.pop("default_joint_angles")
    defaults["unused_joint"] = 0.0
    init_state["default_joint_positions"] = defaults
    if keep_legacy_key:
        init_state["default_joint_angles"] = dict(defaults)

    migrated = migrate_legacy_experiment_config(raw)

    assert migrated["robot"]["init_state"]["default_joint_positions"] == defaults
    with pytest.raises(pydantic.ValidationError, match="Unknown joint name 'unused_joint'"):
        ExperimentConfig(**migrated)


def test_legacy_default_positions_without_dof_names_are_not_filtered() -> None:
    raw = {
        "robot": {
            "init_state": {"pos": [0, 0, 1], "default_joint_angles": {"unknown_joint": 0.1}},
        }
    }
    migrated = migrate_legacy_experiment_config(raw)
    assert migrated["robot"]["init_state"]["default_joint_positions"] == {"unknown_joint": 0.1}


def test_conflicting_legacy_and_current_default_positions_are_rejected() -> None:
    raw = EXPERIMENT_REGISTRY["g1_29dof"].to_serializable_dict()
    raw["robot"]["init_state"]["default_joint_angles"] = {"#all": 0.8}
    with pytest.raises(ValueError, match="conflicting default_joint_angles and default_joint_positions"):
        migrate_legacy_experiment_config(raw)


def test_matching_legacy_and_current_default_positions_keep_only_current_key() -> None:
    raw = EXPERIMENT_REGISTRY["g1_29dof"].to_serializable_dict()
    init_state = raw["robot"]["init_state"]
    init_state["default_joint_angles"] = copy.deepcopy(init_state["default_joint_positions"])
    migrated = migrate_legacy_experiment_config(raw)
    assert "default_joint_angles" not in migrated["robot"]["init_state"]
    assert migrated["robot"]["init_state"]["default_joint_positions"] == init_state["default_joint_positions"]


@pytest.mark.parametrize("name", _ROBOTS)
def test_substring_gain_keys_become_exact_per_dof_keys(name: str) -> None:
    legacy = _legacy(name)
    expected_kp, expected_kd = _preset_gains(name)

    cfg = ExperimentConfig(**migrate_legacy_experiment_config(legacy))
    control = _joint_term(cfg).controller
    assert control is not None

    assert list(control.stiffness) == legacy["robot"]["dof_names"] == list(control.damping)
    assert control.stiffness == expected_kp
    assert control.damping == expected_kd
    for dof, (kp, kd) in _OLD_RULE_SPOT_GAINS[name].items():
        assert (control.stiffness[dof], control.damping[dof]) == (kp, kd)


def test_gain_key_expansion_is_last_match_wins() -> None:
    raw = _legacy("g1_29dof")
    control = raw["robot"]["control"]
    control["stiffness"]["hip"] = 1.0  # appended, so iterated after the specific hip_* keys
    control["damping"]["hip"] = 0.1

    migrated = _joint_controller_dict(migrate_legacy_experiment_config(raw))

    assert (migrated["stiffness"]["left_hip_roll_joint"], migrated["damping"]["left_hip_roll_joint"]) == (1.0, 0.1)
    assert migrated["stiffness"]["left_knee_joint"] == control["stiffness"]["knee"]

    control["stiffness"] = {"hip": 1.0, **{k: v for k, v in control["stiffness"].items() if k != "hip"}}
    control["damping"] = {"hip": 0.1, **{k: v for k, v in control["damping"].items() if k != "hip"}}

    migrated = _joint_controller_dict(migrate_legacy_experiment_config(raw))

    assert migrated["stiffness"]["left_hip_roll_joint"] == control["stiffness"]["hip_roll"]


@pytest.mark.parametrize(
    "legacy_fields",
    [{"num_bodies": 1}, {"init_state": {"pos": [0, 0, 1]}}, {"init_state": {"rot": [0, 0, 0, 1]}}],
    ids=["morphology-schema", "spawn-position-schema", "spawn-orientation-schema"],
)
def test_legacy_complete_name_overlaps_keep_substring_gains(legacy_fields: dict[str, Any]) -> None:
    raw = {
        "action": _synthetic_joint_action(),
        "robot": {
            **legacy_fields,
            "dof_names": ["j", "j2"],
            "control": {
                "control_type": "P",
                "stiffness": {"j2": 20.0, "j": 10.0},
                "damping": {"j2": 2.0, "j": 1.0},
            },
        },
    }
    before = copy.deepcopy(raw)
    migrated = migrate_legacy_experiment_config(raw)
    # The base controller's last matching key was "j" for both DOFs, despite both keys being full names.
    controller = _joint_controller_dict(migrated)
    assert controller["stiffness"] == {"j": 10.0, "j2": 10.0}
    assert controller["damping"] == {"j": 1.0, "j2": 1.0}
    assert "control" not in migrated["robot"]
    assert migrate_legacy_experiment_config(migrated) == migrated
    assert raw == before


def test_unmatched_dof_raises_for_pd_control() -> None:
    raw = _legacy("g1_29dof")
    del raw["robot"]["control"]["stiffness"]["knee"]
    del raw["robot"]["control"]["damping"]["knee"]
    with pytest.raises(ValueError, match="PD gains for joint 'left_knee_joint' were not defined"):
        migrate_legacy_experiment_config(raw)


def test_mixed_tag_and_substring_gain_keys_raise_naming_the_tag_keys() -> None:
    """A half-migrated dump cannot interpret tag keys under the legacy substring rule."""
    raw = _legacy("g1_29dof")
    control = raw["robot"]["control"]
    control["stiffness"] = {"#knees": control["stiffness"].pop("knee"), **control["stiffness"]}
    control["damping"] = {"#knees": control["damping"].pop("knee"), **control["damping"]}
    with pytest.raises(
        ValueError,
        match=r"Cannot mix legacy substring PD gain keys.*#knees",
    ):
        migrate_legacy_experiment_config(raw)


def test_stiffness_key_without_a_damping_entry_raises() -> None:
    raw = _legacy("g1_29dof")
    del raw["robot"]["control"]["damping"]["knee"]
    with pytest.raises(ValueError, match="damping has no entry for stiffness key 'knee' \\(DOF 'left_knee_joint'\\)"):
        migrate_legacy_experiment_config(raw)


@pytest.mark.parametrize("field", ["stiffness", "damping"])
def test_mixed_gain_schemes_cannot_silently_discard_a_tag(field: str) -> None:
    raw = _legacy("g1_29dof")
    raw["robot"]["control"][field]["#knees"] = 123.0
    raw["robot"]["joint_tags"] = {"#knees": ["left_knee_joint", "right_knee_joint"]}
    snapshot = copy.deepcopy(raw)
    with pytest.raises(ValueError, match=r"(mix|Mix).*substring.*#knees"):
        migrate_legacy_experiment_config(raw)
    assert raw == snapshot


def test_unmatched_dof_gets_zero_gains_for_torque_control() -> None:
    raw = _legacy("g1_29dof")
    raw["robot"]["control"]["control_type"] = "T"
    del raw["robot"]["control"]["stiffness"]["knee"]
    del raw["robot"]["control"]["damping"]["knee"]

    cfg = ExperimentConfig(**migrate_legacy_experiment_config(raw))
    control = _joint_term(cfg).controller
    assert control is not None

    assert (control.stiffness["left_knee_joint"], control.damping["right_knee_joint"]) == (0.0, 0.0)
    assert control.stiffness["left_hip_roll_joint"] == 99.098427777


def test_exact_and_tag_gain_keys_are_left_alone() -> None:
    raw = _legacy("g1_29dof")
    exact_kp, exact_kd = _preset_gains("g1_29dof")
    raw["robot"]["control"]["stiffness"] = {**exact_kp, "#waist": 1.0}
    raw["robot"]["control"]["damping"] = {**exact_kd, "#waist": 0.1}

    migrated = migrate_legacy_experiment_config(raw)
    assert _joint_controller_dict(migrated)["stiffness"] == raw["robot"]["control"]["stiffness"]
    assert _joint_controller_dict(migrated)["damping"] == raw["robot"]["control"]["damping"]
    assert "control" not in migrated["robot"]


def test_migration_is_idempotent() -> None:
    once = migrate_legacy_experiment_config(_legacy("g1_29dof"))
    assert migrate_legacy_experiment_config(once) == once


def test_current_dump_and_foreign_dicts_pass_through_unchanged() -> None:
    current = EXPERIMENT_REGISTRY["g1_29dof"].to_serializable_dict()
    assert migrate_legacy_experiment_config(current) == current
    foreign = {"base_field": 1, "robot": "not-a-dict", "nested": {"a": [1, 2]}}
    assert migrate_legacy_experiment_config(foreign) == foreign


def test_input_is_not_mutated() -> None:
    raw = _legacy("g1_29dof")
    snapshot = copy.deepcopy(raw)
    migrate_legacy_experiment_config(raw)
    assert raw == snapshot


def test_apply_dof_armature_false_cannot_migrate() -> None:
    raw = _legacy("g1_29dof")
    raw["robot"]["apply_dof_armature_in_isaacgym"] = False
    with pytest.raises(ValueError, match="apply_dof_armature_in_isaacgym=false"):
        migrate_legacy_experiment_config(raw)


def test_non_default_contact_indices_attr_cannot_migrate() -> None:
    raw = _legacy("g1_29dof")
    raw["termination"]["terms"]["contact"]["params"]["contact_indices_attr"] = "custom_indices"
    with pytest.raises(ValueError, match="contact_indices_attr='custom_indices'"):
        migrate_legacy_experiment_config(raw)


def test_pre_2026_07_09_asset_physics_and_robot_object_are_dropped() -> None:
    raw = _legacy("g1_29dof")
    raw["robot"]["object"] = {"object_urdf_path": "x.urdf"}
    raw["robot"]["asset"].update(
        {"max_linear_velocity": 1000.0, "max_angular_velocity": 1000.0, "linear_damping": 0.0, "angular_damping": 0.0}
    )
    cfg = ExperimentConfig(**migrate_legacy_experiment_config(raw))
    assert cfg.robot.asset.link_physics == EXPERIMENT_REGISTRY["g1_29dof"].robot.asset.link_physics


def test_unknown_robot_keys_are_dropped_with_a_warning(warnings: list[str]) -> None:
    """Fork / subclass fields a dump carries (never typos: dumps are machine-written) are dropped, named."""
    raw = _legacy("g1_29dof")
    raw["robot"]["dof_names_23dof"] = ["left_hip_pitch_joint"]
    raw["robot"]["asset"]["fork_flag"] = True

    migrated = migrate_legacy_experiment_config(raw)

    cfg = ExperimentConfig(**migrated)
    assert _joint_term(cfg).controller is not None
    dropped = next(m for m in warnings if "dropped [" in m)
    assert all(key in dropped for key in ("dof_names_23dof", "asset.fork_flag"))


def test_non_default_soft_limit_is_carried_into_the_motion_command_terms() -> None:
    """The WBT initial-pose clamp consumed robot.soft_dof_pos_limit; the value moves onto every MotionCommand term."""
    raw = _legacy("g1_29dof")
    raw["robot"]["soft_dof_pos_limit"] = 0.8
    raw["command"] = EXPERIMENT_REGISTRY["g1_29dof_wbt"].to_serializable_dict()["command"]
    noise_path = ("setup_terms", "motion_command", "params", "motion_config", "noise_to_initial_pose")

    def _noise(command: dict[str, Any]) -> dict[str, Any]:
        node: Any = command
        for key in noise_path:
            node = node[key]
        return node  # type: ignore[no-any-return]

    del _noise(raw["command"])["soft_dof_pos_limit"]  # a pre-refactor NoiseToInitialPoseConfig had no such field

    migrated = migrate_legacy_experiment_config(raw)

    assert _noise(migrated["command"])["soft_dof_pos_limit"] == 0.8
    cfg = ExperimentConfig(**migrated)
    assert cfg.command is not None
    params = cfg.command.setup_terms["motion_command"].params  # term params stay a dict until the term builds them
    assert params["motion_config"]["noise_to_initial_pose"]["soft_dof_pos_limit"] == 0.8

    # An explicit term value wins over the legacy robot field.
    explicit = _legacy("g1_29dof")
    explicit["robot"]["soft_dof_pos_limit"] = 0.8
    explicit["command"] = EXPERIMENT_REGISTRY["g1_29dof_wbt"].to_serializable_dict()["command"]
    _noise(explicit["command"])["soft_dof_pos_limit"] = 0.9
    assert _noise(migrate_legacy_experiment_config(explicit)["command"])["soft_dof_pos_limit"] == 0.9


def test_non_default_soft_limit_without_a_motion_command_warns(warnings: list[str]) -> None:
    raw = _legacy("g1_29dof")
    raw["robot"]["soft_dof_pos_limit"] = 0.8
    ExperimentConfig(**migrate_legacy_experiment_config(raw))
    assert any("soft_dof_pos_limit=0.8 dropped: no MotionCommand term" in m for m in warnings)


def test_one_way_symmetry_map_fails_loud() -> None:
    """The old code applied a non-reciprocal map one-way; converting it to swap pairs would change behavior."""
    raw = _legacy("g1_29dof")
    del raw["robot"]["symmetry_joint_names"]["right_knee_joint"]  # left_knee -> right_knee has no reverse
    with pytest.raises(ValueError, match=r"not reciprocal: \[\('left_knee_joint', 'right_knee_joint'\)\]"):
        migrate_legacy_experiment_config(raw)


def test_one_way_symmetry_map_is_dropped_when_symmetry_is_off(warnings: list[str]) -> None:
    raw = _legacy("g1_29dof")
    del raw["robot"]["symmetry_joint_names"]["right_knee_joint"]
    raw["algo"]["config"]["use_symmetry"] = False

    cfg = ExperimentConfig(**migrate_legacy_experiment_config(raw))

    left = expand_name_spec(["#mirror_left"], cfg.robot.joint_tags, cfg.robot.dof_names, kind="joint")
    assert "left_knee_joint" not in left and len(left) == 12  # the one-way pair is gone, the other 12 stay
    assert any("one-way, and use_symmetry is off" in m for m in warnings)


def test_nested_unknown_robot_keys_are_dropped(warnings: list[str]) -> None:
    raw = _legacy("g1_29dof")
    raw["robot"]["asset"]["link_physics"] = {"fork_flag": True, "physx": {"fork_knob": 1.0}}

    migrated = migrate_legacy_experiment_config(raw)

    assert migrated["robot"]["asset"]["link_physics"] == {"physx": {}}
    ExperimentConfig(**migrated)
    dropped = next(m for m in warnings if "dropped [" in m)
    assert "asset.link_physics.fork_flag" in dropped and "asset.link_physics.physx.fork_knob" in dropped


def test_soft_limit_equal_to_the_wbt_clamp_default_is_dropped_silently(warnings: list[str]) -> None:
    # The WBT initial-pose clamp now reads NoiseToInitialPoseConfig.soft_dof_pos_limit; only a differing value warns.
    raw = _legacy("g1_29dof")
    raw["robot"]["soft_dof_pos_limit"] = NoiseToInitialPoseConfig().soft_dof_pos_limit
    migrated = migrate_legacy_experiment_config(raw)
    ExperimentConfig(**migrated)
    assert "soft_dof_pos_limit" not in migrated["robot"]
    assert not any("Legacy robot.soft_dof_pos_limit=" in m for m in warnings)


def test_existing_tags_win_over_derived_ones() -> None:
    # Each derived body tag (feet <- foot_body_name, feet_height <- foot_height_name, base <- torso_name) yields to
    # one the dump already declares, and only the undeclared one is derived.
    raw = _legacy("g1_29dof")
    declared = {"#feet": ["pelvis"], "#feet_height": ["pelvis"], "#base": ["pelvis"]}
    for derived_tag in declared:
        raw["robot"]["body_tags"] = {tag: names for tag, names in declared.items() if tag != derived_tag}

        cfg = ExperimentConfig(**migrate_legacy_experiment_config(raw))

        derived = EXPERIMENT_REGISTRY["g1_29dof"].robot.body_tags[derived_tag]
        declared_after = {tag: names for tag, names in cfg.robot.body_tags.items() if tag != "#all"}
        assert declared_after == {**raw["robot"]["body_tags"], derived_tag: derived}


def test_contact_pairs_relocation_is_skipped_when_the_physx_chain_is_missing(warnings: list[str]) -> None:
    null_simulator = _legacy("g1_29dof")
    null_simulator["simulator"] = None
    migrated = migrate_legacy_experiment_config(null_simulator)
    assert migrated["simulator"] is None and "contact_pairs_multiplier" not in migrated["robot"]

    no_sim = _legacy("g1_29dof")
    del no_sim["simulator"]["config"]["sim"]
    migrated = migrate_legacy_experiment_config(no_sim)
    assert "sim" not in migrated["simulator"]["config"] and "contact_pairs_multiplier" not in migrated["robot"]

    skipped = [m for m in warnings if "contact_pairs_multiplier dropped: no simulator.config.sim.physx" in m]
    assert len(skipped) == 2


def test_contact_pairs_multiplier_relocates_by_value() -> None:
    # The fixtures' 16 equals the PhysxConfig default, which cannot tell a relocation from a drop.
    raw = _legacy("g1_29dof")
    raw["robot"]["contact_pairs_multiplier"] = 8

    migrated = migrate_legacy_experiment_config(raw)

    assert migrated["simulator"]["config"]["sim"]["physx"]["max_gpu_contact_pairs"] == 8 * 1024 * 1024
    assert "contact_pairs_multiplier" not in migrated["robot"]
    assert ExperimentConfig(**migrated).simulator.config.sim.physx.max_gpu_contact_pairs == 8 * 1024 * 1024


def test_existing_max_gpu_contact_pairs_is_not_overwritten() -> None:
    raw = _legacy("g1_29dof")
    raw["simulator"]["config"]["sim"]["physx"]["max_gpu_contact_pairs"] = 123

    migrated = migrate_legacy_experiment_config(raw)

    assert migrated["simulator"]["config"]["sim"]["physx"]["max_gpu_contact_pairs"] == 123
    assert "contact_pairs_multiplier" not in migrated["robot"]


def test_has_torso_false_still_derives_the_base_tag() -> None:
    raw = _legacy("g1_29dof")
    raw["robot"]["has_torso"] = False
    cfg = ExperimentConfig(**migrate_legacy_experiment_config(raw))
    assert cfg.robot.body_tags["#base"] == ["torso_link"]
    assert cfg.randomization is not None
    assert cfg.randomization.setup_terms["mass_randomizer"].params["base_body"] == "#base"
    assert cfg.randomization.setup_terms["randomize_base_com_startup"].params["body"] == "#base"


def test_unknown_torso_name_leaves_the_base_params_unset_and_warns(warnings: list[str]) -> None:
    raw = _legacy("g1_29dof")
    raw["robot"]["torso_name"] = "ghost_link"
    cfg = ExperimentConfig(**migrate_legacy_experiment_config(raw))
    assert "#base" not in cfg.robot.body_tags
    assert cfg.randomization is not None
    assert "base_body" not in cfg.randomization.setup_terms["mass_randomizer"].params
    assert "body" not in cfg.randomization.setup_terms["randomize_base_com_startup"].params
    assert any("torso_name='ghost_link' is not in body_names" in m and "base_body" in m for m in warnings)


def test_stale_link_body_name_is_dropped_with_a_warning(warnings: list[str]) -> None:
    raw = _legacy("g1_29dof")
    raw["robot"]["randomize_link_body_names"].append("ghost_link")
    kept = list(raw["robot"]["randomize_link_body_names"][:-1])

    cfg = ExperimentConfig(**migrate_legacy_experiment_config(raw))

    assert cfg.randomization is not None
    assert cfg.randomization.setup_terms["mass_randomizer"].params["link_bodies"] == kept
    assert any("randomize_link_body_names ['ghost_link'] dropped" in m for m in warnings)


def test_foot_pattern_matching_no_body_yields_no_tag_and_warns(warnings: list[str]) -> None:
    raw = _legacy("g1_29dof")
    raw["robot"]["foot_height_name"] = "ghost_point"

    cfg = ExperimentConfig(**migrate_legacy_experiment_config(raw))

    assert "#feet_height" not in cfg.robot.body_tags
    assert cfg.robot.body_tags["#feet"] == ["left_ankle_roll_link", "right_ankle_roll_link"]
    assert any("foot_height_name='ghost_point' matches no body_names entry" in m for m in warnings)


def test_stale_flip_sign_name_is_dropped_with_a_warning(warnings: list[str]) -> None:
    raw = _legacy("g1_29dof")
    raw["robot"]["flip_sign_joint_names"].append("ghost_joint")

    cfg = ExperimentConfig(**migrate_legacy_experiment_config(raw))  # use_symmetry=True: the validator runs

    flipped = expand_name_spec(["#mirror_flip_sign"], cfg.robot.joint_tags, cfg.robot.dof_names, kind="joint")
    assert "ghost_joint" not in flipped and len(flipped) == 16
    assert any("flip_sign_joint_names ['ghost_joint'] dropped" in m for m in warnings)


def test_stale_mirror_pair_member_drops_the_whole_pair(warnings: list[str]) -> None:
    raw = _legacy("g1_29dof")
    mirror = raw["robot"]["symmetry_joint_names"]
    del mirror["right_knee_joint"]
    mirror["left_knee_joint"] = "ghost_joint"
    mirror["ghost_joint"] = "left_knee_joint"

    cfg = ExperimentConfig(**migrate_legacy_experiment_config(raw))

    left = expand_name_spec(["#mirror_left"], cfg.robot.joint_tags, cfg.robot.dof_names, kind="joint")
    right = expand_name_spec(["#mirror_right"], cfg.robot.joint_tags, cfg.robot.dof_names, kind="joint")
    assert len(left) == len(right) == 12
    assert "left_knee_joint" not in left + right
    assert any("[('left_knee_joint', 'ghost_joint')] dropped" in m for m in warnings)


def test_all_stale_symmetry_names_emit_no_mirror_tags() -> None:
    raw = _legacy("g1_29dof")
    raw["robot"]["symmetry_joint_names"] = {"ghost_l": "ghost_r", "ghost_r": "ghost_l"}
    raw["robot"]["flip_sign_joint_names"] = ["ghost_l"]

    migrated = migrate_legacy_experiment_config(raw)

    assert not {"#mirror_left", "#mirror_right", "#mirror_flip_sign"} & set(migrated["robot"].get("joint_tags", {}))
    with pytest.raises(pydantic.ValidationError, match="set use_symmetry=False"):
        ExperimentConfig(**migrated)
    migrated["algo"]["config"]["use_symmetry"] = False
    ExperimentConfig(**migrated)


@pytest.mark.parametrize("reverse", [False, True])
def test_legacy_property_lists_migrate_by_saved_joint_order(reverse: bool) -> None:
    raw = EXPERIMENT_REGISTRY["g1_29dof"].to_serializable_dict()
    robot = raw["robot"]
    if reverse:
        robot["dof_names"].reverse()
    names = robot["dof_names"]
    fields = {
        "dof_vel_limit_list": "dof_vel_limits",
        "dof_effort_limit_list": "dof_effort_limits",
        "dof_armature_list": "dof_armature",
        "dof_joint_friction_list": "dof_joint_friction",
    }
    values = [float(i + 1) for i in range(len(names))]
    for old, new in fields.items():
        robot.pop(new)
        robot[old] = list(values)
    robot.pop("dof_pos_limits")
    robot["dof_pos_lower_limit_list"] = [-v for v in values]
    robot["dof_pos_upper_limit_list"] = list(values)
    original = copy.deepcopy(raw)
    migrated = migrate_legacy_experiment_config(raw)
    config = ExperimentConfig(**migrated)
    for old, new in fields.items():
        assert old not in migrated["robot"]
        assert getattr(config.robot, new) == dict(zip(names, values))
    assert config.robot.dof_pos_limits == {name: (-v, v) for name, v in zip(names, values)}
    assert raw == original
    assert migrate_legacy_experiment_config(migrated) == migrated


@pytest.mark.parametrize("error", ["length", "conflict", "half-range"])
def test_ambiguous_or_incomplete_legacy_properties_fail_without_changing_input(error: str) -> None:
    raw = EXPERIMENT_REGISTRY["g1_29dof"].to_serializable_dict()
    if error == "length":
        raw["robot"]["dof_effort_limit_list"] = [1.0]
    elif error == "conflict":
        raw["robot"]["dof_effort_limit_list"] = [1.0] * len(raw["robot"]["dof_names"])
    else:
        raw["robot"]["dof_pos_lower_limit_list"] = [-1.0] * len(raw["robot"]["dof_names"])
    original = copy.deepcopy(raw)
    with pytest.raises(ValueError, match="entries|conflicts|together"):
        migrate_legacy_experiment_config(raw)
    assert raw == original
