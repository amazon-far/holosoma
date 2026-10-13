"""Sparse legacy WBT params preserve their reset clamp through checkpoint loading."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from holosoma.config_values.experiment import EXPERIMENT_REGISTRY
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.config_migration import migrate_legacy_experiment_config
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
from holosoma.utils.helpers import get_class

pytestmark = pytest.mark.no_sim

_FUNC = "holosoma.managers.command.terms.wbt:MotionCommand"
_DATA = Path(__file__).parent / "data" / "legacy_g1_29dof_experiment_config.yaml"


def _write_clip(path: Path, bodies: list[str], joints: list[str]) -> None:
    """A constant pose between the legacy 0.8 clamp and the current 0.95 default."""
    frames = 3
    joint_pos = np.zeros((frames, 7 + len(joints)), dtype=np.float32)
    joint_pos[:, 3] = 1.0
    joint_pos[:, 7:] = 0.9
    body_quat = np.zeros((frames, len(bodies), 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez(
        path,
        fps=50,
        body_names=bodies,
        joint_names=joints,
        joint_pos=joint_pos,
        joint_vel=np.zeros((frames, 6 + len(joints)), dtype=np.float32),
        body_pos_w=np.zeros((frames, len(bodies), 3), dtype=np.float32),
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((frames, len(bodies), 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, len(bodies), 3), dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("noise", "expected_factor", "expected_target"),
    [
        (None, 0.8, 0.8),
        ({}, 0.8, 0.8),
        ({"dof_pos": 0.125}, 0.8, 0.8),
        ({"soft_dof_pos_limit": 0.6}, 0.6, 0.6),
        ({"soft_dof_pos_limit": 0.95}, 0.95, 0.9),
    ],
    ids=["omitted", "empty", "other-noise-fields", "explicit-factor", "explicit-default"],
)
def test_sparse_motion_params_preserve_reset_clamp_after_checkpoint_load(
    tmp_path: Path, noise: dict[str, float] | None, expected_factor: float, expected_target: float
) -> None:
    raw: dict[str, Any] = yaml.safe_load(_DATA.read_text())
    robot = raw["robot"]
    count = len(robot["dof_names"])
    robot["soft_dof_pos_limit"] = 0.8
    robot["dof_pos_lower_limit_list"] = [-1.0] * count
    robot["dof_pos_upper_limit_list"] = [1.0] * count
    raw["simulator"]["_target_"] = "holosoma.simulator.isaacgym.isaacgym.IsaacGym"
    clip_path = tmp_path / "motion.npz"
    motion: dict[str, Any] = {
        "motion_file": str(clip_path),
        "body_name_ref": ["torso_link"],
        "body_names_to_track": ["pelvis", "torso_link"],
    }
    if noise is not None:
        motion["noise_to_initial_pose"] = dict(noise)
    raw["command"] = {
        "setup_terms": {"motion_command": {"func": _FUNC, "params": {"motion_config": motion}}},
        "reset_terms": {"motion_command": {"func": _FUNC, "params": {}}},
        "step_terms": {"motion_command": {"func": _FUNC, "params": {}}},
    }
    original = copy.deepcopy(raw)
    migrated = migrate_legacy_experiment_config(raw)
    assert raw == original
    assert migrate_legacy_experiment_config(migrated) == migrated
    assert migrated["command"]["reset_terms"] == raw["command"]["reset_terms"]
    assert migrated["command"]["step_terms"] == raw["command"]["step_terms"]

    _write_clip(clip_path, robot["body_names"], robot["dof_names"])
    checkpoint = tmp_path / "model.pt"
    torch.save({"experiment_config": raw}, checkpoint)
    config, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(checkpoint)))
    assert config.command is not None
    term_cfg = config.command.setup_terms["motion_command"]
    expected_noise = {**(noise or {}), "soft_dof_pos_limit": expected_factor}

    # Only the simulator transport is replaced: config construction, checkpoint/NPZ loading,
    # root selection, and the reset's clamp and tensor writes run production code.
    simulator = SimpleNamespace(
        _body_list=config.robot.body_names,
        body_names=config.robot.body_names,
        root_body_name="pelvis",
        dof_names=config.robot.dof_names,
        dof_pos_limits=torch.tensor([[-1.0, 1.0]] * count),
        dof_pos=torch.full((2, count), -0.3),
        dof_vel=torch.zeros(2, count),
        robot_root_states=torch.zeros(2, 13),
        scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
    )
    env: Any = SimpleNamespace(
        robot_config=config.robot,
        simulator=simulator,
        device="cpu",
        num_envs=2,
        dt=0.02,
        is_evaluating=True,
        viewer=False,
    )
    term = MotionCommand(term_cfg, env)
    term.setup()
    term.reset(torch.tensor([1]))

    torch.testing.assert_close(simulator.dof_pos[1], torch.full((count,), expected_target))
    torch.testing.assert_close(simulator.dof_pos[0], torch.full((count,), -0.3))
    assert term_cfg.params["motion_config"]["noise_to_initial_pose"] == expected_noise


@pytest.mark.parametrize("stage", ["setup_terms", "reset_terms", "step_terms"])
@pytest.mark.parametrize("func", [_FUNC, _FUNC.replace(":", ".")], ids=["colon", "dotted"])
@pytest.mark.parametrize("explicit_factor", [None, 0.6, 0.95], ids=["omitted", "explicit-factor", "explicit-default"])
def test_motion_soft_limit_migration_recognizes_canonical_paths_in_every_stage(
    stage: str, func: str, explicit_factor: float | None
) -> None:
    # get_class accepts the dotted spelling. CommandManager requires a colon, so this checks
    # saved-data migration without extending the manager's runtime path syntax.
    assert get_class(func.replace(":", ".")) is MotionCommand
    raw: dict[str, Any] = yaml.safe_load(_DATA.read_text())
    raw["robot"]["soft_dof_pos_limit"] = 0.8
    motion: dict[str, Any] = {
        "motion_file": "motion.npz",
        "body_name_ref": ["torso_link"],
        "body_names_to_track": ["pelvis", "torso_link"],
        "root_body": "pelvis",
    }
    if explicit_factor is not None:
        motion["noise_to_initial_pose"] = {"soft_dof_pos_limit": explicit_factor}
    raw["command"] = {stage: {"motion_command": {"func": func, "params": {"motion_config": motion}}}}
    original = copy.deepcopy(raw)

    migrated = migrate_legacy_experiment_config(raw)

    term = migrated["command"][stage]["motion_command"]
    assert term["func"] == func
    assert term["params"]["motion_config"]["noise_to_initial_pose"] == {
        "soft_dof_pos_limit": 0.8 if explicit_factor is None else explicit_factor
    }
    assert raw == original
    assert migrate_legacy_experiment_config(migrated) == migrated


@pytest.mark.parametrize(
    ("noise", "expected_factor"),
    [(None, 0.8), ({"dof_pos": 0.125}, 0.8), ({"soft_dof_pos_limit": 0.6}, 0.6)],
    ids=["sparse", "existing-noise-dict", "explicit-factor"],
)
@pytest.mark.parametrize("stage", ["setup_terms", "reset_terms", "step_terms"])
def test_extension_motion_command_keeps_legacy_soft_limit_relocation(
    stage: str, noise: dict[str, float] | None, expected_factor: float
) -> None:
    raw: dict[str, Any] = yaml.safe_load(_DATA.read_text())
    raw["robot"]["soft_dof_pos_limit"] = 0.8
    motion: dict[str, Any] = {"extension_option": "preserved"}
    if noise is not None:
        motion["noise_to_initial_pose"] = dict(noise)
    raw["command"] = {
        stage: {
            "extension": {"func": "custom.commands:MotionCommand", "params": {"motion_config": motion}},
            "reference": {"func": "custom.commands:MotionCommand", "params": {}},
        }
    }
    original = copy.deepcopy(raw)
    expected = copy.deepcopy(raw["command"])
    expected[stage]["extension"]["params"]["motion_config"]["noise_to_initial_pose"] = {
        **(noise or {}),
        "soft_dof_pos_limit": expected_factor,
    }

    migrated = migrate_legacy_experiment_config(raw)

    assert migrated["command"] == expected
    assert raw == original
    assert migrate_legacy_experiment_config(migrated) == migrated


@pytest.mark.parametrize(
    "func",
    [
        "custom.commands.MotionCommand",
        "holosoma.managers.command.terms.wbt:OtherMotionCommand",
    ],
)
def test_motion_migration_does_not_rewrite_unrelated_terms(func: str) -> None:
    raw: dict[str, Any] = yaml.safe_load(_DATA.read_text())
    raw["robot"]["soft_dof_pos_limit"] = 0.8
    # Neither a custom dotted path nor a different class had legacy soft-limit relocation.
    raw["command"] = {
        "setup_terms": {
            "other": {"func": func, "params": {"motion_config": {"noise_to_initial_pose": {"dof_pos": 0.125}}}}
        }
    }
    original = copy.deepcopy(raw)

    migrated = migrate_legacy_experiment_config(raw)

    assert migrated["command"] == original["command"]
    assert raw == original


@pytest.mark.parametrize(
    "func", [_FUNC, _FUNC.replace(":", "."), "custom.commands:MotionCommand"], ids=["colon", "dotted", "extension"]
)
def test_current_sparse_motion_config_is_unchanged(func: str) -> None:
    raw = EXPERIMENT_REGISTRY["g1_29dof_wbt"].to_serializable_dict()
    term = raw["command"]["setup_terms"]["motion_command"]
    term["func"] = func
    term["params"]["motion_config"].pop("noise_to_initial_pose")
    original = copy.deepcopy(raw)

    assert migrate_legacy_experiment_config(raw) == original
    assert raw == original
