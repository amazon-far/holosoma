"""Root selection through real NPZ loading, setup, transitions and simulator-state reset.

Only the simulator transport/FK boundary is faked. Body data deliberately disagrees with the raw
root DOF channels, the reference body, canonical column zero and raw NPZ column zero.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from holosoma.config_types.command import CommandTermCfg, MotionConfig
from holosoma.config_types.robot import RobotConfig, RobotInitState
from holosoma.config_values.robot import yam
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
from holosoma.utils.simulator_config import SimulatorType

pytestmark = pytest.mark.no_sim

_FUNC = "holosoma.managers.command.terms.wbt:MotionCommand"
_POS = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
_QUAT = torch.tensor([[0.0, 0.0, 0.6, 0.8], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.8, 0.6]])
_LIN = torch.tensor([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0], [17.0, 18.0, 19.0]])
_ANG = torch.tensor([[21.0, 22.0, 23.0], [24.0, 25.0, 26.0], [27.0, 28.0, 29.0]])


def _robot(body_names: list[str] | None = None) -> RobotConfig:
    return RobotConfig(
        dof_names=["hinge_b", "hinge_a"],
        body_names=body_names if body_names is not None else ["tip", "reference", "anchor"],
        init_state=RobotInitState(
            position=[0.0, 0.0, 0.25],
            default_joint_positions={"#all": 0.0},
            lin_vel=[0.1, 0.2, 0.3],
            ang_vel=[0.4, 0.5, 0.6],
        ),
        asset=dataclasses.replace(yam.asset, fix_base_link=False),
        body_tags={
            "#root": ["anchor"],
            "#mount": ["anchor"],
            "#nested": ["#mount"],
            "#pair": ["anchor", "reference"],
        },
    )


def _write_clip(
    path: Path,
    *,
    bodies: list[str] | None = None,
    joints: list[str] | None = None,
    root_name: str = "anchor",
    shift: float = 0.0,
) -> None:
    names = bodies if bodies is not None else ["world", "anchor", "tip", "reference"]
    joint_names = joints if joints is not None else ["unused", "hinge_a", "hinge_b"]
    body_pos = torch.full((3, len(names), 3), -50.0)
    body_quat = torch.zeros(3, len(names), 4)
    body_quat[..., 3] = 1.0
    body_lin = torch.full_like(body_pos, -60.0)
    body_ang = torch.full_like(body_pos, -70.0)
    for index, name in enumerate(names):
        if name == root_name:
            body_pos[:, index] = _POS + shift
            body_quat[:, index] = _QUAT
            body_lin[:, index] = _LIN + shift
            body_ang[:, index] = _ANG + shift
    # These root channels are intentionally inconsistent with every body channel.
    joint_pos = torch.full((3, len(joint_names) + 7), 999.0)
    joint_vel = torch.full((3, len(joint_names) + 6), -999.0)
    for index, name in enumerate(joint_names):
        joint_pos[:, index + 7] = 0.4 if name == "hinge_b" else 0.2
        joint_vel[:, index + 6] = 4.0 if name == "hinge_b" else 2.0
    np.savez(
        path,
        fps=50,
        body_names=names,
        joint_names=joint_names,
        joint_pos=joint_pos.numpy(),
        joint_vel=joint_vel.numpy(),
        body_pos_w=body_pos.numpy(),
        body_quat_w=body_quat[..., [3, 0, 1, 2]].numpy(),
        body_lin_vel_w=body_lin.numpy(),
        body_ang_vel_w=body_ang.numpy(),
    )


class _Simulator:
    """Tensor transport and deterministic FK for an articulation with a named root body."""

    def __init__(self, robot: RobotConfig, root_name: str = "anchor") -> None:
        self._body_list = robot.body_names
        self.body_names = robot.body_names
        self.root_body_name = root_name
        self.dof_names = robot.dof_names
        self.dof_pos_limits = torch.tensor([[-1.0, 1.0]] * len(robot.dof_names))
        self.robot_root_states = torch.zeros(2, 13)
        self.robot_root_states[:, 6] = 1.0
        self.dof_pos = torch.full((2, len(robot.dof_names)), -0.7)
        self.dof_vel = torch.full_like(self.dof_pos, -0.8)
        self.scene = SimpleNamespace(env_origins=torch.tensor([[10.0, 20.0, 0.0], [30.0, 40.0, 0.0]]))
        self.root_writes: list[torch.Tensor] = []
        self.dof_writes = 0
        self._offsets = torch.tensor(
            [
                [0.0, 0.0, 0.0] if name == root_name else [float(i + 1), 0.0, 0.0]
                for i, name in enumerate(robot.body_names)
            ]
        )
        self.write_state_updates()

    def get_simulator_type(self) -> SimulatorType:
        return SimulatorType.ISAACSIM

    def set_actor_root_state_tensor_robots(self) -> None:
        self.root_writes.append(self.robot_root_states.clone())

    def set_dof_state_tensor_robots(self) -> None:
        self.dof_writes += 1

    def write_state_updates(self) -> None:
        roots = self.robot_root_states
        count = len(self._body_list)
        self._rigid_body_pos = roots[:, None, :3] + self._offsets
        self._rigid_body_rot = roots[:, None, 3:7].expand(-1, count, -1).clone()
        self._rigid_body_vel = roots[:, None, 7:10].expand(-1, count, -1).clone()
        self._rigid_body_ang_vel = roots[:, None, 10:13].expand(-1, count, -1).clone()

    def refresh_sim_tensors(self) -> None:
        pass


def _environment(robot: RobotConfig, simulator: _Simulator) -> Any:
    return SimpleNamespace(
        robot_config=robot,
        simulator=simulator,
        num_envs=2,
        device="cpu",
        dt=0.02,
        viewer=False,
        is_evaluating=True,
        default_dof_pos_base=torch.tensor([list(robot.default_dof_positions().values())]),
    )


def _motion(path: Path, **options: Any) -> MotionConfig:
    return MotionConfig(
        motion_file=str(path), body_name_ref=["reference"], body_names_to_track=["reference", "anchor"], **options
    )


def _command(
    robot: RobotConfig, cfg: MotionConfig, *, physical_root: str = "anchor"
) -> tuple[MotionCommand, _Simulator]:
    simulator = _Simulator(robot, root_name=physical_root)
    term = MotionCommand(CommandTermCfg(func=_FUNC, params={"motion_config": cfg}), _environment(robot, simulator))
    term.setup()
    return term, simulator


def _assert_root(term: MotionCommand, positions: torch.Tensor, frames: torch.Tensor, shift: float = 0.0) -> None:
    torch.testing.assert_close(term.root_pos_w, positions)
    torch.testing.assert_close(term.root_quat_w, _QUAT[frames])
    torch.testing.assert_close(term.root_lin_vel_w, _LIN[frames] + shift)
    torch.testing.assert_close(term.root_ang_vel_w, _ANG[frames] + shift)


@pytest.mark.parametrize("selector", [None, "anchor", "#mount", "#nested"])
@pytest.mark.parametrize("body_names", [["tip", "reference", "anchor"], ["anchor", "reference"]])
def test_root_selection_drives_accessors_and_reset(tmp_path: Path, selector: str | None, body_names: list[str]) -> None:
    path = tmp_path / "motion.npz"
    _write_clip(path)
    options = {} if selector is None else {"root_body": selector}
    cfg = _motion(path, **options)
    # The actual constructor used by CLI/checkpoint term params also sees the new field.
    cfg = MotionConfig(**dataclasses.asdict(cfg))
    term, simulator = _command(_robot(body_names), cfg)

    frames = torch.tensor([0, 2])
    term.time_steps[:] = frames
    origins = simulator.scene.env_origins
    _assert_root(term, _POS[frames] + origins, frames)
    torch.testing.assert_close(term.ref_pos_w, torch.full((2, 3), -50.0) + origins)
    torch.testing.assert_close(term.joint_pos, torch.tensor([[0.4, 0.2], [0.4, 0.2]]))
    torch.testing.assert_close(term.joint_vel, torch.tensor([[4.0, 2.0], [4.0, 2.0]]))

    untouched = simulator.robot_root_states[1].clone()
    term.reset(torch.tensor([0]))

    expected = torch.cat((_POS[0] + origins[0], _QUAT[0], _LIN[0], _ANG[0]))
    torch.testing.assert_close(simulator.robot_root_states[0], expected)
    torch.testing.assert_close(simulator.robot_root_states[1], untouched)


def _two_clips(tmp_path: Path, *, different_extras: bool = False) -> str:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _write_clip(first / "a.npz")
    _write_clip(
        second / "b.npz",
        bodies=["reference", "tip", "anchor"] if different_extras else ["reference", "tip", "anchor", "world"],
        joints=["hinge_b", "hinge_a"] if different_extras else ["hinge_b", "unused", "hinge_a"],
        shift=100.0,
    )
    return f"{first},{second}"


@pytest.mark.parametrize("selector", ["reference", "#nested"])
def test_explicit_selector_can_choose_a_body_other_than_root_tag(tmp_path: Path, selector: str) -> None:
    path = tmp_path / "alternate-root.npz"
    _write_clip(path, root_name="reference")
    robot = _robot()
    robot.body_tags["#mount"] = ["reference"]
    term, simulator = _command(robot, _motion(path, root_body=selector), physical_root="reference")

    term.time_steps[:] = torch.tensor([0, 2])
    _assert_root(term, _POS[[0, 2]] + simulator.scene.env_origins, torch.tensor([0, 2]))
    term.reset(None)
    expected = torch.cat((_POS[0], _QUAT[0], _LIN[0], _ANG[0])).repeat(2, 1)
    expected[:, :3] += simulator.scene.env_origins
    torch.testing.assert_close(simulator.robot_root_states, expected)


@pytest.mark.parametrize("different_extras", [False, True])
def test_each_motion_uses_its_own_column_mapping(tmp_path: Path, different_extras: bool) -> None:
    term, simulator = _command(
        _robot(), _motion(tmp_path / "unused.npz", motion_dir=_two_clips(tmp_path, different_extras=different_extras))
    )
    term.time_steps[:] = torch.tensor([3, 5])
    _assert_root(term, _POS[[0, 2]] + 100.0 + simulator.scene.env_origins, torch.tensor([0, 2]), shift=100.0)
    torch.testing.assert_close(term.joint_pos, torch.tensor([[0.4, 0.2], [0.4, 0.2]]))
    torch.testing.assert_close(term.joint_vel, torch.tensor([[4.0, 2.0], [4.0, 2.0]]))
    assert term.motion.motion_start_idx.tolist() == [0, 3]
    assert term.motion.motion_end_idx.tolist() == [3, 6]

    # Seeded sampling is real: enough env resets visit both clips without mocking internal sampling.
    torch.manual_seed(9)
    seen: set[int] = set()
    for _ in range(8):
        term.reset(None)
        for row, motion_id in enumerate(term.motion_ids.tolist()):
            seen.add(motion_id)
            shift = 100.0 * motion_id
            expected = torch.cat(
                (_POS[0] + shift + simulator.scene.env_origins[row], _QUAT[0], _LIN[0] + shift, _ANG[0] + shift)
            )
            torch.testing.assert_close(simulator.robot_root_states[row], expected)
    assert seen == {0, 1}


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("prepend", [False, True])
def test_transition_anchors_to_selected_body_and_restores_simulator(tmp_path: Path, multi: bool, prepend: bool) -> None:
    path = tmp_path / "motion.npz"
    _write_clip(path)
    cfg = _motion(
        path,
        root_body="#nested",
        motion_dir=_two_clips(tmp_path) if multi else "",
        enable_default_pose_prepend=prepend,
        enable_default_pose_append=not prepend,
        default_pose_prepend_duration_s=0.04,
        default_pose_append_duration_s=0.04,
    )
    robot = _robot()
    simulator = _Simulator(robot)
    root_before, pos_before, vel_before = (
        simulator.robot_root_states.clone(),
        simulator.dof_pos.clone(),
        simulator.dof_vel.clone(),
    )
    term = MotionCommand(CommandTermCfg(func=_FUNC, params={"motion_config": cfg}), _environment(robot, simulator))

    term.setup()

    frame = 0 if prepend else 2
    shift = 100.0 if multi and not prepend else 0.0
    default_pos = torch.tensor([_POS[frame, 0] + shift, _POS[frame, 1] + shift, 0.25])
    default_lin = torch.tensor([0.1, 0.2, 0.3])
    default_ang = torch.tensor([0.4, 0.5, 0.6])
    expected_write = torch.cat((default_pos + simulator.scene.env_origins[0], _QUAT[frame], default_lin, default_ang))
    assert len(simulator.root_writes) == 2 and simulator.dof_writes == 2
    torch.testing.assert_close(simulator.root_writes[0][0], expected_write)
    torch.testing.assert_close(simulator.robot_root_states, root_before)
    torch.testing.assert_close(simulator.dof_pos, pos_before)
    torch.testing.assert_close(simulator.dof_vel, vel_before)
    term.time_steps[:] = 0 if prepend else term.motion.time_step_total - 1
    torch.testing.assert_close(term.root_pos_w, default_pos.expand(2, -1) + simulator.scene.env_origins)
    torch.testing.assert_close(term.root_quat_w, _QUAT[frame].expand(2, -1))
    torch.testing.assert_close(term.root_lin_vel_w, default_lin.expand(2, -1))
    torch.testing.assert_close(term.root_ang_vel_w, default_ang.expand(2, -1))
    term.time_steps[:] = 1 if prepend else term.motion.time_step_total - 2
    torch.testing.assert_close(
        term.root_pos_w, ((default_pos + _POS[frame] + shift) / 2).expand(2, -1) + simulator.scene.env_origins
    )
    torch.testing.assert_close(term.root_lin_vel_w, ((default_lin + _LIN[frame] + shift) / 2).expand(2, -1))
    torch.testing.assert_close(term.root_ang_vel_w, ((default_ang + _ANG[frame] + shift) / 2).expand(2, -1))


@pytest.mark.parametrize("selector", ["", "missing", "#missing", "#pair", "#empty", "#root"])
def test_invalid_selector_fails_before_motion_loading_or_simulator_writes(tmp_path: Path, selector: str) -> None:
    robot = _robot()
    # Mutation after config construction exercises runtime validation too.
    robot.body_tags["#empty"] = []
    robot.body_tags.pop("#root")
    simulator = _Simulator(robot)
    before = simulator.robot_root_states.clone()
    cfg = _motion(tmp_path / "does-not-exist.npz", root_body=selector, enable_default_pose_prepend=True)
    term = MotionCommand(CommandTermCfg(func=_FUNC, params={"motion_config": cfg}), _environment(robot, simulator))

    with pytest.raises(ValueError, match=r"MotionConfig.root_body=") as error:
        term.setup()

    assert repr(selector) in str(error.value)
    if selector in ("#pair", "#empty"):
        assert "physical root" in str(error.value)
    assert simulator.root_writes == [] and simulator.dof_writes == 0
    torch.testing.assert_close(simulator.robot_root_states, before)


@pytest.mark.parametrize("roots", [[], ["anchor", "anchor"]])
def test_motion_file_requires_unique_selected_body_before_transition_writes(tmp_path: Path, roots: list[str]) -> None:
    path = tmp_path / "bad.npz"
    _write_clip(path, bodies=["reference", "tip", *roots])
    robot = _robot()
    simulator = _Simulator(robot)
    term = MotionCommand(
        CommandTermCfg(func=_FUNC, params={"motion_config": _motion(path, enable_default_pose_prepend=True)}),
        _environment(robot, simulator),
    )

    with pytest.raises(ValueError, match="exactly one entry for 'anchor'"):
        term.setup()

    assert simulator.root_writes == [] and simulator.dof_writes == 0


def test_legacy_checkpoint_load_then_reset_preserves_old_first_body(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[3] / "utils/tests/data/legacy_g1_29dof_experiment_config.yaml"
    raw: dict[str, Any] = yaml.safe_load(fixture.read_text())
    old_root = raw["robot"]["body_names"][0]
    # A conflicting tag and torso reference must not override the former runtime column-zero choice.
    raw["robot"]["body_tags"] = {"#root": ["torso_link"]}
    path = tmp_path / "legacy-motion.npz"
    _write_clip(
        path,
        bodies=list(reversed(raw["robot"]["body_names"])),
        joints=list(reversed(raw["robot"]["dof_names"])),
        root_name=old_root,
    )
    raw["command"] = {
        "setup_terms": {
            "motion_command": {
                "func": _FUNC,
                "params": {
                    "motion_config": {
                        "motion_file": str(path),
                        "body_name_ref": ["torso_link"],
                        "body_names_to_track": [old_root, "torso_link"],
                    }
                },
            }
        }
    }
    checkpoint = tmp_path / "model.pt"
    torch.save({"experiment_config": raw}, checkpoint)

    loaded, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(checkpoint)))
    assert loaded.command is not None
    term_cfg = loaded.command.setup_terms["motion_command"]
    assert term_cfg.params["motion_config"]["root_body"] == old_root
    # Reordering after loading cannot change the migrated exact body's identity.
    robot = dataclasses.replace(loaded.robot, body_names=list(reversed(loaded.robot.body_names)))
    simulator = _Simulator(robot, root_name=old_root)
    term = MotionCommand(term_cfg, _environment(robot, simulator))
    term.setup()
    term.reset(None)

    expected = torch.cat((_POS[0], _QUAT[0], _LIN[0], _ANG[0])).repeat(2, 1)
    expected[:, :3] += simulator.scene.env_origins
    torch.testing.assert_close(simulator.robot_root_states, expected)


def test_auto_root_needs_no_root_tag(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    _write_clip(path)
    robot = _robot()
    robot.body_tags.pop("#root")
    term, _ = _command(robot, _motion(path))
    assert term.root_body_index == robot.body_names.index("anchor")


def test_nonroot_selector_fails_before_motion_loading_or_writes(tmp_path: Path) -> None:
    robot = _robot()
    simulator = _Simulator(robot)
    term = MotionCommand(
        CommandTermCfg(func=_FUNC, params={"motion_config": _motion(tmp_path / "missing.npz", root_body="reference")}),
        _environment(robot, simulator),
    )
    with pytest.raises(ValueError, match="physical root 'anchor'.*reference"):
        term.setup()
    assert simulator.root_writes == [] and simulator.dof_writes == 0
