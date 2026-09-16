from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from holosoma.managers.command.terms.wbt import MotionCommand, MotionLoader, MultiMotionLoader

pytestmark = pytest.mark.no_sim


def _write_motion(
    path: Path,
    joint_positions: list[float],
    *,
    fps: float = 10.0,
    motion_ends: list[bool] | None = None,
    motion_idxs: list[int] | None = None,
) -> None:
    num_frames = len(joint_positions)
    joint_pos = np.zeros((num_frames, 8), dtype=np.float32)
    joint_pos[:, 7] = joint_positions
    joint_vel = np.zeros((num_frames, 7), dtype=np.float32)
    body_pos_w = np.zeros((num_frames, 1, 3), dtype=np.float32)
    body_pos_w[:, 0, 0] = joint_positions
    body_quat_w = np.zeros((num_frames, 1, 4), dtype=np.float32)
    body_quat_w[:, 0, 0] = 1.0
    arrays: dict[str, Any] = {
        "fps": fps,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": np.zeros_like(body_pos_w),
        "body_ang_vel_w": np.zeros_like(body_pos_w),
        "body_names": np.array(["pelvis"]),
        "joint_names": np.array(["joint"]),
    }
    if motion_ends is not None:
        arrays["motion_ends"] = np.asarray(motion_ends)
    if motion_idxs is not None:
        arrays["motion_idxs"] = np.asarray(motion_idxs)
    np.savez(path, **arrays)


def _load(path: Path, *, recompute: bool = False) -> MotionLoader:
    return MotionLoader(
        str(path),
        robot_body_names=["pelvis"],
        robot_joint_names=["joint"],
        recompute_velocities_from_positions=recompute,
    )


def test_motion_loader_validates_boundary_metadata(tmp_path: Path) -> None:
    motion_file = tmp_path / "bad_ends.npz"
    _write_motion(motion_file, [0.0, 1.0], motion_ends=[True, False])

    with pytest.raises(ValueError, match="mark the final frame"):
        _load(motion_file)


def test_motion_loader_rejects_source_id_changes_inside_a_clip(tmp_path: Path) -> None:
    motion_file = tmp_path / "bad_ids.npz"
    _write_motion(
        motion_file,
        [0.0, 1.0, 2.0],
        motion_ends=[False, False, True],
        motion_idxs=[0, 1, 1],
    )

    with pytest.raises(ValueError, match="change only after"):
        _load(motion_file)


def test_velocity_recomputation_stays_within_source_clips(tmp_path: Path) -> None:
    motion_file = tmp_path / "combined_motion.npz"
    _write_motion(
        motion_file,
        [0.0, 1.0, 100.0, 101.0, 999.0],
        motion_ends=[False, True, False, True, True],
        motion_idxs=[0, 0, 1, 1, 2],
    )

    loader = _load(motion_file, recompute=True)

    torch.testing.assert_close(loader._joint_vel[:, 0], torch.tensor([10.0, 10.0, 10.0, 10.0, 0.0]))
    torch.testing.assert_close(loader._body_lin_vel_w[:, 0, 0], torch.tensor([10.0, 10.0, 10.0, 10.0, 0.0]))
    assert torch.count_nonzero(loader._body_ang_vel_w) == 0
    assert loader.motion_ends.tolist() == [False, True, False, True, True]
    assert loader.motion_idxs.tolist() == [0, 0, 1, 1, 2]
    assert loader.motion_filenames == ["combined_motion"]


def test_multi_motion_loader_rejects_invalid_shards(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shard_world_size"):
        MultiMotionLoader(str(tmp_path), ["pelvis"], ["joint"], shard_world_size=0)

    with pytest.raises(ValueError, match="shard_rank"):
        MultiMotionLoader(str(tmp_path), ["pelvis"], ["joint"], shard_rank=2, shard_world_size=2)


def test_multi_motion_loader_rejects_mixed_frame_rates(tmp_path: Path) -> None:
    _write_motion(tmp_path / "a.npz", [0.0, 1.0], fps=30.0)
    _write_motion(tmp_path / "b.npz", [0.0, 1.0], fps=60.0)

    with pytest.raises(ValueError, match="same fps"):
        MultiMotionLoader(str(tmp_path), ["pelvis"], ["joint"])


def test_multi_motion_loader_shards_files_round_robin(tmp_path: Path) -> None:
    for index in range(3):
        _write_motion(tmp_path / f"motion_{index}.npz", [0.0, 1.0])

    loader = MultiMotionLoader(
        str(tmp_path),
        ["pelvis"],
        ["joint"],
        shard_rank=1,
        shard_world_size=2,
    )

    assert loader.num_motions == 1
    assert loader.motion_filenames == ["motion_1"]
    assert loader.motion_ends.tolist() == [False, True]
    assert loader.motion_idxs.tolist() == [0, 0]


def test_motion_command_properties_respect_raw_body_order() -> None:
    command = cast("Any", MotionCommand.__new__(MotionCommand))
    command.time_steps = torch.tensor([0])
    command._root_body_index_in_motion = 1
    command._ref_body_index_in_motion = 0
    command.motion = SimpleNamespace(
        _body_pos_w=torch.tensor([[[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]]),
        _body_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]]),
        _body_lin_vel_w=torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        _body_ang_vel_w=torch.tensor([[[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]),
    )
    command._env = SimpleNamespace(
        simulator=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.tensor([[100.0, 0.0, 0.0]])))
    )

    torch.testing.assert_close(command.ref_pos_w, torch.tensor([[110.0, 0.0, 0.0]]))
    torch.testing.assert_close(command.root_pos_w, torch.tensor([[120.0, 0.0, 0.0]]))
    torch.testing.assert_close(command.ref_lin_vel_w, torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(command.root_ang_vel_w, torch.tensor([[4.0, 0.0, 0.0]]))


def test_motion_command_soft_resets_after_consuming_source_boundary() -> None:
    command = cast("Any", MotionCommand.__new__(MotionCommand))
    command.time_steps = torch.tensor([1, 2, 3])
    command.motion = SimpleNamespace(motion_ends=torch.tensor([False, False, True, False, True]))

    def reset(env_ids: torch.Tensor) -> None:
        command.time_steps[env_ids] = 0

    command.reset = Mock(side_effect=reset)
    simulator = SimpleNamespace(
        robot_root_states=torch.zeros(3, 13),
        dof_state=torch.zeros(3, 2),
        set_actor_root_state_tensor_robots=Mock(),
        set_dof_state_tensor_robots=Mock(),
        clear_contact_forces_history=Mock(),
        refresh_sim_tensors=Mock(),
    )
    observation_manager = SimpleNamespace(reset=Mock())
    command._env = SimpleNamespace(simulator=simulator, observation_manager=observation_manager)

    command._advance_or_resample_motions(torch.ones(3, dtype=torch.bool))

    # Newly reached terminal frames remain observable for one policy step;
    # only the terminal frame already consumed by env 1 is resampled.
    assert command.time_steps.tolist() == [2, 0, 4]
    ended_env_ids = command.reset.call_args.args[0]
    assert ended_env_ids.tolist() == [1]
    simulator.set_actor_root_state_tensor_robots.assert_called_once_with(ended_env_ids, simulator.robot_root_states)
    simulator.set_dof_state_tensor_robots.assert_called_once_with(ended_env_ids, simulator.dof_state)
    simulator.clear_contact_forces_history.assert_called_once_with(ended_env_ids)
    simulator.refresh_sim_tensors.assert_called_once_with()
    observation_manager.reset.assert_called_once_with(ended_env_ids)


def test_reference_height_buffer_survives_inference_rollout_then_normal_reset() -> None:
    command = cast("Any", MotionCommand.__new__(MotionCommand))
    command.time_steps = torch.tensor([0, 1])
    command._root_body_index_in_motion = 0
    command.motion = SimpleNamespace(
        _body_pos_w=torch.tensor(
            [
                [[0.0, 0.0, 0.2]],
                [[0.0, 0.0, 0.4]],
            ]
        )
    )
    command.running_ref_root_height = torch.tensor([1.0, 1.0])

    with torch.inference_mode():
        command._update_running_ref_root_height()

    assert not torch.is_inference(command.running_ref_root_height)
    torch.testing.assert_close(command.running_ref_root_height, torch.tensor([0.92, 0.94]))

    command.running_ref_root_height[torch.tensor([0])] = 0.5
    torch.testing.assert_close(command.running_ref_root_height, torch.tensor([0.5, 0.94]))
