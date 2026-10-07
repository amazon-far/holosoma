"""CPU regression tests for the WBT MotionCommand, driven through a fake env (no simulator)."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from holosoma.config_types.command import CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig
from holosoma.managers.command.terms.wbt import AdaptiveTimestepsSampler, MotionCommand
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType

pytestmark = pytest.mark.no_sim

# Like the shipped motions (e.g. sub3_largebox_003_mj.npz), the motion file stores its bodies in
# its own order behind a static "world" body, while the robot lists the pelvis first.
MOTION_BODIES = ["world", "torso_link", "left_foot", "pelvis"]
ROBOT_BODIES = ["pelvis", "left_foot", "torso_link"]
MOTION_JOINTS = ["j1", "j0"]
ROBOT_JOINTS = ["j0", "j1"]

BODY_POS = {
    "world": (0.0, 0.0, 0.0),
    "torso_link": (1.1, 2.0, 1.0),
    "pelvis": (1.0, 2.0, 0.8),
    "left_foot": (1.0, 2.1, 0.05),
}
BODY_YAW = {"world": 0.0, "torso_link": math.radians(40.0), "pelvis": math.radians(10.0), "left_foot": 0.3}
BODY_LIN_VEL = {
    "world": (0.0, 0.0, 0.0),
    "torso_link": (0.7, 0.8, 0.9),
    "pelvis": (0.1, 0.2, 0.3),
    "left_foot": (0.0, 0.1, 0.0),
}
BODY_ANG_VEL = {
    "world": (0.0, 0.0, 0.0),
    "torso_link": (1.0, 1.1, 1.2),
    "pelvis": (0.4, 0.5, 0.6),
    "left_foot": (0.0, 0.0, 0.1),
}
DT = 0.02


def _yaw_quat_xyzw(yaw: float) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def _write_motion(
    path: Path, num_frames: int, motion_ends: list[int] | None = None, x_drift_per_frame: float = 0.0
) -> Path:
    """Write a holosoma-format motion; joint values encode the frame index (j0 = 0.01 t, j1 = 0.02 t)."""
    t = np.arange(num_frames, dtype=np.float64)
    joint_pos = np.zeros((num_frames, 7 + len(MOTION_JOINTS)))
    joint_pos[:, 7 + MOTION_JOINTS.index("j0")] = 0.01 * t
    joint_pos[:, 7 + MOTION_JOINTS.index("j1")] = 0.02 * t
    joint_vel = np.zeros((num_frames, 6 + len(MOTION_JOINTS)))

    body_pos = np.zeros((num_frames, len(MOTION_BODIES), 3))
    body_quat_wxyz = np.zeros((num_frames, len(MOTION_BODIES), 4))
    body_lin_vel = np.zeros((num_frames, len(MOTION_BODIES), 3))
    body_ang_vel = np.zeros((num_frames, len(MOTION_BODIES), 3))
    for i, name in enumerate(MOTION_BODIES):
        body_pos[:, i] = BODY_POS[name]
        if name != "world":
            body_pos[:, i, 0] += x_drift_per_frame * t
        yaw = BODY_YAW[name]
        body_quat_wxyz[:, i] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        body_lin_vel[:, i] = BODY_LIN_VEL[name]
        body_ang_vel[:, i] = BODY_ANG_VEL[name]

    data: dict[str, Any] = {
        "fps": np.array(1.0 / DT),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat_wxyz,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
        "body_names": np.array(MOTION_BODIES),
        "joint_names": np.array(MOTION_JOINTS),
    }
    if motion_ends is not None:
        ends = np.zeros(num_frames, dtype=bool)
        ends[motion_ends] = True
        data["motion_ends"] = ends
        data["motion_idxs"] = np.cumsum(np.concatenate([[0], ends[:-1]])).astype(np.int64)
    np.savez(path, **data)
    return path


def _make_env(num_envs: int, push_state: Any = None) -> SimpleNamespace:
    num_bodies = len(ROBOT_BODIES)
    simulator = SimpleNamespace(
        _body_list=list(ROBOT_BODIES),
        dof_names=list(ROBOT_JOINTS),
        get_simulator_type=lambda: SimulatorType.MUJOCO,
        scene=SimpleNamespace(
            env_origins=torch.arange(num_envs, dtype=torch.float32)[:, None] * torch.tensor([5.0, -3.0, 0.0])
        ),
        dof_pos=torch.zeros(num_envs, len(ROBOT_JOINTS)),
        dof_vel=torch.zeros(num_envs, len(ROBOT_JOINTS)),
        dof_state=torch.zeros(num_envs, len(ROBOT_JOINTS), 2),
        dof_pos_limits=torch.tensor([[-10.0, 10.0]] * len(ROBOT_JOINTS)),
        robot_root_states=torch.zeros(num_envs, 13),
        # Rigid-body buffers stay stale (zeros), as on IsaacGym right after a teleport.
        _rigid_body_pos=torch.zeros(num_envs, num_bodies, 3),
        _rigid_body_rot=torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(num_envs, num_bodies, 1),
        _rigid_body_vel=torch.zeros(num_envs, num_bodies, 3),
        _rigid_body_ang_vel=torch.zeros(num_envs, num_bodies, 3),
        set_actor_root_state_tensor_robots=lambda *_: None,
        set_dof_state_tensor_robots=lambda *_: None,
        refresh_sim_tensors=lambda: None,
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        dt=DT,
        viewer=False,
        is_evaluating=False,
        simulator=simulator,
        episode_length_buf=torch.ones(num_envs, dtype=torch.long),
        termination_manager=SimpleNamespace(terminated=torch.zeros(num_envs, dtype=torch.bool)),
        randomization_manager=SimpleNamespace(get_state=lambda _name: push_state),
        robot_config=SimpleNamespace(
            init_state=SimpleNamespace(
                pos=[0.0, 0.0, 0.75], rot=[0.0, 0.0, 0.0, 1.0], lin_vel=[0.0, 0.0, 0.0], ang_vel=[0.0, 0.0, 0.0]
            )
        ),
        default_dof_pos_base=torch.zeros(1, len(ROBOT_JOINTS)),
    )


def _make_command(env: SimpleNamespace, params: dict[str, Any] | None = None, **motion_kwargs: Any) -> MotionCommand:
    motion_kwargs.setdefault("motion_file", "")
    motion_cfg = MotionConfig(
        body_name_ref=["torso_link"],
        body_names_to_track=list(ROBOT_BODIES),
        **motion_kwargs,
    )
    cfg = CommandTermCfg(
        func="holosoma.managers.command.terms.wbt:MotionCommand",
        params={"motion_config": motion_cfg, **(params or {})},
    )
    command = MotionCommand(cfg, cast("Any", env))
    command.setup()
    return command


@pytest.fixture
def captured_default_roots(monkeypatch: pytest.MonkeyPatch) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Stand in for the IsaacSim-only FK used by default-pose transitions; record the anchor root."""
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture_body_states(
        self: MotionCommand,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        calls.append((root_pos.reshape(3).clone(), root_quat.reshape(4).clone()))
        num_bodies = len(ROBOT_BODIES)
        return {
            "pos": root_pos.reshape(1, 3).expand(num_bodies, 3).clone(),
            "quat": root_quat.reshape(1, 4).expand(num_bodies, 4).clone(),
            "lin_vel": torch.zeros(num_bodies, 3),
            "ang_vel": torch.zeros(num_bodies, 3),
        }

    monkeypatch.setattr(MotionCommand, "_capture_body_states", capture_body_states)
    return calls


def _motion_body(name: str, attr: dict[str, tuple[float, float, float]], num_envs: int) -> torch.Tensor:
    return torch.tensor(attr[name], dtype=torch.float32).expand(num_envs, 3)


def _assert_quat_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # q and -q are the same rotation.
    dot = (actual * expected).sum(dim=-1).abs()
    assert torch.allclose(dot, torch.ones_like(dot), atol=1e-5)


#########################################################################################################
## Motion-side root/ref bodies are read in the motion file's body order
#########################################################################################################
def test_root_and_ref_read_motion_bodies_by_name(tmp_path: Path) -> None:
    env = _make_env(num_envs=3)
    command = _make_command(env, motion_file=str(_write_motion(tmp_path / "clip.npz", 12)))
    command.time_steps[:] = 4
    origins = env.simulator.scene.env_origins

    assert torch.allclose(command.root_pos_w, _motion_body("pelvis", BODY_POS, 3) + origins)
    _assert_quat_close(command.root_quat_w, _yaw_quat_xyzw(BODY_YAW["pelvis"]).expand(3, 4))
    assert torch.allclose(command.root_lin_vel_w, _motion_body("pelvis", BODY_LIN_VEL, 3))
    assert torch.allclose(command.root_ang_vel_w, _motion_body("pelvis", BODY_ANG_VEL, 3))

    assert torch.allclose(command.ref_pos_w, _motion_body("torso_link", BODY_POS, 3) + origins)
    _assert_quat_close(command.ref_quat_w, _yaw_quat_xyzw(BODY_YAW["torso_link"]).expand(3, 4))
    assert torch.allclose(command.ref_lin_vel_w, _motion_body("torso_link", BODY_LIN_VEL, 3))
    assert torch.allclose(command.ref_ang_vel_w, _motion_body("torso_link", BODY_ANG_VEL, 3))

    # Same bodies as the name-mapped tracked-body accessors.
    torso = ROBOT_BODIES.index("torso_link")
    assert torch.allclose(command.ref_pos_w, command.body_pos_w[:, torso])
    assert torch.allclose(command.root_pos_w, command.body_pos_w[:, ROBOT_BODIES.index("pelvis")])


def test_reset_places_robot_on_motion_root(tmp_path: Path) -> None:
    env = _make_env(num_envs=4)
    command = _make_command(env, motion_file=str(_write_motion(tmp_path / "clip.npz", 12)))
    command.reset(None)

    sim = env.simulator
    assert torch.allclose(sim.robot_root_states[:, :3], _motion_body("pelvis", BODY_POS, 4) + sim.scene.env_origins)
    _assert_quat_close(sim.robot_root_states[:, 3:7], _yaw_quat_xyzw(BODY_YAW["pelvis"]).expand(4, 4))
    assert torch.allclose(sim.robot_root_states[:, 7:10], _motion_body("pelvis", BODY_LIN_VEL, 4))
    assert torch.allclose(sim.robot_root_states[:, 10:13], _motion_body("pelvis", BODY_ANG_VEL, 4))
    frames = command.time_steps.float()
    assert torch.allclose(sim.dof_pos, torch.stack([0.01 * frames, 0.02 * frames], dim=1))


@pytest.mark.parametrize("prepend", [True, False])
def test_default_pose_transition_anchors_on_motion_root(
    tmp_path: Path, captured_default_roots: list[tuple[torch.Tensor, torch.Tensor]], prepend: bool
) -> None:
    num_frames, drift = 12, 0.05
    motion_file = _write_motion(tmp_path / "clip.npz", num_frames, x_drift_per_frame=drift)
    _make_command(
        _make_env(num_envs=2),
        motion_file=str(motion_file),
        enable_default_pose_prepend=prepend,
        enable_default_pose_append=not prepend,
        default_pose_prepend_duration_s=0.1,
        default_pose_append_duration_s=0.1,
    )

    assert len(captured_default_roots) == 1
    root_pos, root_quat = captured_default_roots[0]
    anchor_frame = 0 if prepend else num_frames - 1
    pelvis = BODY_POS["pelvis"]
    assert torch.allclose(root_pos[:2], torch.tensor([pelvis[0] + drift * anchor_frame, pelvis[1]]))
    assert root_pos[2].item() == pytest.approx(0.75)
    _assert_quat_close(root_quat, _yaw_quat_xyzw(BODY_YAW["pelvis"]))


#########################################################################################################
## Relative body targets: the motion side anchors on the same body as the robot side
#########################################################################################################
def _place_robot_on_motion_root(env: SimpleNamespace) -> None:
    sim = env.simulator
    sim.robot_root_states[:, :3] = _motion_body("pelvis", BODY_POS, env.num_envs) + sim.scene.env_origins
    sim.robot_root_states[:, 3:7] = _yaw_quat_xyzw(BODY_YAW["pelvis"])


def test_relative_targets_have_no_offset_at_episode_start(tmp_path: Path) -> None:
    env = _make_env(num_envs=3)
    command = _make_command(env, motion_file=str(_write_motion(tmp_path / "clip.npz", 12)))
    command.time_steps[:] = 2
    _place_robot_on_motion_root(env)
    env.episode_length_buf[:] = 0

    command.step()

    assert torch.allclose(command.body_pos_relative_w, command.body_pos_w, atol=1e-5)
    _assert_quat_close(command.body_quat_relative_w, command.body_quat_w)


def test_relative_targets_have_no_offset_after_motion_end_resample(tmp_path: Path) -> None:
    num_frames = 12
    env = _make_env(num_envs=3)
    command = _make_command(
        env, motion_file=str(_write_motion(tmp_path / "clip.npz", num_frames)), resample_on_motion_end=True
    )
    command.time_steps[:] = num_frames - 2  # step() advances onto the clip-end frame and resamples
    env.episode_length_buf[:] = 7

    command.step()

    assert command.motion_end_reset.all()
    assert torch.allclose(command.body_pos_relative_w, command.body_pos_w, atol=1e-5)
    _assert_quat_close(command.body_quat_relative_w, command.body_quat_w)


#########################################################################################################
## Adaptive sampler: adaptive_uniform_ratio sets the uniform floor
#########################################################################################################
@pytest.mark.parametrize("ratio", [0.1, 0.5])
def test_adaptive_uniform_ratio_sets_uniform_floor(ratio: float) -> None:
    sampler = AdaptiveTimestepsSampler(99 * 50, "cpu", 50, adaptive_uniform_ratio=ratio)
    assert sampler.num_bins == 100
    sampler.bin_failed_count[0] = 1.0

    expected = torch.full((100,), ratio / 100)
    expected[0] += 1.0
    assert torch.allclose(sampler.sampling_probabilities, expected / expected.sum())


def test_adaptive_uniform_ratio_is_plumbed_from_motion_config(tmp_path: Path) -> None:
    command = _make_command(
        _make_env(num_envs=2),
        motion_file=str(_write_motion(tmp_path / "clip.npz", 12)),
        use_adaptive_timesteps_sampler=True,
        adaptive_uniform_ratio=0.5,
    )
    assert command.adaptive_timesteps_sampler.adaptive_uniform_ratio == 0.5


#########################################################################################################
## Reset root-velocity noise
#########################################################################################################
def _root_vel_offsets(command: MotionCommand, env: SimpleNamespace) -> torch.Tensor:
    command.reset(None)
    root_vel = env.simulator.robot_root_states[:, 7:13]
    motion_vel = torch.cat(
        [_motion_body("pelvis", BODY_LIN_VEL, env.num_envs), _motion_body("pelvis", BODY_ANG_VEL, env.num_envs)], dim=1
    )
    return root_vel - motion_vel


def _noise_cfg(scale: float, vel: float) -> NoiseToInitialPoseConfig:
    return NoiseToInitialPoseConfig(overall_noise_scale=scale, root_lin_vel=[vel] * 3, root_ang_vel=[vel] * 3)


def _push_state(enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(enabled=enabled, max_push_vel=torch.full((6,), 5.0))


def test_reset_velocity_noise_defaults_to_configured_noise(tmp_path: Path) -> None:
    motion_file = str(_write_motion(tmp_path / "clip.npz", 12))

    env = _make_env(num_envs=64, push_state=_push_state(enabled=True))
    command = _make_command(env, motion_file=motion_file, noise_to_initial_pose=_noise_cfg(scale=0.0, vel=0.5))
    assert torch.equal(_root_vel_offsets(command, env), torch.zeros(64, 6))

    env = _make_env(num_envs=64, push_state=_push_state(enabled=True))
    command = _make_command(env, motion_file=motion_file, noise_to_initial_pose=_noise_cfg(scale=0.5, vel=0.02))
    offsets = _root_vel_offsets(command, env)
    assert offsets.abs().max() <= 0.01 + 1e-6
    assert offsets.abs().max() > 0.0


@pytest.mark.parametrize(("enabled", "scale"), [(False, 1.0), (True, 0.0)])
def test_push_velocity_reset_noise_respects_enabled_and_scale(tmp_path: Path, enabled: bool, scale: float) -> None:
    env = _make_env(num_envs=64, push_state=_push_state(enabled=enabled))
    command = _make_command(
        env,
        params={"use_configured_root_velocity_noise": False},
        motion_file=str(_write_motion(tmp_path / "clip.npz", 12)),
        noise_to_initial_pose=_noise_cfg(scale=scale, vel=0.0),
    )
    assert torch.equal(_root_vel_offsets(command, env), torch.zeros(64, 6))


def test_push_velocity_reset_noise_scales_max_push_vel(tmp_path: Path) -> None:
    env = _make_env(num_envs=64, push_state=_push_state(enabled=True))
    command = _make_command(
        env,
        params={"use_configured_root_velocity_noise": False},
        motion_file=str(_write_motion(tmp_path / "clip.npz", 12)),
        noise_to_initial_pose=_noise_cfg(scale=0.1, vel=0.0),
    )
    offsets = _root_vel_offsets(command, env)
    assert offsets.abs().max() <= 0.5 + 1e-5
    assert offsets.abs().max() > 0.0


#########################################################################################################
## Default-pose transitions keep the per-frame clip markers aligned with the motion
#########################################################################################################
def _assert_clip_markers_aligned(command: MotionCommand) -> None:
    motion = command.motion
    assert motion.motion_ends.shape == (motion.time_step_total,)
    assert motion.motion_idxs.shape == (motion.time_step_total,)
    assert bool(motion.motion_ends[-1])
    for end in motion.motion_end_idx.tolist():
        assert bool(motion.motion_ends[end - 1])


def _walk(command: MotionCommand, env: SimpleNamespace, num_steps: int) -> None:
    for _ in range(num_steps):
        command.step()
        assert int(command.time_steps.max()) < command.motion.time_step_total
        env.episode_length_buf += 1


@pytest.mark.parametrize(("prepend", "append"), [(True, False), (False, True), (True, True)])
def test_single_file_transitions_extend_clip_markers(
    tmp_path: Path, captured_default_roots: list[tuple[torch.Tensor, torch.Tensor]], prepend: bool, append: bool
) -> None:
    # One combined file holding two source clips: frames 0-3 and 4-9.
    motion_file = _write_motion(tmp_path / "combined.npz", 10, motion_ends=[3, 9])
    env = _make_env(num_envs=64)
    torch.manual_seed(0)
    command = _make_command(
        env,
        motion_file=str(motion_file),
        enable_default_pose_prepend=prepend,
        enable_default_pose_append=append,
        default_pose_prepend_duration_s=0.1,
        default_pose_append_duration_s=0.1,
    )
    added = 5 * (int(prepend) + int(append))
    assert command.motion.time_step_total == 10 + added
    _assert_clip_markers_aligned(command)

    # The first clip-end marker moves with a prepend; an append moves the last clip's end to the new last frame.
    offset = 5 if prepend else 0
    expected_ends = [3 + offset, command.motion.time_step_total - 1]
    assert torch.nonzero(command.motion.motion_ends).flatten().tolist() == expected_ends
    expected_idxs = [0] * (offset + 4) + [1] * (command.motion.time_step_total - offset - 4)
    assert command.motion.motion_idxs.tolist() == expected_idxs

    command.reset(None)
    _walk(command, env, 40)


@pytest.mark.parametrize(("prepend", "append"), [(True, False), (False, True), (True, True)])
def test_motion_dir_transitions_extend_clip_markers(
    tmp_path: Path, captured_default_roots: list[tuple[torch.Tensor, torch.Tensor]], prepend: bool, append: bool
) -> None:
    motion_dir = tmp_path / "clips"
    motion_dir.mkdir()
    _write_motion(motion_dir / "a.npz", 6)
    _write_motion(motion_dir / "b.npz", 8)
    env = _make_env(num_envs=64)
    torch.manual_seed(0)
    command = _make_command(
        env,
        motion_dir=str(motion_dir),
        enable_default_pose_prepend=prepend,
        enable_default_pose_append=append,
        default_pose_prepend_duration_s=0.1,
        default_pose_append_duration_s=0.1,
    )
    motion = command.motion
    assert motion.num_motions == 2 + int(prepend) + int(append)
    _assert_clip_markers_aligned(command)
    # Each transition is its own motion: one end marker per motion, motion_idxs follow the boundaries.
    assert int(motion.motion_ends.sum()) == motion.num_motions
    for k, (start, end) in enumerate(zip(motion.motion_start_idx.tolist(), motion.motion_end_idx.tolist())):
        assert motion.motion_idxs[start:end].tolist() == [k] * (end - start)

    command.reset(None)
    _walk(command, env, 40)


#########################################################################################################
## Reset only touches the reset envs and keeps motion_ids in sync with time_steps
#########################################################################################################
def _two_clip_command(tmp_path: Path, num_envs: int, **motion_kwargs: Any) -> tuple[MotionCommand, SimpleNamespace]:
    motion_dir = tmp_path / "clips"
    motion_dir.mkdir()
    _write_motion(motion_dir / "a.npz", 6)  # global frames 0-5
    _write_motion(motion_dir / "b.npz", 8)  # global frames 6-13
    env = _make_env(num_envs=num_envs)
    return _make_command(env, motion_dir=str(motion_dir), **motion_kwargs), env


@pytest.mark.parametrize("resample_on_motion_end", [True, False])
def test_reset_leaves_other_envs_untouched(tmp_path: Path, resample_on_motion_end: bool) -> None:
    command, _ = _two_clip_command(tmp_path, num_envs=3, resample_on_motion_end=resample_on_motion_end)
    # Envs 0 and 2 sit on the last frame of their clip (clip 0 and the last clip).
    command.time_steps[:] = torch.tensor([5, 0, 13])
    command.motion_ids[:] = torch.tensor([0, 0, 1])

    command.reset(torch.tensor([1]))

    assert command.time_steps[[0, 2]].tolist() == [5, 13]
    assert command.motion_ids[[0, 2]].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("resample_on_motion_end", "expected_steps", "expected_ids"),
    [
        # Clip-end frames are pre-advanced onto the next clip (or wrapped to the first), and the motion id follows.
        (True, [6, 0, 2, 7], [1, 0, 0, 1]),
        # Without clip-end resampling, the last frame of a clip is clamped back to its second-to-last.
        (False, [4, 12, 2, 7], [0, 1, 0, 1]),
    ],
)
def test_reset_keeps_motion_ids_in_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resample_on_motion_end: bool,
    expected_steps: list[int],
    expected_ids: list[int],
) -> None:
    command, env = _two_clip_command(
        tmp_path,
        num_envs=4,
        resample_on_motion_end=resample_on_motion_end,
        use_adaptive_timesteps_sampler=True,
    )
    sampled = torch.tensor([5, 13, 2, 7])
    monkeypatch.setattr(command.adaptive_timesteps_sampler, "sample_global_time_steps", lambda _n: sampled.clone())

    command.reset(None)

    assert command.time_steps.tolist() == expected_steps
    assert command.motion_ids.tolist() == expected_ids
    starts = command.motion.motion_start_idx[command.motion_ids]
    ends = command.motion.motion_end_idx[command.motion_ids]
    assert bool(((starts <= command.time_steps) & (command.time_steps < ends)).all())
    # The robot is placed on the frame the env will track (joint values encode the clip-local frame).
    local = (command.time_steps - starts).float()
    assert torch.allclose(env.simulator.dof_pos, torch.stack([0.01 * local, 0.02 * local], dim=1))
