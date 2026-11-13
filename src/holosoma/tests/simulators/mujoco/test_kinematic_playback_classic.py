"""Live classic-MuJoCo tests for kinematic playback.

Builds a real single-env CPU simulation task with the ``motion-playback`` plugin installed and
``sim.kinematic_playback`` set, steps it as ``SimulationTask.run`` does, and asserts: the robot tracks
the clip exactly (no gravity sag, no dynamics drift), body transforms follow via FK, the
kinematic clock advances while the engine clock stands still, and the plugin requests shutdown
at clip end.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

pytestmark = pytest.mark.mujoco_classic

from holosoma.config_types.plugin import MotionPlaybackPluginConfig  # noqa: E402
from tests.simulators.mujoco._build import build_classic_sim, classic_run_sim_config  # noqa: E402

ROOT_Z = 1.5  # clip root height: held in the air, so any gravity leak shows immediately
JOINT_RAMP = 0.5


def _write_clip(path, num_dof, *, frames: int, ramp: bool):
    """A matrix clip: root(7) at ROOT_Z + joints, flat or ramping to JOINT_RAMP."""
    mat = np.zeros((frames, 7 + num_dof), dtype=np.float32)
    mat[:, 2] = ROOT_Z
    mat[:, 3] = 1.0  # identity quat (wxyz)
    if ramp:
        mat[:, 7:] = np.linspace(0.0, JOINT_RAMP, frames, dtype=np.float32)[:, None]
    file = path / "clip.npy"
    np.save(file, mat)
    return file


@pytest.fixture(scope="module")
def num_dof():
    return build_classic_sim().num_dof


@contextmanager
def _playback_task(tmp_path, num_dof, *, ramp: bool = False, seconds: float = 0.1):
    """Yield ``(task, frames)`` for a reset playback task; one clip frame per control frame."""
    from holosoma.config_values.run_sim import mujoco as mujoco_preset

    control_hz = mujoco_preset.config.sim.fps / mujoco_preset.config.sim.control_decimation_steps
    frames = int(seconds * control_hz) + 1
    clip = _write_clip(tmp_path, num_dof, frames=frames, ramp=ramp)
    cfg = MotionPlaybackPluginConfig(motion_files=[str(clip)], fps=control_hz)
    with classic_run_sim_config(plugin={"play": cfg}, kinematic_playback=True).create_task() as task:
        task.reset_all()
        yield task, frames


def test_kinematic_playback_no_gravity_sag(tmp_path, num_dof):
    with _playback_task(tmp_path, num_dof) as (task, frames):
        sim = task.simulator
        for _ in range(frames - 1):
            task.step()
        assert sim.robot_root_states[0, 2].item() == pytest.approx(ROOT_Z, abs=1e-6)
        assert torch.allclose(sim.dof_pos[:], torch.zeros(1, sim.num_dof), atol=1e-6)


def test_kinematic_playback_body_transforms_track_fk(tmp_path, num_dof):
    with _playback_task(tmp_path, num_dof) as (task, _):
        sim = task.simulator
        task.step()
        # FK propagated the written root pose into the derived body transforms.
        data = sim.backend.get_render_data()
        pelvis = mujoco.mj_name2id(sim.root_model, mujoco.mjtObj.mjOBJ_BODY, "robot_pelvis")
        assert pelvis >= 0
        assert data.xpos[pelvis][2] == pytest.approx(ROOT_Z, abs=1e-5)


def test_kinematic_playback_joints_follow_clip(tmp_path, num_dof):
    with _playback_task(tmp_path, num_dof, ramp=True) as (task, frames):
        sim = task.simulator
        n = frames // 2
        for _ in range(n):
            task.step()
        # Clip fps == control rate: frame k's write samples clip index k-1 exactly.
        expected = JOINT_RAMP * (n - 1) / (frames - 1)
        assert torch.allclose(sim.dof_pos[:], torch.full((1, sim.num_dof), expected), atol=1e-5)


def test_kinematic_clock_advances_engine_clock_does_not(tmp_path, num_dof):
    with _playback_task(tmp_path, num_dof) as (task, _):
        sim = task.simulator
        assert sim.time() == 0.0
        for _ in range(5):
            task.step()
        sim_cfg = sim.simulator_config.sim
        expected = 5 * sim_cfg.control_decimation_steps / sim_cfg.fps
        assert sim.time() == pytest.approx(expected)
        assert sim._physics_time() == pytest.approx(0.0)


def test_playback_requests_task_shutdown_at_clip_end(tmp_path, num_dof):
    with _playback_task(tmp_path, num_dof) as (task, frames):
        for _ in range(frames + 2):
            task.step()
        assert task.shutdown_requested
