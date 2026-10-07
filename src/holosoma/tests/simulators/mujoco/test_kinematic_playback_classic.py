"""Live classic-MuJoCo tests for kinematic playback.

Builds a real single-env CPU sim with the ``motion-playback`` plugin installed and
``sim.kinematic_playback`` set, drives the run_sim phase cycle, and asserts: the robot tracks
the clip exactly (no gravity sag, no dynamics drift), body transforms follow via FK, the
kinematic clock advances while the engine clock stands still, and the plugin requests shutdown
at clip end.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

pytestmark = pytest.mark.mujoco_classic

from holosoma.config_types.plugin import MotionPlaybackPluginConfig  # noqa: E402
from holosoma.simulator.base_simulator.hooks import Phase  # noqa: E402
from tests.simulators.mujoco._build import build_classic_sim  # noqa: E402

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


def _frame(sim):
    """One control frame of the run_sim loop shape."""
    control_decimation = sim.simulator_config.sim.control_decimation_steps
    for substep in range(control_decimation):
        if substep == 0:
            sim.hooks.emit(Phase.FRAME_BEGIN)
        sim.hooks.emit(Phase.PRE_STEP)
        sim.simulate_at_each_physics_step()
        sim.hooks.emit(Phase.POST_STEP)
        if substep == control_decimation - 1:
            # The frame closes on the LAST substep, as run_sim's loop does.
            sim.hooks.emit(Phase.FRAME_END)


@pytest.fixture(scope="module")
def num_dof():
    return build_classic_sim().num_dof


def _build_playback_sim(tmp_path, num_dof, *, ramp: bool = False, seconds: float = 0.1):
    # Clip fps = the sim's control rate, so one clip frame maps to one control frame exactly.
    from holosoma.config_values.run_sim import mujoco as mujoco_preset

    control_hz = mujoco_preset.config.sim.fps / mujoco_preset.config.sim.control_decimation_steps
    frames = int(seconds * control_hz) + 1
    clip = _write_clip(tmp_path, num_dof, frames=frames, ramp=ramp)
    cfg = MotionPlaybackPluginConfig(motion_files=[str(clip)], fps=control_hz)
    sim = build_classic_sim(plugin={"play": cfg}, kinematic_playback=True)
    return sim, frames


def test_kinematic_playback_no_gravity_sag(tmp_path, num_dof):
    sim, frames = _build_playback_sim(tmp_path, num_dof)
    for _ in range(frames - 1):
        _frame(sim)
    assert sim.robot_root_states[0, 2].item() == pytest.approx(ROOT_Z, abs=1e-6)
    assert torch.allclose(sim.dof_pos[:], torch.zeros(1, sim.num_dof), atol=1e-6)


def test_kinematic_playback_body_transforms_track_fk(tmp_path, num_dof):
    sim, _ = _build_playback_sim(tmp_path, num_dof)
    _frame(sim)
    # FK propagated the written root pose into the derived body transforms.
    data = sim.backend.get_render_data()
    pelvis = mujoco.mj_name2id(sim.root_model, mujoco.mjtObj.mjOBJ_BODY, "robot_pelvis")
    assert pelvis >= 0
    assert data.xpos[pelvis][2] == pytest.approx(ROOT_Z, abs=1e-5)


def test_kinematic_playback_joints_follow_clip(tmp_path, num_dof):
    sim, frames = _build_playback_sim(tmp_path, num_dof, ramp=True)
    n = frames // 2
    for _ in range(n):
        _frame(sim)
    # Clip fps == control rate: frame k's write samples clip index k-1 exactly.
    expected = JOINT_RAMP * (n - 1) / (frames - 1)
    assert torch.allclose(sim.dof_pos[:], torch.full((1, sim.num_dof), expected), atol=1e-5)


def test_kinematic_clock_advances_engine_clock_does_not(tmp_path, num_dof):
    sim, _ = _build_playback_sim(tmp_path, num_dof)
    assert sim.time() == 0.0
    for _ in range(5):
        _frame(sim)
    sim_cfg = sim.simulator_config.sim
    expected = 5 * sim_cfg.control_decimation_steps / sim_cfg.fps
    assert sim.time() == pytest.approx(expected)
    assert sim._physics_time() == pytest.approx(0.0)


def test_playback_requests_shutdown_at_clip_end(tmp_path, num_dof):
    sim, frames = _build_playback_sim(tmp_path, num_dof)
    for _ in range(frames + 2):
        _frame(sim)
    assert sim.shutdown_requested
