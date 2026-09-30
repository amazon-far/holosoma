"""Live regression tests for the MuJoCo backend simulation clocks."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from tests.simulators.mujoco._build import build_classic_sim, build_warp_sim  # noqa: E402


@pytest.mark.mujoco_classic
def test_classic_clock_tracks_native_mujoco_time():
    sim = build_classic_sim()
    timestep = float(sim.root_model.opt.timestep)
    assert sim.time() == pytest.approx(0.0)

    sim.simulate_at_each_physics_step()
    assert sim.time() == pytest.approx(timestep)
    sim.simulate_at_each_physics_step()

    assert sim.time() == pytest.approx(2 * timestep)
    assert sim.time() == pytest.approx(float(sim.root_data.time))


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="WarpBackend clock test requires CUDA")
def test_headless_warp_clock_advances_without_render_sync():
    sim = build_warp_sim(num_envs=1)
    timestep = float(sim.root_model.opt.timestep)
    assert sim.time() == pytest.approx(0.0)
    assert sim.root_data.time == pytest.approx(0.0)

    sim.simulate_at_each_physics_step()
    assert sim.time() == pytest.approx(timestep)
    sim.simulate_at_each_physics_step()

    assert sim.time() == pytest.approx(2 * timestep)
    assert sim.root_data.time == pytest.approx(0.0)

    sim.backend.initialize_state(sim.root_model, sim.root_data)
    assert sim.time() == pytest.approx(0.0)

    sim.simulate_at_each_physics_step()
    assert sim.time() == pytest.approx(timestep)
    assert sim.backend.get_render_data().time == pytest.approx(timestep)
