"""Live MuJoCo Warp check: syncing after a mid-frame state write updates body poses."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("WarpBackend reset-sync test requires CUDA", allow_module_level=True)

from tests.simulators.mujoco._build import build_warp_sim  # noqa: E402

SHIFT_X = 1.0


def test_synchronize_after_reset_moves_bodies():
    sim = build_warp_sim(num_envs=2)
    for _ in range(20):
        sim.simulate_at_each_physics_step()
    sim.refresh_sim_tensors()  # the frame-end refresh
    # Body poses after a step come from the step's kinematics, before integration; recompute
    # them from the current state so the comparison isolates the write.
    sim.synchronize_after_reset()

    bodies = sim._rigid_body_pos.clone()
    root_states = sim.robot_root_states[:].clone()
    root_states[0, 0] += SHIFT_X
    dof_states = torch.stack((sim.dof_pos.clone(), sim.dof_vel.clone()), dim=-1)

    # Warp state writes defer kinematics to the next step, so a plain refresh would read stale poses.
    env_ids = torch.tensor([0], device=sim.sim_device)
    sim.set_robot_state(env_ids, root_states=root_states, dof_states=dof_states)
    sim.synchronize_after_reset()

    shift = sim._rigid_body_pos - bodies
    expected = torch.zeros_like(shift)
    expected[0, :, 0] = SHIFT_X
    torch.testing.assert_close(shift, expected, atol=1e-4, rtol=0)
