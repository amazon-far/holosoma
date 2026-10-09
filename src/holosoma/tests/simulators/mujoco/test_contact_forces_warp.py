"""Live regression test for MuJoCo Warp contact-force refresh."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("WarpBackend contact-force test requires CUDA", allow_module_level=True)

from holosoma.config_types.scene import RigidObjectConfig, SceneConfig  # noqa: E402
from tests.simulators.mujoco._build import build_warp_sim, object_body_id  # noqa: E402

SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"


def test_contact_force_updates_after_a_box_lands():
    scene = SceneConfig(
        rigid_objects={
            "box": RigidObjectConfig(
                urdf_file=SMALL_BOX,
                position=[0.0, 0.0, 0.2],
            )
        }
    )
    sim = build_warp_sim(scene, num_envs=2)
    box_id = object_body_id(sim, "box")
    env_ids = torch.arange(2, dtype=torch.long, device=sim.sim_device)

    airborne_peak = [0.0, 0.0]
    lowest_z = [float("inf"), float("inf")]
    for _ in range(300):
        sim.backend.step()
        forces = torch.linalg.vector_norm(sim.backend.compute_contact_forces()[:, box_id], dim=-1).tolist()
        heights = sim.get_actor_states(["box"], env_ids)[:, 2].tolist()
        for world_id, (force, z) in enumerate(zip(forces, heights, strict=True)):
            lowest_z[world_id] = min(lowest_z[world_id], z)
            if z > 0.12:
                airborne_peak[world_id] = max(airborne_peak[world_id], force)

    settled_forces = sim.backend.compute_contact_forces()[:, box_id]
    assert max(airborne_peak) < 0.01, f"airborne boxes reported contact force: {airborne_peak} N"
    assert max(lowest_z) < 0.08, f"a box never reached the floor (lowest z={lowest_z} m)"
    expected = torch.tensor(
        [0.0, 0.0, sim.root_model.body_mass[box_id] * abs(sim.root_model.opt.gravity[2])],
        device=sim.sim_device,
        dtype=settled_forces.dtype,
    )
    torch.testing.assert_close(settled_forces, expected.expand_as(settled_forces), rtol=1e-4, atol=1e-5)


def test_applied_force_on_airborne_box_is_not_reported_as_contact():
    scene = SceneConfig(
        rigid_objects={
            "box": RigidObjectConfig(
                urdf_file=SMALL_BOX,
                position=[3.0, 0.0, 1.0],
            )
        }
    )
    sim = build_warp_sim(scene, num_envs=1)
    box_id = object_body_id(sim, "box")
    env_ids = torch.zeros(1, dtype=torch.long, device=sim.sim_device)
    sim.backend.get_applied_forces_view()[0, box_id, 0] = 10.0

    for _ in range(10):
        sim.backend.step()

    assert float(sim.get_actor_states(["box"], env_ids)[0, 7]) > 0.0, "applied force did not accelerate the box"
    force = float(torch.linalg.vector_norm(sim.backend.compute_contact_forces()[0, box_id]))
    assert force < 0.01, f"applied wrench leaked into contact force ({force:.3f} N)"
