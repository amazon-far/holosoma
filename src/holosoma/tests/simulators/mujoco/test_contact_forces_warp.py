"""GPU (WarpBackend, multi-env) tests for per-body contact forces.

The ground pushes back on a settled object with exactly its weight, straight up. Reading
cfrc_ext's torque half instead of its force half gives ~0 here, and leaving xfrc_applied in
gives the weight when the object is partly held up.

The robot is unactuated and collapses, so the object sits well outside its reach. A limb
landing against it would change the expected force rather than catch a bug.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo WarpBackend (CUDA) only.
pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("WarpBackend multi-env tests require a CUDA device", allow_module_level=True)

from holosoma.config_types.scene import RigidObjectConfig, SceneConfig  # noqa: E402
from tests.simulators.mujoco._build import build_warp_sim, object_body_id  # noqa: E402

SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"
NUM_ENVS = 4
SETTLE_STEPS = 200


def _settled_box_sim():
    return build_warp_sim(
        SceneConfig(rigid_objects={"box": RigidObjectConfig(urdf_file=SMALL_BOX, position=[3.0, 0.0, 0.2])}),
        num_envs=NUM_ENVS,
    )


def _weight(sim, bid: int) -> float:
    return float(sim.backend.model.body_mass[bid]) * abs(float(sim.backend.model.opt.gravity[2]))


def test_settled_object_contact_force_equals_its_weight():
    sim = _settled_box_sim()
    # g1 has an accelerometer, so mjwarp writes cfrc_ext without being asked: this test would
    # still pass with the backend's sensor_rne_postconstraint line deleted. Assert it directly.
    assert sim.backend.mjw_model.sensor_rne_postconstraint

    for _ in range(SETTLE_STEPS):
        sim.backend.step()

    bid = object_body_id(sim, "box")
    weight = _weight(sim, bid)
    forces = sim.backend.compute_contact_forces()[:, bid].cpu()

    expected = torch.tensor([0.0, 0.0, weight]).expand(NUM_ENVS, 3)
    torch.testing.assert_close(forces, expected, atol=0.02 * weight, rtol=0)


def test_applied_force_is_not_reported_as_contact():
    """Holding up half the object's weight halves the ground reaction, it does not leave it at mg."""
    sim = _settled_box_sim()
    bid = object_body_id(sim, "box")
    weight = _weight(sim, bid)

    sim.backend.xfrc_applied_t[:, bid, 2] = 0.5 * weight
    for _ in range(SETTLE_STEPS):
        sim.backend.step()

    forces = sim.backend.compute_contact_forces()[:, bid].cpu()
    expected = torch.tensor([0.0, 0.0, 0.5 * weight]).expand(NUM_ENVS, 3)
    torch.testing.assert_close(forces, expected, atol=0.02 * weight, rtol=0)
