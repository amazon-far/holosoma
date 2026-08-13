"""Live CPU (ClassicBackend) test: kinematically relocating a STATIC scene body at runtime.

Static (welded, jointless) MuJoCo bodies have no qpos slice; they move via the model
``body_pos``/``body_quat`` + forward-kinematics path (``set_static_body_world_pose``), routed
through the unified ``set_actor_states`` / ``set_static_body_pose`` API. This proves a static body
BOTH (a) moves (``get_actor_states``
reports the new full pose) AND (b) is collided-with at the new pose — asserted directly on the
MuJoCo contact list (a freebox<->pillar contact appears only when the pillar was relocated under
the box), with a control proving no such contact when the pillar stays parked away.

Contact is used rather than settling height because the small box falls slowly under gravity at
this dt, so a "rests at height X" check would need many steps and a thin margin; a geom-pair
contact is immediate and unambiguous.

Runs in the MuJoCo (hsmujoco) CPU env — no CUDA. The Warp GPU analogue is the cluster test.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo ClassicBackend (CPU) only.
pytestmark = pytest.mark.mujoco_classic

from holosoma.config_types.scene import RigidObjectConfig, SceneConfig  # noqa: E402
from holosoma.managers.randomization.terms import objects as O  # noqa: E402
from holosoma.simulator.shared.object_registry import ObjectType  # noqa: E402
from tests.simulators.mujoco._build import build_classic_sim  # noqa: E402

SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"


def _scene():
    # Free box dropped from just above a static pillar's relocation target; pillar parked far away
    # (x=5) so it does not touch initially. Both away from the robot at the origin.
    return SceneConfig(
        rigid_objects={
            "freebox": RigidObjectConfig(urdf_file=SMALL_BOX, position=[2.0, 0.0, 0.35]),
            "pillar": RigidObjectConfig(urdf_file=SMALL_BOX, position=[5.0, 0.0, 0.1], fixed=True),
        }
    )


def _geom_ids(sim, name):
    body_ids = {b for b, _ in O._mujoco_object_bodies(sim, name)}
    m = sim.backend.model
    return {g for g in range(m.ngeom) if m.geom_bodyid[g] in body_ids}


def _has_freebox_pillar_contact(sim, steps=120):
    """Step and return True as soon as a freebox-geom <-> pillar-geom contact appears."""
    fg, pg = _geom_ids(sim, "freebox"), _geom_ids(sim, "pillar")
    for _ in range(steps):
        sim.backend.step()
        d = sim.backend.data
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in fg and c.geom2 in pg) or (c.geom1 in pg and c.geom2 in fg):
                return True
    return False


def test_static_body_relocates_and_is_collided_with():
    sim = build_classic_sim(_scene())
    eid = torch.tensor([0])
    assert sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL) == ["freebox"]
    assert sim.object_registry.get_names_by_type(ObjectType.SCENE) == ["pillar"]

    before = sim.get_actor_states(["pillar"], eid)[0, :3].tolist()
    assert abs(before[0] - 5.0) < 1e-3, "pillar should start parked at x=5"

    # Relocate the static pillar UNDER the free box (x=2, z=0.25) with a 90deg yaw — full pose.
    c, sn = math.cos(math.pi / 4), math.sin(math.pi / 4)
    sim.set_static_body_pose(["pillar"], eid, torch.tensor([[2.0, 0.0, 0.25, 0.0, 0.0, sn, c]]))

    after = sim.get_actor_states(["pillar"], eid)[0]
    # (a) MOVED: get_actor_states reports the new position AND orientation (xyzw).
    assert torch.allclose(after[:3], torch.tensor([2.0, 0.0, 0.25]), atol=1e-3), f"pillar pos not moved: {after[:3]}"
    assert abs(float(after[5]) - sn) < 1e-2 and abs(float(after[6]) - c) < 1e-2, f"pillar yaw not applied: {after[3:7]}"

    # (b) COLLIDED-WITH: the dropping free box makes contact with the relocated pillar.
    assert _has_freebox_pillar_contact(sim), "no freebox<->pillar contact after relocating the static under the box"


def test_unmoved_static_control_no_contact():
    """Control: with the static left parked away, the dropping box never contacts it."""
    sim = build_classic_sim(_scene())
    assert not _has_freebox_pillar_contact(sim), "freebox<->pillar contact with the static parked away — false positive"
