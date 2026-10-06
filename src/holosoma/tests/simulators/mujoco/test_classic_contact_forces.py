"""Live CPU (ClassicBackend) test: the frame and sign of per-body contact forces.

A box settled under gravity has a closed-form answer, mg up, and on level ground the contact
normal lands in the x slot, so an unrotated accumulation cannot pass.
"""

from __future__ import annotations

import types

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("torch")

from holosoma.simulator.mujoco.backends import ClassicBackend  # noqa: E402

# MuJoCo ClassicBackend (CPU) only.
pytestmark = pytest.mark.mujoco_classic

MASS = 2.0
GRAVITY = 9.81
SETTLE_STEPS = 2000
BOX_CORNERS = 4  # all four down => the box lies flat on the plane, so the net is pure normal

SETTLED_BOX = """
<mujoco>
  <option gravity="0 0 -{gravity}"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 .1" euler="0 {tilt} 0"/>
    <body name="box" pos="0 0 0.4">
      <freejoint/>
      <geom name="box" type="box" size="0.1 0.1 0.1" mass="{mass}" friction="5 5 5"/>
    </body>
  </worldbody>
</mujoco>
"""


def _settled_backend(tilt: float = 0.0):
    """A box dropped onto a ``tilt``-degree plane and stepped to rest, with its backend."""
    model = mujoco.MjModel.from_xml_string(SETTLED_BOX.format(gravity=GRAVITY, tilt=tilt, mass=MASS))
    data = mujoco.MjData(model)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
    config = types.SimpleNamespace(training=types.SimpleNamespace(num_envs=1))
    # compute_contact_forces reaches only model/data/device, so the stub config satisfies it.
    return model, data, ClassicBackend(model, data, config, "cpu")  # type: ignore[arg-type]


def test_resting_box_pushed_straight_up():
    model, data, backend = _settled_backend()
    forces = backend.compute_contact_forces()

    assert data.ncon == BOX_CORNERS
    assert forces[0, model.body("box").id].tolist() == pytest.approx([0.0, 0.0, MASS * GRAVITY], abs=1e-2)


def test_ground_reaction_opposes_the_box():
    model, data, backend = _settled_backend()
    forces = backend.compute_contact_forces()

    assert data.ncon == BOX_CORNERS
    box = forces[0, model.body("box").id]
    assert forces[0, model.body("world").id].tolist() == pytest.approx((-box).tolist(), abs=1e-4)


def test_tilted_ground_still_balances_gravity():
    """A 20-degree normal rules out a fixed axis permutation."""
    model, data, backend = _settled_backend(tilt=20.0)
    forces = backend.compute_contact_forces()

    # Past ~21 degrees the box tumbles downslope instead of settling, and the force assert
    # alone would fail without saying that it never came to rest.
    assert data.ncon == BOX_CORNERS
    assert forces[0, model.body("box").id].tolist() == pytest.approx([0.0, 0.0, MASS * GRAVITY], abs=1e-2)
