"""Live CPU regressions for Classic MuJoCo contact-force extraction."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from holosoma.simulator.mujoco.backends.classic_backend import ClassicBackend  # noqa: E402

if TYPE_CHECKING:
    from holosoma.config_types.full_sim import FullSimConfig


def test_tilted_contact_force_is_reported_in_world_coordinates():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <geom name="slope" type="plane" size="2 2 0.1"
                  quat="0.9238795325 0 0.3826834324 0"/>
            <body name="ball" pos="0.0353553391 0 0.0353553391">
              <freejoint/>
              <geom name="ball_geom" type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon > 0

    config = types.SimpleNamespace(training=types.SimpleNamespace(num_envs=1))
    backend = ClassicBackend(model, data, config, "cpu")
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    force = backend.compute_contact_forces()[0, ball_id].numpy()

    magnitude = np.linalg.norm(force)
    assert magnitude > 1.0
    np.testing.assert_allclose(force / magnitude, [2**-0.5, 0.0, 2**-0.5], atol=1e-6)


MASS = 2.0
GRAVITY = 9.81
SETTLE_STEPS = 2000
BOX_CORNERS = 4  # all four down => the box lies flat on the plane, so the net is pure normal

_SETTLED_BOX = """
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
    model = mujoco.MjModel.from_xml_string(_SETTLED_BOX.format(gravity=GRAVITY, tilt=tilt, mass=MASS))
    data = mujoco.MjData(model)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
    # compute_contact_forces reaches only model/data/device, so the stub config satisfies it.
    config = cast("FullSimConfig", types.SimpleNamespace(training=types.SimpleNamespace(num_envs=1)))
    return model, data, ClassicBackend(model, data, config, "cpu")


def test_resting_box_is_pushed_straight_up_by_its_weight():
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
