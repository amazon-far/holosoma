"""Live CPU regressions for Classic MuJoCo contact-force extraction."""

from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from holosoma.simulator.mujoco.backends.classic_backend import ClassicBackend  # noqa: E402


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
