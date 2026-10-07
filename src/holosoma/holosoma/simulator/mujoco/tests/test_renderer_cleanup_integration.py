"""Native MuJoCo renderer lifecycle coverage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

mujoco = pytest.importorskip("mujoco")

from holosoma.simulator.mujoco.backends.classic_backend import ClassicBackend  # noqa: E402
from holosoma.simulator.shared.sensor_manager import CameraRecord  # noqa: E402

pytestmark = pytest.mark.mujoco_classic


def test_classic_backend_allocates_and_closes_native_renderer() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom type="plane" size="1 1 0.1"/>
            <camera name="head" pos="0 0 2"/>
          </worldbody>
        </mujoco>
        """
    )
    backend = object.__new__(ClassicBackend)
    backend.model = model
    backend._closed = False
    backend._cam_ids = {}
    backend._mj_renderers = {}
    camera = SimpleNamespace(
        name="head",
        config=SimpleNamespace(data_types=("rgb",), height=1080, width=1920),
    )

    backend.create_renderers([cast("CameraRecord", camera)])
    renderer = backend._mj_renderers["head"]["rgb"]
    assert model.vis.global_.offwidth == 1920
    assert model.vis.global_.offheight == 1080

    backend.close()

    with pytest.raises(RuntimeError):
        renderer.render()
    assert backend._mj_renderers == {}
    assert backend._cam_ids == {}
    assert backend._closed
