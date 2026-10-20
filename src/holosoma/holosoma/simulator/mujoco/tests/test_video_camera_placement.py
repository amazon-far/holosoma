"""The MuJoCo video recorder puts the camera where the shared camera config says (pure, no simulator).

IsaacGym and IsaacSim place the recording camera at ``CameraParameters.position`` directly; MuJoCo drives an
``MjvCamera`` through lookat / distance / azimuth / elevation, whose angles describe the VIEW direction rather
than the camera position. Without the 180 degree azimuth flip the MuJoCo video looked at the robot from the
opposite side of the target (mirrored relative to the other backends).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest

if TYPE_CHECKING:
    import mujoco
else:
    mujoco = pytest.importorskip("mujoco")

from holosoma.config_types.video import FixedCameraConfig
from holosoma.simulator.mujoco.video_recorder import MuJoCoVideoRecorder
from holosoma.simulator.shared.camera_controller import CameraController

pytestmark = pytest.mark.no_sim

POSITION, TARGET = (1.1, -1.1, 0.75), (0.12, 0.05, 0.28)


def _scene_camera_position(camera: mujoco.MjvCamera) -> npt.NDArray[np.float64]:
    """World position of an MjvCamera: midpoint of the stereo pair MuJoCo derives from it."""
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size="0.1"/></worldbody></mujoco>')
    scene = mujoco.MjvScene(model, maxgeom=10)
    mujoco.mjv_updateScene(
        model, mujoco.MjData(model), mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    return np.asarray((np.array(scene.camera[0].pos) + np.array(scene.camera[1].pos)) / 2, dtype=np.float64)


def test_fixed_camera_lands_at_the_configured_position() -> None:
    controller = CameraController(FixedCameraConfig(position=list(POSITION), target=list(TARGET)), simulator=None)  # type: ignore[arg-type]
    recorder: Any = object.__new__(MuJoCoVideoRecorder)
    recorder._get_camera_parameters = lambda robot_pos=None: controller.update(robot_pos)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)

    MuJoCoVideoRecorder._update_camera_position(recorder, camera)

    np.testing.assert_allclose(camera.lookat, TARGET, atol=1e-6)
    np.testing.assert_allclose(_scene_camera_position(camera), POSITION, atol=1e-3)
