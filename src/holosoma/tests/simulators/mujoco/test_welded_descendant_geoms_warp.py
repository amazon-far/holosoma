"""Real CUDA placement and graph replay for a fixed robot with welded descendants."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.mujoco_warp
if not torch.cuda.is_available():
    pytest.skip("Welded-descendant runtime regression requires CUDA", allow_module_level=True)
pytest.importorskip("mujoco_warp")
wp = pytest.importorskip("warp")

from holosoma.config_types.experiment import TrainingConfig  # noqa: E402
from holosoma.config_types.full_sim import FullSimConfig  # noqa: E402
from holosoma.config_types.logger import DisabledLoggerConfig  # noqa: E402
from holosoma.config_values.run_sim import mjwarp  # noqa: E402
from holosoma.simulator.mujoco.backends.warp_backend import WarpBackend  # noqa: E402
from holosoma.simulator.mujoco.tests.test_welded_descendant_geoms import (  # noqa: E402
    STATIC_ROBOT_GEOMS,
    WELDED_ROBOT_XML,
    reference_poses,
)
from tests.simulators._robot_presets import test_arm_fixed  # noqa: E402


def test_welded_child_follows_selected_root_through_placement_and_step():
    model = mujoco.MjModel.from_xml_string(WELDED_ROBOT_XML)
    root = model.body("base").id
    num_envs = 3
    config = FullSimConfig(
        simulator=replace(
            mjwarp.config,
            mujoco_warp=replace(mjwarp.config.mujoco_warp, nconmax_per_env=32, njmax_per_env=64),
        ),
        robot=test_arm_fixed,  # Backend construction consumes only simulator/training settings.
        training=TrainingConfig(num_envs=num_envs, headless=True),
        logger=DisabledLoggerConfig(),
    )
    data = mujoco.MjData(model)
    data.qpos[:] = 0.4
    mujoco.mj_forward(model, data)
    backend = WarpBackend(model, data, config, "cuda:0")
    try:
        backend.initialize_state(model, data)
        positions = np.tile(model.body_pos[root], (num_envs, 1))
        quats = np.tile(model.body_quat[root], (num_envs, 1))
        env_ids = torch.tensor([1], device="cuda:0")
        cached_ids = backend._geom_ids_for_bodies([root])
        np.testing.assert_array_equal(cached_ids.numpy(), [model.geom(name).id for name in STATIC_ROBOT_GEOMS])
        body_pos_ptr, body_quat_ptr = backend.mjw_model.body_pos.ptr, backend.mjw_model.body_quat.ptr

        def check_native_poses():
            joints = backend.qpos_t.cpu().numpy()
            expected_pos, _, expected_geom_pos, expected_geom_mat = reference_poses(positions, quats, joints)
            np.testing.assert_allclose(backend.mjw_data.xpos.numpy(), expected_pos, atol=2e-5)
            np.testing.assert_allclose(backend.mjw_data.geom_xpos.numpy(), expected_geom_pos, atol=2e-5)
            np.testing.assert_allclose(
                backend.mjw_data.geom_xmat.numpy().reshape(num_envs, -1, 9), expected_geom_mat, atol=2e-5
            )

        check_native_poses()
        for position, quat in [
            ([2.0, -1.0, 1.5], [2**-0.5, 0.0, 0.0, 2**-0.5]),
            ([-2.0, 1.5, 2.2], [3**0.5 / 2, 0.5, 0.0, 0.0]),
        ]:
            positions[1], quats[1] = position, quat
            backend.set_static_body_world_pose(
                [root],
                torch.tensor([[position]], device="cuda:0"),
                torch.tensor(np.array(quat)[[1, 2, 3, 0]][None, None], device="cuda:0"),
                env_ids=env_ids,
            )
            check_native_poses()  # Includes unselected envs, the articulated arm/hand, and unrelated actors.
            assert backend._geom_ids_for_bodies([root]) is cached_ids
            assert (backend.mjw_model.body_pos.ptr, backend.mjw_model.body_quat.ptr) == (body_pos_ptr, body_quat_ptr)

            before = backend.qpos_t.clone()
            backend.ctrl_t[:] = 0.5
            backend.step()  # Replay the real graph captured before placement.
            backend.forward_kinematics()  # Align derived poses with the integrated qpos.
            assert torch.any(backend.qpos_t != before), "The articulated descendant must remain dynamic."
            check_native_poses()
    finally:
        backend.close()
