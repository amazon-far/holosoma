"""Resets place robots and objects at the environment origins a task was created with (MJWarp)."""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("WarpBackend multi-env tests require a CUDA device", allow_module_level=True)

from holosoma.config_types.run_sim import RunSimConfig  # noqa: E402
from holosoma.config_types.scene import RigidObjectConfig, SceneConfig  # noqa: E402
from holosoma.config_values import simulator  # noqa: E402
from holosoma.utils.simulator_config import set_simulator_type  # noqa: E402

BOX = [0.4, 0.3, 0.6]
ORIGINS = [[3.0, 4.0, 0.0], [-5.0, 7.0, 0.0], [8.0, -2.0, 0.0]]


def test_resets_use_the_passed_environment_origins() -> None:
    pytest.importorskip("mujoco_warp")
    preset = simulator.mjwarp
    init = replace(
        preset.config,
        bridge=replace(preset.config.bridge, enabled=False),
        virtual_gantry=replace(preset.config.virtual_gantry, enabled=False),
    )
    scene = SceneConfig(
        rigid_objects={
            "box": RigidObjectConfig(urdf_file="holosoma/data/scene_objects/boxes/small_box.urdf", position=BOX)
        }
    )
    run = RunSimConfig(simulator=replace(preset, config=init), device="cuda:0", time_scale=0, scene=scene)
    run = replace(run, training=replace(run.training, num_envs=len(ORIGINS), headless=True))
    set_simulator_type(run.simulator)
    origins = torch.tensor(ORIGINS, device="cuda:0")

    with run.create_task(env_origins=origins) as task:
        sim = task.simulator
        all_envs = torch.arange(task.num_envs, device=task.device)
        for env_ids in (torch.tensor([2, 0], device=task.device), None):
            task.reset_envs_idx(env_ids)
            ids = all_envs if env_ids is None else env_ids
            robot_xy = sim.robot_root_states[ids, :2]
            box_xy = sim.get_actor_states(["box"], ids)[:, :2]
            init_xy = task.base_init_state[:2]
            assert torch.allclose(robot_xy, origins[ids, :2] + init_xy)
            assert torch.allclose(box_xy - robot_xy, torch.tensor(BOX[:2], device=task.device) - init_xy, atol=1e-5)
