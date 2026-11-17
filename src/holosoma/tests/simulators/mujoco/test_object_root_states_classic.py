"""ClassicBackend object roots written through SimState reach MuJoCo as the actor API reads them."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")

import torch

from holosoma.config_types.scene import RigidObjectConfig, SceneConfig

pytestmark = pytest.mark.mujoco_classic

_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"


def test_object_root_states_writes_match_the_actor_api() -> None:
    from tests.simulators.mujoco._build import build_classic_sim

    sim = build_classic_sim(
        SceneConfig(
            rigid_objects={
                "free": RigidObjectConfig(urdf_file=_BOX, position=[0.4, 0.0, 0.8]),
                "pillar": RigidObjectConfig(urdf_file=_BOX, position=[0.8, 0.0, 0.3], fixed=True),
                "post": RigidObjectConfig(urdf_file=_BOX, position=[0.0, 0.8, 0.3], fixed=True),
            }
        )
    )
    names = list(sim.object_state_names)
    free, pillar, post = names.index("free"), names.index("pillar"), names.index("post")
    env_ids = torch.arange(sim.num_envs, device=sim.sim_device)
    target = sim.object_root_states.clone()
    rotation = torch.tensor([0.3, -0.2, 0.5, 0.78])
    target[:, free, :3] += torch.tensor([0.1, 0.2, 0.3])
    target[:, free, 3:7] = rotation / rotation.norm()
    target[:, free, 7:13] = torch.tensor([0.1, 0.2, 0.3, 1.0, -2.0, 0.5])  # world-frame angular velocity
    target[:, pillar, :3] += torch.tensor([0.5, 0.0, 0.0])
    target[:, post, :3] += torch.tensor([0.0, -0.3, 0.1])

    sim.object_root_states.copy_(target)
    sim.write_state_updates()

    torch.testing.assert_close(sim.get_actor_states(["free"], env_ids), target[:, free], atol=1e-5, rtol=1e-5)
    for name, column in (("pillar", pillar), ("post", post)):
        static_state = sim.get_actor_states([name], env_ids)
        torch.testing.assert_close(static_state[:, :7], target[:, column, :7], atol=1e-5, rtol=1e-5)
        assert not static_state[:, 7:].any()
    torch.testing.assert_close(sim.object_root_states, target, atol=1e-5, rtol=1e-5)
