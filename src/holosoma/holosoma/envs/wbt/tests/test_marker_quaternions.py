"""Exercise WBT marker drawing against both native quaternion conventions."""

from __future__ import annotations

import pytest

from holosoma.config_types.command import CommandManagerCfg, MotionConfig
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.manager import CommandManager
from holosoma.managers.command.terms.wbt import MotionCommand, MotionLoader
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.state import SimState
from holosoma.simulator.isaacsim.backends.base import IsaacLabBackend
from holosoma.simulator.isaacsim.backends.isaaclab2 import IsaacLab2Backend
from holosoma.simulator.isaacsim.backends.isaaclab3 import IsaacLab3Backend
from holosoma.simulator.shared.scene_types import EnvOriginsScene
from holosoma.simulator.types import ActorNames, ActorStates, EnvIds
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType

pytestmark = pytest.mark.no_sim


class _MarkerSink:
    """Replace only the external marker renderer and retain its inputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def visualize(self, position: torch.Tensor, orientation: torch.Tensor | None = None) -> None:
        self.calls.append((position.clone(), None if orientation is None else orientation.clone()))


class _StateFixture(BaseSimulator):
    """Fixed state provider; the quaternion conversion stays in the real backend."""

    def __init__(self, backend: IsaacLabBackend) -> None:
        self.backend = backend
        self.state = SimState("cpu")
        self.scene = EnvOriginsScene(torch.zeros(1, 3), "cpu")
        self._rigid_body_pos = torch.tensor([[[11.0, 12.0, 13.0]]])
        self._rigid_body_rot = torch.tensor([[[0.6, 0.0, 0.0, 0.8]]])

    def get_simulator_type(self) -> SimulatorType:
        return SimulatorType.ISAACSIM

    def get_actor_states(self, names: ActorNames, env_ids: EnvIds | None) -> ActorStates:
        assert names == ["box"]
        assert env_ids is None
        return torch.tensor([[21.0, 22.0, 23.0, 0.0, 0.0, 0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


@pytest.mark.parametrize("family", [2, 3])
@pytest.mark.parametrize("has_object", [False, True])
def test_wbt_markers_receive_the_native_quaternion_convention(family: int, has_object: bool) -> None:
    env = WholeBodyTrackingManager.__new__(WholeBodyTrackingManager)
    env.simulator = _StateFixture(IsaacLab2Backend() if family == 2 else IsaacLab3Backend())
    env.command_manager = CommandManager(CommandManagerCfg(), env, "cpu")

    # Real motion loader and command properties read these fixed XYZW states.
    loader = MotionLoader.__new__(MotionLoader)
    loader._body_indexes = torch.tensor([0, 1])
    loader._body_pos_w = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    loader._body_quat_w = torch.tensor([[[0.0, 0.6, 0.0, 0.8], [0.0, 0.0, 0.0, 1.0]]])
    loader._object_pos_w = torch.tensor([[31.0, 32.0, 33.0]])
    loader._object_quat_w = torch.tensor([[0.2, 0.4, 0.4, 0.8]])
    loader.has_object = has_object
    command = MotionCommand.__new__(MotionCommand)
    command._env = env
    command.motion = loader
    command.motion_cfg = MotionConfig(motion_file="", body_name_ref=["torso"], body_names_to_track=["torso", "hand"])
    command.ref_body_index = 0
    command.tracked_body_indexes = [0, 1]
    command.time_steps = torch.tensor([0])
    command.object_name = "box"
    sinks = {
        name: _MarkerSink()
        for name in ("real_robot", "motion_robot", "real_object", "motion_object", "motion_torso", "motion_hand")
    }
    command.visualization_markers = sinks
    env.command_manager._state_terms["motion_command"] = command

    env._draw_debug_vis()

    # Expected values are literal native orientations, independent of the helper.
    expected_orientations = {
        2: {
            "real_robot": [0.8, 0.6, 0.0, 0.0],
            "motion_robot": [0.8, 0.0, 0.6, 0.0],
            "real_object": [0.8, 0.0, 0.0, 0.6],
            "motion_object": [0.8, 0.2, 0.4, 0.4],
        },
        3: {
            "real_robot": [0.6, 0.0, 0.0, 0.8],
            "motion_robot": [0.0, 0.6, 0.0, 0.8],
            "real_object": [0.0, 0.0, 0.6, 0.8],
            "motion_object": [0.2, 0.4, 0.4, 0.8],
        },
    }
    expected_positions = {
        "real_robot": [11.0, 12.0, 13.0],
        "motion_robot": [1.0, 2.0, 3.0],
        "real_object": [21.0, 22.0, 23.0],
        "motion_object": [31.0, 32.0, 33.0],
        "motion_torso": [1.0, 2.0, 3.0],
        "motion_hand": [4.0, 5.0, 6.0],
    }
    for name, sink in sinks.items():
        if not has_object and name in ("real_object", "motion_object"):
            assert sink.calls == []
            continue
        assert len(sink.calls) == 1
        position, orientation = sink.calls[0]
        torch.testing.assert_close(position, torch.tensor([expected_positions[name]]), rtol=0, atol=0)
        if name in ("motion_torso", "motion_hand"):
            assert orientation is None
        else:
            torch.testing.assert_close(orientation, torch.tensor([expected_orientations[family][name]]), rtol=0, atol=0)
