from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.observation.base import ObservationTermBase
from holosoma.managers.observation.manager import ObservationManager
from holosoma.simulator.shared.sensor_manager import SensorManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

NUM_ENVS = 5


class _RecordingTerm(ObservationTermBase):
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def on_state_overwritten(self, env_ids: torch.Tensor) -> None:
        self.env.events.append(("obs_overwritten", env_ids.tolist()))

    def read(self, env: Any, env_ids: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        env.events.append(("read", None if env_ids is None else env_ids.tolist()))
        return torch.zeros(NUM_ENVS if env_ids is None else len(env_ids), 1)


def _manager(events: list[tuple[Any, ...]]) -> ObservationManager:
    env = SimpleNamespace(
        num_envs=NUM_ENVS,
        device="cpu",
        events=events,
        simulator=SimpleNamespace(rerender_sensors=lambda env_ids: events.append(("rerender", env_ids.tolist()))),
    )
    term = ObsTermCfg(func="holosoma.envs.base_task.tests.test_state_overwrite:_RecordingTerm")
    cfg = ObservationManagerCfg(groups={"policy": ObsGroupCfg(terms={"term": term})})
    manager = ObservationManager(cfg, env, device="cpu")
    events.clear()  # drop reads made while sizing the term
    return manager


def test_terminal_read_sees_the_state_before_the_overwrite() -> None:
    events: list[tuple[Any, ...]] = []
    manager = _manager(events)

    manager.new_frame()
    manager.step_terminal(torch.tensor([2]))
    manager.reset(torch.tensor([2]))
    manager.notify_state_overwritten(torch.tensor([2]))
    manager.step()

    assert events == [("read", [2]), ("rerender", [2]), ("obs_overwritten", [2]), ("read", None)]


def test_overwrites_are_applied_once_before_the_next_step() -> None:
    events: list[tuple[Any, ...]] = []
    manager = _manager(events)

    manager.notify_state_overwritten(torch.tensor([3, 1]))
    manager.notify_state_overwritten(torch.tensor([3]))
    manager.new_frame()
    manager.step()
    manager.new_frame()
    manager.step()

    assert events == [("rerender", [1, 3]), ("obs_overwritten", [1, 3]), ("read", None), ("read", None)]


def test_no_overwrite_means_no_rerender() -> None:
    events: list[tuple[Any, ...]] = []
    manager = _manager(events)

    # BaseTask._post_physics_step resets an empty set of envs on most steps.
    manager.notify_state_overwritten(torch.tensor([], dtype=torch.long))
    manager.new_frame()
    manager.step()

    assert events == [("read", None)]


def test_reset_notifies_after_the_state_is_written() -> None:
    events: list[tuple[Any, ...]] = []
    task: Any = object.__new__(BaseTask)
    task.num_envs = NUM_ENVS
    task.device = "cpu"
    task.simulator = SimpleNamespace(hooks=SimpleNamespace(emit=lambda *_args: None))
    task.observation_manager = SimpleNamespace(
        reset=lambda env_ids: events.append(("obs_reset", env_ids.tolist())),
        notify_state_overwritten=lambda env_ids: events.append(("notify_overwritten", env_ids.tolist())),
    )
    task._pending_episode_lengths = torch.zeros(NUM_ENVS, dtype=torch.long)
    task._pending_episode_update_mask = torch.zeros(NUM_ENVS, dtype=torch.bool)
    task.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
    task._reset_envs_idx_impl = lambda env_ids, *_args: events.append(("write_state", env_ids.tolist()))
    for name in ("randomization_manager", "action_manager", "command_manager", "curriculum_manager"):
        setattr(task, name, None)
    task.termination_manager = None
    task.reset_manager = SimpleNamespace(reset_scene=lambda env_ids: events.append(("reset_scene", env_ids.tolist())))

    task.reset_envs_idx(torch.tensor([0, 4]))

    assert events == [
        ("obs_reset", [0, 4]),
        ("write_state", [0, 4]),
        ("reset_scene", [0, 4]),
        ("notify_overwritten", [0, 4]),
    ]


def _depth_camera(manager: SensorManager, name: str) -> Any:
    config = CameraSensorConfig(
        mount=SensorMountConfig(target_kind="robot_link", target="pelvis"),
        near=0.1,
        far=3.0,
        depth_clipping_behavior="max",
        data_types=["depth"],
    )
    record = manager.register_camera(name, config)
    record.set_buffer("depth", torch.ones(NUM_ENVS, 2, 2, 1))
    return record


def test_sensor_manager_rerenders_only_sensors_that_support_it() -> None:
    manager = SensorManager(device="cpu", control_hz=50.0)
    partial = _depth_camera(manager, "partial")
    _depth_camera(manager, "full_only")
    calls: list[list[int]] = []
    partial.rerender = lambda env_ids: calls.append(env_ids.tolist())

    manager.rerender(torch.tensor([1, 2]))

    assert calls == [[1, 2]]


def test_camera_buffer_rows_apply_depth_clipping_and_drop_device_copies() -> None:
    manager = SensorManager(device="cpu", control_hz=50.0)
    record = _depth_camera(manager, "cam")
    stale_copy = record.buffer_on("depth", "meta")
    rows = torch.tensor([float("inf"), 1.0, 0.05, 2.0]).view(1, 2, 2, 1)

    record.set_buffer_rows("depth", torch.tensor([3]), rows)

    assert record.buffers["depth"][3].flatten().tolist() == [3.0, 1.0, 3.0, 2.0]
    assert torch.all(record.buffers["depth"][[0, 1, 2, 4]] == 1.0)
    assert record.buffer_on("depth", "meta") is not stale_copy


def test_simulator_rerender_without_sensor_manager_is_a_no_op() -> None:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

    sim: Any = object.__new__(BaseSimulator)
    sim.sensor_manager = None
    sim.rerender_sensors(torch.tensor([0]))

    sim.sensor_manager = MagicMock()
    sim.rerender_sensors(torch.tensor([0]))
    sim.sensor_manager.rerender.assert_called_once()
