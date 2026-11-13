"""Native startup ordering without simulator SDKs or a custom session adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from holosoma.config_types.env import EnvConfig, get_run_sim_env_config
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.envs.base_task import base_task
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

pytestmark = pytest.mark.no_sim
TaskFixture = tuple[type[base_task.BaseTask], list[Any]]


@pytest.fixture
def task_type(monkeypatch: pytest.MonkeyPatch) -> TaskFixture:
    calls: list[Any] = []

    class Simulator(BaseSimulator):
        def __init__(self, tyro_config: Any, terrain_manager: Any, device: str) -> None:
            super().__init__(tyro_config, terrain_manager, device)
            self.state.register("test", lambda: torch.ones(2, 3))

        def setup(self) -> None:
            calls.append("setup")
            self.sim_dt = 0.005

        def setup_terrain(self) -> None:
            pass

        def create_envs(
            self,
            num_envs: int,
            env_origins: Any,
            base_init_state: Any,
            env_config: dict[str, Any] | None = None,
        ) -> None:
            self.env_origins = env_origins
            calls.append(("origins", env_origins.clone()))

        def get_dof_limits_properties(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return torch.zeros(1), torch.zeros(1), torch.zeros(1)

        def prepare_sim(self) -> None:
            calls.append(("first-read", self.state.get("test")))

        def install_plugins(self) -> None:
            pass

    class Task(base_task.BaseTask):
        def _load_assets(self) -> None:
            self.base_init_state = torch.zeros(13)

        def _setup_robot_body_indices(self) -> None:
            pass

        def _initialize_task_managers(self) -> None:
            pass

        def _init_buffers(self) -> None:
            pass

        def _prepare_manager_fields(self) -> None:
            pass

        def _setup_task_managers(self) -> None:
            pass

    monkeypatch.setattr(base_task, "TerrainManager", lambda *_: object())
    monkeypatch.setattr(base_task, "get_class", lambda _name: Simulator)
    monkeypatch.setattr(base_task, "ResetEventManager", lambda *_: object())
    return Task, calls


def config() -> EnvConfig:
    run = RunSimConfig(device="cpu", time_scale=0)
    return get_run_sim_env_config(replace(run, training=replace(run.training, num_envs=2, headless=True)))


def test_allocator_is_installed_before_the_first_native_read(
    task_type: TaskFixture,
) -> None:
    task, calls = task_type
    allocations: list[Any] = []

    def allocator(name: str, value: torch.Tensor, device: Any) -> torch.Tensor:
        allocations.append((name, device))
        return torch.empty_like(value)

    origins = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    constructed = task(config(), buffer_allocator=allocator, env_origins=origins)
    assert allocations == [("test", None)]
    assert constructed.simulator.state.get("test").data_ptr() == calls[-1][1].data_ptr()
    assert torch.equal(constructed.simulator.env_origins, origins)


def test_wrong_origin_shape_is_rejected_before_prepare_sim(
    task_type: TaskFixture,
) -> None:
    task, calls = task_type
    with pytest.raises(ValueError, match="Environment origins"):
        task(config(), env_origins=torch.zeros(1, 3))
    assert calls == ["setup"]
