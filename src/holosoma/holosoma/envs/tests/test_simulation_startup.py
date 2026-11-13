"""Native startup ordering without simulator SDKs or a custom session adapter."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from holosoma.config_types.env import EnvConfig, get_run_sim_env_config
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.envs.base_task import base_task
from holosoma.envs.render_task import RenderTask
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.simulator.shared.scene_types import EnvOriginsScene

pytestmark = pytest.mark.no_sim
TaskFixture = tuple[type[base_task.BaseTask], type[RenderTask], list[Any]]


@pytest.fixture
def task_type(monkeypatch: pytest.MonkeyPatch) -> TaskFixture:
    calls: list[Any] = []

    class Simulator(BaseSimulator):
        def __init__(self, tyro_config: Any, terrain_manager: Any, device: str) -> None:
            super().__init__(tyro_config, terrain_manager, device)
            self.scene = EnvOriginsScene(torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]), device)
            self.state.register("test", lambda: torch.ones(2, 3))
            camera = SimpleNamespace(effective_decimation=1)
            self.sensor_manager = cast("Any", SimpleNamespace(cameras=[camera], lidars=[]))

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

        def native_env_origins(self) -> torch.Tensor:
            return self.scene.env_origins

        def prepare_render_only(self) -> None:
            calls.append("render-only")

        def render_applied_state(self, dt: float) -> None:
            calls.append(("render", dt, self.time()))

        def render_sensors(self) -> None:
            calls.append("sensors")

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

    class Render(Task, RenderTask):
        pass

    monkeypatch.setattr(base_task, "TerrainManager", lambda *_: object())
    monkeypatch.setattr(base_task, "get_class", lambda _name: Simulator)
    monkeypatch.setattr(base_task, "ResetEventManager", lambda *_: object())
    return Task, Render, calls


def config() -> EnvConfig:
    run = RunSimConfig(device="cpu", time_scale=0)
    return get_run_sim_env_config(replace(run, training=replace(run.training, num_envs=2, headless=True)))


def test_allocator_is_installed_before_the_first_native_read(
    task_type: TaskFixture,
) -> None:
    task, _render, calls = task_type
    allocations: list[Any] = []

    def allocator(name: str, value: torch.Tensor, device: Any) -> torch.Tensor:
        allocations.append((name, device))
        return torch.empty_like(value)

    origins = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    constructed = task(config(), buffer_allocator=allocator, env_origins=origins)
    assert allocations == [("test", None)]
    assert constructed.simulator.state.get("test").data_ptr() == calls[-1][1].data_ptr()
    assert torch.equal(constructed.simulator.env_origins, origins)


def test_render_task_uses_native_scene_placement_and_rejects_physics_step(
    task_type: TaskFixture,
) -> None:
    _task, render, calls = task_type
    constructed = render(config())
    assert torch.equal(constructed.simulator.env_origins, constructed.simulator.scene.env_origins)
    assert calls[-1] == "render-only"
    assert constructed.simulator.kinematic_playback
    with pytest.raises(RuntimeError, match="no physics step"):
        constructed.step()


def test_render_frames_run_at_the_external_clock(task_type: TaskFixture) -> None:
    _task, render, calls = task_type
    constructed = render(config())
    constructed.hooks.add(Phase.FRAME_BEGIN, lambda: calls.append("begin"))
    del calls[:]
    constructed.render_frame(sim_time=0.035)
    constructed.render_frame(sim_time=0.07)
    assert [call if isinstance(call, str) else call[0] for call in calls] == ["begin", "render", "sensors"] * 2
    renders = [call for call in calls if not isinstance(call, str)]
    assert renders[0] == ("render", 0.035, 0.035)
    assert renders[1][1] == pytest.approx(0.035) and renders[1][2] == 0.07
    for sim_time in (0.07, -1.0, float("nan")):
        with pytest.raises(ValueError, match="Source simulation time"):
            constructed.render_frame(sim_time=sim_time)
    constructed.render_frame(sim_time=0.07, repeat=True)
    assert calls[-2] == ("render", 0.0, 0.07)
    with pytest.raises(ValueError, match="Source simulation time"):
        constructed.render_frame(sim_time=0.035, repeat=True)
    assert constructed.frames_rendered == 2


def test_wrong_origin_shape_is_rejected_before_prepare_sim(
    task_type: TaskFixture,
) -> None:
    task, _render, calls = task_type
    with pytest.raises(ValueError, match="Environment origins"):
        task(config(), env_origins=torch.zeros(1, 3))
    assert calls == ["setup"]
