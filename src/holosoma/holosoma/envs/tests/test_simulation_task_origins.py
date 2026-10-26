"""Environment placement for simulation-only tasks."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.envs.simulation_task import SimulationTask


def _task(num_envs: int, terrain_origins: torch.Tensor) -> SimulationTask:
    task = object.__new__(SimulationTask)
    task.num_envs = num_envs
    task.device = "cpu"
    task.terrain_manager = SimpleNamespace(get_state=lambda _name: SimpleNamespace(env_origins=terrain_origins))  # type: ignore[assignment]
    return task


def test_a_single_environment_sits_at_the_world_origin() -> None:
    origins = _task(1, torch.tensor([[36.0, 4.0, 0.5]]))._get_env_origins()
    assert torch.equal(origins, torch.zeros(1, 3))
