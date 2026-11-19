"""Defaults for the ``sim`` task: an interactive simulation with no policy."""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import holosoma.config_values.simulator
import holosoma.config_values.task
from holosoma.config_types.experiment import ExperimentConfig, RunConfig
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.task import TaskConfigType
from holosoma.utils.config_registry import UseRegistry


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RunSimConfig(ExperimentConfig):
    """``ExperimentConfig`` with the ``sim`` task's defaults.

    One environment of a ``SimulationTask`` on an interactive simulator preset (bridge and virtual
    gantry on), paced to real time, with no policy managers.

    Usage Examples:
        holosoma simulator:sim-mujoco robot:t1-29dof-waist-wrist
        holosoma simulator:sim-isaacsim robot:g1-29dof terrain:terrain-locomotion-mix
    """

    task: Annotated[TaskConfigType, UseRegistry(holosoma.config_values.task.TASK_REGISTRY)] = (
        holosoma.config_values.task.sim
    )
    env_class: str = "holosoma.envs.simulation_task.SimulationTask"
    simulator: Annotated[SimulatorConfig, UseRegistry(holosoma.config_values.simulator.SIMULATOR_REGISTRY)] = (
        holosoma.config_values.simulator.sim_mujoco
    )
    run: RunConfig = RunConfig(num_envs=1, headless=False, seed=42, time_scale=1.0)
