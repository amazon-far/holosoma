"""Workflow task presets, selected with ``task:<name>``."""

from __future__ import annotations

from holosoma.config_types.task import EvalTaskConfig, TaskConfig, TrainTaskConfig
from holosoma.utils.config_registry import ConfigRegistry

TASK_REGISTRY = ConfigRegistry(TaskConfig, group="holosoma.config.task")

sim = TASK_REGISTRY.add(
    "sim",
    TaskConfig(_target_="holosoma.tasks.sim.SimTask", env_base="holosoma.envs.simulation_task.SimulationTask"),
)
train = TASK_REGISTRY.add(
    "train",
    TrainTaskConfig(_target_="holosoma.tasks.train.TrainTask", env_base="holosoma.envs.policy_task.PolicyTask"),
)
evaluate = TASK_REGISTRY.add(
    "eval",
    EvalTaskConfig(_target_="holosoma.tasks.eval.EvalTask", env_base="holosoma.envs.policy_task.PolicyTask"),
)
replay = TASK_REGISTRY.add(
    "replay",
    TaskConfig(
        _target_="holosoma.tasks.replay.ReplayTask",
        env_base="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    ),
)
