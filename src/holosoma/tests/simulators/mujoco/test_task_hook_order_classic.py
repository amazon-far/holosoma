"""The resolved hook order of a policy task's phases, and ordering against its steps (MuJoCo Classic)."""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.mujoco_classic

from holosoma.config_values import logger as logger_values  # noqa: E402
from holosoma.config_values import simulator as simulator_values  # noqa: E402
from holosoma.config_values.loco.g1.experiment import g1_29dof  # noqa: E402
from holosoma.envs.base_task.task_hooks import TASK_SLOT, TaskHook  # noqa: E402
from holosoma.hooks import HookRef, Phase  # noqa: E402
from holosoma.managers.curriculum import CurriculumManager  # noqa: E402

FRAME_END = (
    "sensors.render",
    "task.count_frame",
    "task.termination",
    "task.observation_frame",
    "task.pre_compute_observations",
    "task.reward",
    "task.update_log_dict",
    "task.terminal_observations",
    "task.reset",
    "task.command_step",
    "task.randomization_step",
    "task.metrics",
    "task.observations",
)
RESET = (
    "task.prepare_reset",
    "task.write_state",
    "task.count_episode",
    "task.reward_reset",
    "task.randomization_reset",
    "task.termination_reset",
    "task.action_reset",
    "task.command_reset",
    "task.managers_reset",
    "curriculum.reset",
    "task.refresh",
    "task.after_reset_sync",
)


def test_policy_task_phases_resolve_in_the_lifecycle_order() -> None:
    config = dataclasses.replace(
        g1_29dof,
        simulator=simulator_values.mujoco,
        training=dataclasses.replace(g1_29dof.training, num_envs=1, headless=True),
        logger=logger_values.disabled,
    )
    with config.create_task() as task:
        # Attached the way the training loop attaches it.
        assert config.curriculum is not None
        CurriculumManager(config.curriculum, task, task.device).install()

        assert task.hooks.names(Phase.PRE_STEP)[0] == "task.apply_actions"
        assert task.hooks.names(Phase.FRAME_END) == FRAME_END
        assert task.hooks.names(Phase.RESET) == RESET

        calls: list[None] = []
        task.hooks.add(
            Phase.FRAME_END,
            lambda: calls.append(None),
            name="test.after_reward",
            after=HookRef(TASK_SLOT, TaskHook.REWARD),
            before=HookRef(TASK_SLOT, TaskHook.UPDATE_LOG_DICT),
        )
        names = task.hooks.names(Phase.FRAME_END)
        assert names[names.index("task.reward") + 1] == "test.after_reward"

        task.reset_envs_idx()
        task.compute_reset_observations()
        task.step({"actions": task.action_manager.action * 0})
        assert len(calls) == 1
