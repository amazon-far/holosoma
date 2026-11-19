"""Configuration types for workflow tasks: what a run does with its environment."""

from __future__ import annotations

from pydantic.dataclasses import dataclass

from holosoma.config_types.eval_callback import EvalCallbacksConfig


@dataclass(frozen=True)
class TaskConfig:
    """A workflow task: the class that runs it and the environment class it drives."""

    _target_: str
    """Import path of the ``holosoma.tasks.base.Task`` subclass that runs this workflow."""

    env_base: str
    """Import path of the class that ``env_class`` must be or subclass."""


@dataclass(frozen=True)
class EvalTaskConfig(TaskConfig):
    """Evaluation of a saved checkpoint."""

    checkpoint: str | None = None
    """Checkpoint to evaluate: a local path or ``wandb://<entity>/<project>/<run_id>[/<checkpoint>]``."""

    callbacks: EvalCallbacksConfig = EvalCallbacksConfig()
    """Callbacks run during evaluation."""
