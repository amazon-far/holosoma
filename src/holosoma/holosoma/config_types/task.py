"""Configuration types for workflow tasks: what a run does with its environment."""

from __future__ import annotations

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from holosoma.config_types.eval_callback import EvalCallbacksConfig

# Unknown fields are rejected so a serialized task validates back to exactly its own type.
_STRICT = ConfigDict(extra="forbid")


@dataclass(frozen=True, config=_STRICT)
class TaskConfig:
    """A workflow task: the class that runs it and the environment class it drives."""

    _target_: str
    """Import path of the ``holosoma.tasks.base.Task`` subclass that runs this workflow."""

    env_base: str
    """Import path of the class that ``env_class`` must be or subclass."""


@dataclass(frozen=True, config=_STRICT)
class TrainTaskConfig(TaskConfig):
    """Training a policy."""

    resume_from: str | None = None
    """Checkpoint to resume training from: restores the full algorithm state (for distillation this
    includes student, critic, optimizer and storage)."""

    teacher_checkpoint: str | None = None
    """Optional comma-separated teacher-only checkpoints, loaded into the student-teacher policy's
    teacher branch after setup. Orthogonal to ``resume_from``: only populates the frozen teacher
    MLP from a trained PPO-actor checkpoint."""

    multigpu: bool = False
    """Train across every GPU on the machine (used by launchers that start the distributed run)."""


@dataclass(frozen=True, config=_STRICT)
class EvalTaskConfig(TaskConfig):
    """Evaluation of a saved checkpoint."""

    checkpoint: str | None = None
    """Checkpoint to evaluate: a local path or ``wandb://<entity>/<project>/<run_id>[/<checkpoint>]``."""

    max_eval_steps: int | None = None
    """Maximum number of evaluation steps (None for unlimited)."""

    export_onnx: bool = True
    """Export the policy as an ONNX model next to the checkpoint."""

    callbacks: EvalCallbacksConfig = EvalCallbacksConfig()
    """Callbacks run during evaluation."""


TaskConfigType = TaskConfig | TrainTaskConfig | EvalTaskConfig
"""The ``ExperimentConfig.task`` type: a serialized task validates back to the right subclass."""
