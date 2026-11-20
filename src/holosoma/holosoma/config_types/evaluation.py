"""Configuration type for evaluating a trained checkpoint."""

from __future__ import annotations

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from holosoma.config_types.eval_callback import EvalCallbacksConfig


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class EvalConfig:
    """What to evaluate and what to do while evaluating it."""

    checkpoint: str | None = None
    """Checkpoint to evaluate: a local path or ``wandb://<entity>/<project>/<run_id>[/<checkpoint>]``."""

    max_eval_steps: int | None = None
    """Maximum number of evaluation steps (None for unlimited)."""

    export_onnx: bool = True
    """Export the policy as an ONNX model next to the checkpoint."""

    callbacks: EvalCallbacksConfig = EvalCallbacksConfig()
    """Callbacks run during evaluation."""
