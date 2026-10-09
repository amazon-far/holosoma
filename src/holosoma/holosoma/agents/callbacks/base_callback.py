from __future__ import annotations

from typing import Any

from torch.nn import Module


class RLEvalCallback(Module):
    def __init__(self, config: Any, training_loop: Any) -> None:
        super().__init__()
        self.config = config
        self.training_loop = training_loop
        self.device = self.training_loop.device

    def on_pre_evaluate_policy(self) -> None:
        pass

    def on_pre_eval_env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        return actor_state

    def on_post_eval_env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        return actor_state

    def on_post_evaluate_policy(self) -> None:
        pass
