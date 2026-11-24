"""Curriculum hooks for locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from loguru import logger

from holosoma.managers.curriculum.base import CurriculumTermBase


class AverageEpisodeLengthTracker(CurriculumTermBase):
    """Track moving average of episode length for locomotion tasks."""

    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)
        params = cfg.params or {}
        base_num_compute_average_epl = float(params.get("num_compute_average_epl", 1000))
        base_denominator = getattr(env, "BASE_NUM_ENVS", env.num_envs)
        self.num_compute_average_epl = max(1, int(base_num_compute_average_epl * env.num_envs / base_denominator))
        self.average_episode_length = torch.tensor(0.0, device=env.device, dtype=torch.float)
        self._suppress_next_update = False
        self.metrics: dict[str, torch.Tensor] = {}

    def setup(self) -> None:
        self.average_episode_length = torch.as_tensor(
            float(self.average_episode_length), device=self.env.device, dtype=torch.float
        )

    def reset(self, env_ids: Any) -> None:
        if env_ids is None:
            return

        if not torch.is_tensor(env_ids):
            env_ids_tensor = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        else:
            env_ids_tensor = env_ids.to(device=self.env.device, dtype=torch.long)

        if env_ids_tensor.numel() == 0:
            return

        pending = self.env._pending_episode_lengths
        mask_tensor = self.env._pending_episode_update_mask

        update_mask = mask_tensor.index_select(0, env_ids_tensor)
        if not torch.any(update_mask):
            return
        active_ids = env_ids_tensor[update_mask]

        if active_ids.numel() == 0:
            return

        episode_lengths = pending.index_select(0, active_ids).to(dtype=torch.float)
        if episode_lengths.numel() == 0:
            return

        self.update(active_ids, episode_lengths)

        zeros_long = torch.zeros(active_ids.shape, device=pending.device, dtype=pending.dtype)
        pending.index_copy_(0, active_ids, zeros_long)

        zeros_bool = torch.zeros(active_ids.shape, device=mask_tensor.device, dtype=mask_tensor.dtype)
        mask_tensor.index_copy_(0, active_ids, zeros_bool)

    def step(self) -> None:
        return

    def update(self, env_ids: torch.Tensor, episode_lengths: torch.Tensor) -> None:
        if self._suppress_next_update:
            self._suppress_next_update = False
            return

        num = env_ids.numel()
        if num == 0:
            return
        current_average = torch.mean(episode_lengths.to(dtype=torch.float), dtype=torch.float)
        weight = min(num / self.num_compute_average_epl, 1.0)
        self.average_episode_length = self.average_episode_length * (1 - weight) + current_average * weight

    def update_metrics(self) -> None:
        self.metrics["average_episode_length"] = self.average_episode_length.detach().cpu()

    def suppress_next_update(self) -> None:
        self._suppress_next_update = True

    def get_average(self) -> torch.Tensor:
        return self.average_episode_length

    def set_average(self, value: float | torch.Tensor, *, suppress_update: bool = True) -> None:
        self.average_episode_length = torch.as_tensor(float(value), device=self.env.device, dtype=torch.float)
        if suppress_update:
            self._suppress_next_update = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "average_episode_length": self.average_episode_length.detach().to("cpu"),
            "suppress_next_update": self._suppress_next_update,
        }

    def after_checkpoint_load(self) -> None:
        # The first reset after a resume sees episodes that restarted at 0; don't let it overwrite the restored average.
        self.suppress_next_update()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        avg = state.get("average_episode_length")
        if avg is not None:
            self.average_episode_length = torch.as_tensor(float(avg), device=self.env.device, dtype=torch.float)
        self._suppress_next_update = bool(state.get("suppress_next_update", False))


class PenaltyCurriculum(CurriculumTermBase):
    """Stateful penalty curriculum that scales reward term weights based on episode length.

    This curriculum term adaptively scales penalty reward weights during training.
    When episodes are short (robot falls quickly), penalties are reduced to make
    learning easier. As episodes get longer (robot stays up), penalties gradually
    increase to refine behavior.
    """

    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)

        # Get parameters from config
        params = cfg.params
        self.enabled = params.get("enabled", True)
        self.tag = params.get("tag", "penalty_curriculum")
        self.min_scale = float(params.get("min_scale", 0.0))
        self.max_scale = float(params.get("max_scale", 1.0))
        self.level_down_threshold = float(params.get("level_down_threshold", 150.0))
        self.level_up_threshold = float(params.get("level_up_threshold", 850.0))
        self.degree = float(params.get("degree", 0.0))

        # State variables (previously stored on env)
        self.current_scale = float(params.get("initial_scale", 1.0))
        self.penalty_reward_names: list[str] = []
        self.original_weights: dict[str, float] = {}
        self.metrics: dict[str, torch.Tensor] = {}

    def setup(self) -> None:
        """Setup penalty curriculum - identify rewards and apply initial scaling."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        # Identify penalty rewards by tag using public reward manager APIs
        for term_name in self.env.reward_manager.active_terms:
            term_cfg = self.env.reward_manager.get_term_cfg(term_name)
            if self.tag in term_cfg.tags:
                self.penalty_reward_names.append(term_name)

        # The user explicitly enabled the curriculum and named a tag; if no active reward term carries
        # it (typo or tag/term mismatch), the curriculum scales nothing yet reports itself enabled
        # below. Warn rather than run a silently inert curriculum.
        if not self.penalty_reward_names:
            logger.warning(
                f"PenaltyCurriculum enabled but no active reward term carries tag '{self.tag}'; "
                f"the penalty curriculum will have no effect."
            )

        # Store original weights and apply initial scaling
        for name in self.penalty_reward_names:
            if name not in self.env.reward_manager.active_terms:
                continue

            term_cfg = self.env.reward_manager.get_term_cfg(name)
            # Store original weight
            self.original_weights[name] = float(term_cfg.weight)
            # Apply initial scale
            scaled_cfg = replace(term_cfg, weight=term_cfg.weight * self.current_scale)
            self.env.reward_manager.set_term_cfg(name, scaled_cfg)

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None) -> None:
        """Update penalty scale based on average episode length."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        if not self.penalty_reward_names or not self.original_weights:
            return

        average_length = self._average_episode_length()

        # Update current scale based on episode length
        if average_length < self.level_down_threshold:
            self.current_scale *= 1.0 - self.degree
        elif average_length > self.level_up_threshold:
            self.current_scale *= 1.0 + self.degree

        # Clamp scale
        self.current_scale = float(np.clip(self.current_scale, self.min_scale, self.max_scale))

        self._apply_scale()

    def _average_episode_length(self) -> float:
        tracker = self.manager.get_term("average_episode_tracker") if self.manager is not None else None
        if not isinstance(tracker, AverageEpisodeLengthTracker):
            raise RuntimeError("AverageEpisodeLengthTracker is not registered with the curriculum manager.")
        return float(tracker.get_average().detach().cpu().item())

    def _apply_weights(self) -> None:
        """Set each tagged penalty weight to ``original_weight * current_scale``."""
        for name in self.penalty_reward_names:
            if name not in self.original_weights or name not in self.env.reward_manager.active_terms:
                continue
            term_cfg = self.env.reward_manager.get_term_cfg(name)
            scaled_cfg = replace(term_cfg, weight=self.original_weights[name] * self.current_scale)
            self.env.reward_manager.set_term_cfg(name, scaled_cfg)

    def _apply_scale(self) -> None:
        """Apply ``current_scale`` to the penalty weights and mirror it for logging."""
        self._apply_weights()

        self.metrics["penalty_scale"] = torch.tensor(self.current_scale, dtype=torch.float)

    def state_dict(self) -> dict[str, Any]:
        """Curriculum state for checkpointing.

        Empty when disabled: the weights were never scaled, so there is no scale to resume.
        """
        if not self.enabled:
            return {}
        return {"current_scale": float(self.current_scale)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the curriculum scale and re-apply it to the reward weights immediately."""
        if not self.enabled or "current_scale" not in state:
            return
        self.current_scale = float(np.clip(float(state["current_scale"]), self.min_scale, self.max_scale))
        self._apply_scale()

    def step(self) -> None:
        """Clamp penalty scale within bounds each step."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        if not self.penalty_reward_names or not self.original_weights:
            return

        # Clamp current_scale
        self.current_scale = float(np.clip(self.current_scale, self.min_scale, self.max_scale))

        # Re-apply clamped scale to weights
        self._apply_weights()
