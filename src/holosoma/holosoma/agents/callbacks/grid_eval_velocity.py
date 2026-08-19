"""Grid evaluation velocity callback for locomotion policies.

Registers velocity sweep axes with the GridConditionManager and
overrides ``env.command_manager.commands`` each step so each condition env
walks at its assigned velocity.

Each non-trivial axis is swept independently (other axes default to 0).
E.g. lin_vel_x=[0.25, 0.5, 0.75, 1.0], ang_vel_yaw=[0.3, 0.5, 0.7]
produces 4 + 3 = 7 conditions (not 12).

Warmup: the first ``warmup_s`` seconds use zero velocity on all envs.

Usage example (4 velocity conditions):

    --grid-velocity.config.enabled
    --grid-velocity.config.lin-vel-x 0.25 0.5 0.75 1.0
    --training.num-envs 4
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import GridEvalVelocityConfig
from holosoma.utils.safe_torch_import import torch


class GridEvalVelocityCallback(RLEvalCallback):
    """Assigns per-env velocity commands from a grid.

    Only handles velocity command injection. Recording is done by
    EvalRecordingCallback.
    """

    def __init__(self, config: GridEvalVelocityConfig, training_loop: Any = None):
        super().__init__(config, training_loop)

        self._command_tensor: torch.Tensor | None = None
        self._zero_command_tensor: torch.Tensor | None = None
        self._num_conditions: int = 0
        self._warmup_steps: int = 0
        self._step_count: int = 0
        self._recording_cb: Any = None

    def on_pre_evaluate_policy(self) -> None:
        self._recording_cb = self._require_recording_cb()

        conditions = self._build_velocity_conditions(self.config)
        cm = self._recording_cb.condition_manager

        cm.register_axis(
            ["lin_vel_x", "lin_vel_y", "ang_vel_yaw"],
            [(c["lin_vel_x"], c["lin_vel_y"], c["ang_vel_yaw"]) for c in conditions],
            group="velocity",
        )

        logger.info(f"GridEvalVelocityCallback: {len(conditions)} velocity conditions")

    def _deferred_setup(self) -> None:
        """Build command tensors after condition manager is finalized."""
        self._recording_cb.ensure_finalized()

        env = self._get_env()
        cm = self._recording_cb.condition_manager
        conditions = cm.conditions
        self._num_conditions = cm.num_conditions

        self._command_tensor = self._build_command_tensor(conditions, env.device)
        self._zero_command_tensor = torch.zeros_like(self._command_tensor)

        dt = float(env.dt)
        self._warmup_steps = int(self.config.warmup_s / dt) if self.config.warmup_s > 0 else 0
        cm.warmup_steps = self._warmup_steps

        self._recording_cb.set_metadata("warmup_s", self.config.warmup_s)
        self._recording_cb.set_metadata("warmup_steps", self._warmup_steps)

        logger.info(
            f"GridEvalVelocityCallback: {self._num_conditions} total conditions, warmup={self._warmup_steps} steps"
        )

    def on_pre_eval_env_step(self, actor_state: dict) -> dict:
        if self._step_count == 0:
            self._deferred_setup()

        env = self._get_env()
        commands = env.command_manager.commands
        num_c = self._num_conditions

        if self._step_count < self._warmup_steps:
            commands[:num_c] = self._zero_command_tensor
        else:
            commands[:num_c] = self._command_tensor

        self._step_count += 1
        return actor_state

    @staticmethod
    def _build_velocity_conditions(config: GridEvalVelocityConfig) -> list[dict[str, float]]:
        """Build independent velocity conditions from config.

        Each non-trivial axis is swept separately; other axes default to 0.
        Deduplicates here because multiple axes that include 0.0 would
        otherwise produce identical (0,0,0) entries — e.g.
        lin_vel_x=(0.0, 0.5) and lin_vel_y=(0.0, 0.3) both generate
        a zero-velocity condition.
        """
        zeros = {"lin_vel_x": 0.0, "lin_vel_y": 0.0, "ang_vel_yaw": 0.0}
        axes = {
            "lin_vel_x": list(config.lin_vel_x),
            "lin_vel_y": list(config.lin_vel_y),
            "ang_vel_yaw": list(config.ang_vel_yaw),
        }

        seen: set[tuple[float, ...]] = set()
        conditions: list[dict[str, float]] = []
        for axis_name, values in axes.items():
            if values == [0.0]:
                continue
            for val in values:
                cond = {**zeros, axis_name: val}
                key = (cond["lin_vel_x"], cond["lin_vel_y"], cond["ang_vel_yaw"])
                if key not in seen:
                    seen.add(key)
                    conditions.append(cond)
        return conditions or [zeros.copy()]

    def _build_command_tensor(
        self,
        conditions: list[dict[str, Any]],
        device: torch.device,
    ) -> torch.Tensor:
        """Convert conditions to [num_conditions, 3] command tensor.

        Extracts velocity keys from conditions which may also contain
        non-velocity keys from other sweep dimensions.
        """
        num_c = len(conditions)
        commands = torch.zeros(num_c, 3, dtype=torch.float32, device=device)
        for i, cond in enumerate(conditions):
            commands[i, 0] = float(cond.get("lin_vel_x", 0.0))
            commands[i, 1] = float(cond.get("lin_vel_y", 0.0))
            commands[i, 2] = float(cond.get("ang_vel_yaw", 0.0))
        return commands
