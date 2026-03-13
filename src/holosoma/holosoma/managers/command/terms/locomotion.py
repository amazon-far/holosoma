"""Locomotion command sampling and gait state terms."""

from __future__ import annotations

from typing import Any, Sequence, cast

from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.torch_utils import torch_rand_float


class LocomotionCommand(CommandTermBase):
    """Stateful command term that owns command buffers and resampling logic."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        ranges = params.get("command_ranges")
        if ranges is None:
            raise ValueError("LocomotionCommand requires 'command_ranges' in params.")
        self.command_ranges: dict[str, Sequence[float]] = {key: tuple(value) for key, value in ranges.items()}
        self.stand_prob: float = float(params.get("stand_prob", 0.0))
        self.command_dim: int = params.get("command_dim", 3)
        self.commands: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle hooks
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        env = self.env
        commands = torch.zeros(env.num_envs, self.command_dim, dtype=torch.float32, device=env.device)
        self.commands = commands
        if hasattr(env, "simulator"):
            env.simulator.commands = commands

    def reset(self, env_ids: torch.Tensor | None) -> None:
        commands = self.commands
        if commands is None:
            return
        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        self._resample(idx)

    def step(self) -> None:
        commands = self.commands
        if commands is None or self.env.is_evaluating:
            return

        command_cfg = getattr(self.manager, "command_cfg", None) if hasattr(self, "manager") else None
        resample_time = getattr(command_cfg, "locomotion_command_resampling_time", None) if command_cfg else None
        if resample_time is None or resample_time <= 0:
            return

        interval = int(resample_time / self.env.dt)
        if interval <= 0 or not hasattr(self.env, "episode_length_buf"):
            return

        env_ids = (self.env.episode_length_buf % interval == 0).nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        self._resample(env_ids.to(device=self.env.device, dtype=torch.long))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _resample(self, env_ids: torch.Tensor) -> None:
        commands = self.commands
        if commands is None or env_ids.numel() == 0:
            return

        device = self.env.device
        ranges = self.command_ranges

        commands[env_ids, 0] = torch_rand_float(
            ranges["lin_vel_x"][0],
            ranges["lin_vel_x"][1],
            (env_ids.shape[0], 1),
            device=device,
        ).squeeze(1)
        commands[env_ids, 1] = torch_rand_float(
            ranges["lin_vel_y"][0],
            ranges["lin_vel_y"][1],
            (env_ids.shape[0], 1),
            device=device,
        ).squeeze(1)
        commands[env_ids, 2] = torch_rand_float(
            ranges["ang_vel_yaw"][0],
            ranges["ang_vel_yaw"][1],
            (env_ids.shape[0], 1),
            device=device,
        ).squeeze(1)

        manager = getattr(self, "manager", None)
        if manager is not None:
            gait_state = manager.get_state("locomotion_gait")
        else:
            gait_state = None

        if gait_state is not None:
            cast("LocomotionGait", gait_state).resample_frequency(env_ids)

        if self.stand_prob > 0.0:
            stand_mask = torch.rand(env_ids.shape[0], device=device) <= self.stand_prob
            if stand_mask.any():
                commands[env_ids[stand_mask], :3] = 0.0

    def _ensure_index_tensor(self, env_ids: torch.Tensor | Sequence[int] | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.env.device, dtype=torch.long)


class LocomotionGait(CommandTermBase):
    """Stateful term that owns gait phase buffers and updates them each step."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.gait_period: float = float(params.get("gait_period", 1.0))
        self.gait_period_randomization_width: float = float(params.get("gait_period_randomization_width", 0.0))
        self.randomize_phase: bool = bool(params.get("randomize_phase", True))
        self.stand_phase_value: float = float(params.get("stand_phase_value", torch.pi))

        self.phase_offset: torch.Tensor | None = None
        self.phase: torch.Tensor | None = None
        self.gait_freq: torch.Tensor | None = None
        self.phase_dt: torch.Tensor | None = None
        self.mean_gait_freq: float = 1.0 / self.gait_period

    def setup(self) -> None:
        env = self.env
        device = env.device
        num_envs = env.num_envs

        self.phase_offset = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.phase = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.gait_freq = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)
        self.phase_dt = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)

        self._initialize_indices(None, evaluating=env.is_evaluating)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        self._initialize_indices(env_ids, evaluating=self.env.is_evaluating)

    def step(self) -> None:
        if self.phase is None or self.phase_offset is None or self.phase_dt is None:
            return

        env = self.env
        phase_tp1 = env.episode_length_buf.unsqueeze(1) * self.phase_dt + self.phase_offset
        self.phase.copy_(torch.fmod(phase_tp1 + torch.pi, 2 * torch.pi) - torch.pi)

        command_tensor = getattr(self.manager, "commands", None) if hasattr(self, "manager") else None
        if command_tensor is None:
            return

        stand_mask = torch.logical_and(
            torch.linalg.norm(command_tensor[:, :2], dim=1) < 0.01,
            torch.abs(command_tensor[:, 2]) < 0.01,
        )
        if stand_mask.any():
            self.phase[stand_mask] = torch.full(
                (int(stand_mask.sum().item()), 2), self.stand_phase_value, device=env.device
            )

    def set_eval_mode(self, evaluating: bool) -> None:
        self._initialize_indices(None, evaluating=evaluating)

    def resample_frequency(self, env_ids: torch.Tensor) -> None:
        if self.gait_freq is None or self.phase_dt is None:
            return

        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        if self.env.is_evaluating or self.gait_period_randomization_width <= 0.0:
            self.gait_freq[idx] = self.mean_gait_freq
        else:
            low = self.mean_gait_freq - self.gait_period_randomization_width
            high = self.mean_gait_freq + self.gait_period_randomization_width
            self.gait_freq[idx] = torch_rand_float(low, high, (idx.shape[0], 1), device=self.env.device)

        self.phase_dt[idx] = 2 * torch.pi * self.env.dt * self.gait_freq[idx]

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #

    def _initialize_indices(self, env_ids: torch.Tensor | None, *, evaluating: bool) -> None:
        if self.phase_offset is None or self.phase is None or self.gait_freq is None or self.phase_dt is None:
            return

        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        if evaluating:
            self.phase_offset[idx, 0] = 0.0
            self.phase_offset[idx, 1] = -torch.pi
        elif self.randomize_phase:
            self.phase_offset[idx, 0] = torch_rand_float(
                -torch.pi, torch.pi, (idx.shape[0], 1), device=self.env.device
            ).squeeze(1)
            self.phase_offset[idx, 1] = torch.fmod(self.phase_offset[idx, 0] + 2 * torch.pi, 2 * torch.pi) - torch.pi
        else:
            self.phase_offset[idx] = 0.0

        self.phase[idx] = self.phase_offset[idx]
        self.resample_frequency(idx)

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
