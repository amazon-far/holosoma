"""Student/teacher modules for distillation algorithms.

Parallels ``ppo_modules.py`` (``PPOActor``/``PPOCritic``). Produced by the
distillation algorithm's ``setup()``. Each teacher is rebuilt from its own
checkpoint (:mod:`holosoma.agents.modules.teacher_policies`) and frozen; the
student MLP is trained against teacher actions (DAgger) plus PPO advantages via
the critic.

Obs layout (matches the distillation rollout storage):
- ``actor_obs``: proprioception only, consumed by the student.
- ``depth_obs``: raw depth image, consumed by the depth CNN backbone whose
  latent is concatenated onto ``actor_obs`` before the student MLP. Only the
  ``Depth*`` modules take it.
- ``teacher_obs``: privileged obs seen by the frozen teacher (same layout as
  what the teacher's original PPO training saw in its actor group). It is raw;
  each teacher normalizes it with the frozen statistics of its own PPO
  ``EmpiricalNormalization``, if it was trained with one.
- ``critic_obs``: privileged obs seen by the critic.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
from loguru import logger
from torch import nn
from torch.distributions import Normal

from holosoma.agents.modules.teacher_policies import TeacherPolicy, build_mlp, build_teacher, teacher_from_checkpoint

MIN_ACTION_STD = 1.0e-6


class TeacherActions(NamedTuple):
    """DAgger labels from :meth:`StudentTeacher.teacher_act`."""

    actions: torch.Tensor
    """``[N, num_actions]`` teacher actions; zero in rows without a teacher."""

    valid: torch.Tensor
    """``[N]`` bool: whether a teacher produced the row. Losses must ignore invalid rows."""


class _UnsetTeacher(nn.Module):
    """Placeholder for a teacher slot whose teacher has not been built yet."""

    def __init__(self, num_obs: int, num_actions: int) -> None:
        super().__init__()
        self.num_obs = num_obs
        self.num_actions = num_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("This teacher slot is empty: load a teacher checkpoint or a full distillation checkpoint.")


class StudentTeacher(nn.Module):
    """Proprio-only student + one or more frozen privileged teachers.

    Student consumes ``actor_obs`` (+ optional depth latent from a subclass).
    Each teacher consumes the full ``teacher_obs``. Teacher slots start empty and
    are filled from a teacher's own checkpoint (:meth:`load_teacher_checkpoint`) or,
    when resuming, rebuilt from the construction configs a full checkpoint saved
    (:meth:`build_teachers`). Teachers are frozen (``eval()`` + no grad).

    Routing across teachers is driven by an explicit ``teacher_idx`` argument
    to :meth:`teacher_act` rather than a reserved column of the observation:
    where that index comes from, and what counts as "no teacher", are decisions
    for the application. See :meth:`teacher_act`.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        student_hidden_dims: list[int],
        activation: str = "ELU",
        init_noise_std: float = 0.01,
        num_teachers: int = 1,
    ):
        super().__init__()
        if num_teachers < 1:
            raise ValueError(f"num_teachers must be >= 1, got {num_teachers}")
        self.num_actor_obs = num_actor_obs
        self.num_teacher_obs = num_teacher_obs
        self.num_actions = num_actions
        # Each teacher sees the whole teacher_obs. An application that smuggles
        # routing metadata through the observation is responsible for stripping
        # it before calling teacher_act, and for declaring num_teacher_obs as the
        # width its teachers actually expect.
        self.student = build_mlp(num_actor_obs, student_hidden_dims, num_actions, activation)
        self.teachers = nn.ModuleList([_UnsetTeacher(num_teacher_obs, num_actions) for _ in range(num_teachers)])

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)
        self.project_action_std()

        self._loaded_teacher = [False] * num_teachers

        print(f"Student MLP: {self.student}")

    @property
    def num_teachers(self) -> int:
        return len(self.teachers)

    def train(self, mode: bool = True) -> StudentTeacher:
        super().train(mode)
        # Teachers stay frozen whatever mode the student is in.
        self.teachers.eval()
        return self

    def set_teacher(self, teacher_index: int, teacher: TeacherPolicy, loaded: bool = True) -> None:
        """Install ``teacher`` in slot ``teacher_index``, frozen and on the student's device."""
        if teacher_index < 0 or teacher_index >= self.num_teachers:
            raise IndexError(f"teacher_index={teacher_index} out of range for {self.num_teachers} teachers")
        if (teacher.num_obs, teacher.num_actions) != (self.num_teacher_obs, self.num_actions):
            raise ValueError(
                f"Teacher maps {teacher.num_obs} observations to {teacher.num_actions} actions, but this "
                f"policy's teachers map {self.num_teacher_obs} to {self.num_actions}."
            )
        teacher = teacher.to(self.std.device).eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        self.teachers[teacher_index] = teacher
        self._loaded_teacher[teacher_index] = loaded
        logger.info(f"Teacher[{teacher_index}]: {type(teacher).__name__}")

    def load_teacher_checkpoint(self, checkpoint: dict[str, Any], teacher_index: int = 0) -> None:
        """Rebuild slot ``teacher_index``'s teacher from its own training checkpoint and load it."""
        teacher = teacher_from_checkpoint(checkpoint, num_obs=self.num_teacher_obs, num_actions=self.num_actions)
        self.set_teacher(teacher_index, teacher)

    def teacher_construction_configs(self) -> list[dict[str, Any] | None]:
        """Per slot, the config :meth:`build_teachers` rebuilds the teacher from; ``None`` if empty."""
        return [
            teacher.construction_config() if isinstance(teacher, TeacherPolicy) else None for teacher in self.teachers
        ]

    def build_teachers(self, configs: list[dict[str, Any] | None]) -> None:
        """Rebuild teacher structures from :meth:`teacher_construction_configs`, before loading a state dict.

        Rebuilt teachers count as not loaded until the caller restores their readiness.
        """
        if len(configs) != self.num_teachers:
            raise ValueError(f"Expected {self.num_teachers} teacher configs, got {len(configs)}.")
        for index, config in enumerate(configs):
            if config is None:
                self.teachers[index] = _UnsetTeacher(self.num_teacher_obs, self.num_actions)
                self._loaded_teacher[index] = False
            else:
                self.set_teacher(index, build_teacher(config), loaded=False)

    @property
    def loaded_teachers(self) -> tuple[bool, ...]:
        return tuple(self._loaded_teacher)

    def set_loaded_teachers(self, loaded: list[bool] | tuple[bool, ...]) -> None:
        """Restore teacher-readiness state from a full training checkpoint."""
        if len(loaded) != self.num_teachers:
            raise ValueError(f"Expected readiness state for {self.num_teachers} teachers, got {len(loaded)}.")
        if any(type(value) is not bool for value in loaded):
            raise TypeError("Teacher-readiness checkpoint metadata must contain only booleans.")
        self._loaded_teacher = list(loaded)

    def require_loaded_teachers(self) -> None:
        """Fail before rollout if any frozen teacher still has random weights."""
        missing = [index for index, loaded in enumerate(self._loaded_teacher) if not loaded]
        if missing:
            raise RuntimeError(
                f"Teacher weights are not loaded for slots {missing}. "
                "Set training.teacher_checkpoint for a new run or resume from a full "
                "distillation checkpoint."
            )

    # ---------- action interfaces (parallel PPOActor) ----------

    def _student_input(self, actor_obs: torch.Tensor, depth_obs: torch.Tensor | None = None) -> torch.Tensor:
        return actor_obs

    @property
    def action_mean(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs: torch.Tensor, depth_obs: torch.Tensor | None = None) -> None:
        self.project_action_std()
        mean = self.student(self._student_input(actor_obs, depth_obs))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    @torch.no_grad()
    def project_action_std(self) -> None:
        """Keep the directly optimized Gaussian scale finite and positive."""
        if not torch.isfinite(self.std).all():
            raise FloatingPointError("Student action standard deviation became non-finite.")
        self.std.clamp_(min=MIN_ACTION_STD)

    def act(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        self.update_distribution(
            policy_state_dict["actor_obs"],
            policy_state_dict.get("depth_obs"),
        )
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.student(self._student_input(policy_state_dict["actor_obs"], policy_state_dict.get("depth_obs")))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    @torch.no_grad()
    def teacher_act(self, teacher_obs: torch.Tensor, teacher_idx: torch.Tensor | None = None) -> TeacherActions:
        """Deterministic teacher forward. Used to generate DAgger labels.

        ``teacher_obs`` is the raw privileged observation; each teacher applies
        its own frozen input processing (e.g. a PPO teacher's observation
        normalizer). ``teacher_idx`` optionally selects a teacher per env:

        - ``None`` (default): every env goes to ``self.teachers[0]``.
        - an integer tensor of shape ``[N]``: env ``n`` is routed to
          ``self.teachers[teacher_idx[n]]``. ``-1`` means "no teacher applies":
          the row is marked invalid in the returned :class:`TeacherActions`, and
          the algorithms exclude it from the behaviour-cloning loss.

        The index is an argument rather than a reserved observation column so
        that this module has no opinion on how an application derives it. See
        the ``DistillationPPO._dagger_loss`` docstring for the corresponding
        loss-side extension point.
        """
        self.require_loaded_teachers()
        if teacher_obs.ndim != 2 or teacher_obs.shape[1] != self.num_teacher_obs:
            raise ValueError(
                f"teacher_obs must have shape [N, {self.num_teacher_obs}], got {tuple(teacher_obs.shape)}."
            )
        if teacher_idx is None:
            if self.num_teachers != 1:
                raise ValueError(f"teacher_idx is required when {self.num_teachers} teachers are configured.")
            actions = self.teachers[0](teacher_obs)
            return TeacherActions(actions, torch.ones(actions.shape[0], dtype=torch.bool, device=actions.device))

        if teacher_idx.shape != (teacher_obs.shape[0],):
            raise ValueError(f"teacher_idx must have shape [{teacher_obs.shape[0]}], got {tuple(teacher_idx.shape)}.")
        if teacher_idx.device != teacher_obs.device:
            raise ValueError(f"teacher_idx is on {teacher_idx.device}, but teacher_obs is on {teacher_obs.device}.")
        if teacher_idx.dtype == torch.bool or teacher_idx.is_complex():
            raise TypeError(f"teacher_idx must contain integer-valued indices, got {teacher_idx.dtype}.")
        if teacher_idx.is_floating_point():
            if not torch.isfinite(teacher_idx).all() or not torch.equal(teacher_idx, teacher_idx.round()):
                raise ValueError("teacher_idx must contain finite integer-valued indices.")

        idx = teacher_idx.to(dtype=torch.long)
        invalid = (idx < -1) | (idx >= self.num_teachers)
        if invalid.any():
            invalid_values = torch.unique(idx[invalid]).tolist()
            raise IndexError(
                f"teacher_idx contains invalid values {invalid_values}; expected -1 or "
                f"indices in [0, {self.num_teachers - 1}]."
            )

        out = teacher_obs.new_zeros((teacher_obs.shape[0], self.num_actions))
        for i in range(self.num_teachers):
            mask = idx == i
            if mask.any():
                out[mask] = self.teachers[i](teacher_obs[mask])
        return TeacherActions(out, idx >= 0)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass


class DepthStudentTeacher(StudentTeacher):
    """Student-teacher composite with a CNN depth backbone on the student side.

    The backbone is built externally and passed in so the config layer can swap
    architectures without touching this module.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        depth_backbone: nn.Module,
        depth_output_dim: int,
        student_hidden_dims: list[int],
        activation: str = "ELU",
        init_noise_std: float = 0.01,
        num_teachers: int = 1,
    ):
        # Student sees actor_obs (proprio) concatenated with depth latent.
        super().__init__(
            num_actor_obs=num_actor_obs + depth_output_dim,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            student_hidden_dims=student_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            num_teachers=num_teachers,
        )
        self.proprio_dim = num_actor_obs
        self.depth_output_dim = depth_output_dim
        self.depth_backbone = depth_backbone

    def _student_input(self, actor_obs: torch.Tensor, depth_obs: torch.Tensor | None = None) -> torch.Tensor:
        if depth_obs is None:
            raise ValueError("DepthStudentTeacher.student requires depth_obs")
        return self._combine_student_input(actor_obs, self.depth_backbone(depth_obs))

    def _combine_student_input(self, actor_obs: torch.Tensor, depth_latent: torch.Tensor) -> torch.Tensor:
        """Student MLP input from proprioception and the depth latent.

        Override this, not ``_student_input``, to prepare the student's input (e.g. to
        normalize ``actor_obs``): :meth:`export_modules` includes it in the exported student.
        """
        return torch.cat([actor_obs, depth_latent], dim=-1)

    def export_modules(self) -> tuple[nn.Module, nn.Module]:
        """The deployed ``(depth_backbone, student)`` pair.

        The backbone maps a depth image to its latent. The student maps
        ``[actor_obs | depth_latent]`` to actions, applying :meth:`_combine_student_input`.
        """
        return self.depth_backbone, _ExportedDepthStudent(self)


class _ExportedDepthStudent(nn.Module):
    """Student half of a depth policy export: ``[actor_obs | depth_latent]`` -> actions."""

    def __init__(self, policy: DepthStudentTeacher) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        actor_obs, depth_latent = obs.split([self.policy.proprio_dim, self.policy.depth_output_dim], dim=-1)
        return self.policy.student(self.policy._combine_student_input(actor_obs, depth_latent))


class StudentTeacherCritic(StudentTeacher):
    """Student-teacher with a privileged critic for PPO updates."""

    def __init__(
        self,
        num_actor_obs: int,
        num_teacher_obs: int,
        num_critic_obs: int,
        num_actions: int,
        student_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str = "ELU",
        init_noise_std: float = 0.01,
        num_teachers: int = 1,
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            student_hidden_dims=student_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            num_teachers=num_teachers,
        )
        self.num_critic_obs = num_critic_obs
        self.critic = build_mlp(num_critic_obs, critic_hidden_dims, 1, activation)
        print(f"Critic MLP: {self.critic}")

    def evaluate(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(policy_state_dict["critic_obs"])


class DepthStudentTeacherCritic(DepthStudentTeacher):
    """Depth-aware student, frozen teacher, and privileged critic."""

    def __init__(
        self,
        num_actor_obs: int,
        num_teacher_obs: int,
        num_critic_obs: int,
        num_actions: int,
        depth_backbone: nn.Module,
        depth_output_dim: int,
        student_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str = "ELU",
        init_noise_std: float = 0.01,
        num_teachers: int = 1,
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            depth_backbone=depth_backbone,
            depth_output_dim=depth_output_dim,
            student_hidden_dims=student_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            num_teachers=num_teachers,
        )
        self.num_critic_obs = num_critic_obs
        self.critic = build_mlp(num_critic_obs, critic_hidden_dims, 1, activation)
        print(f"Critic MLP: {self.critic}")

    def evaluate(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(policy_state_dict["critic_obs"])
