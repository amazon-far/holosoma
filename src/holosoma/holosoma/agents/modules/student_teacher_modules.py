"""Student/teacher modules for distillation algorithms.

Parallels ``ppo_modules.py`` (``PPOActor``/``PPOCritic``). Produced by the
distillation algorithm's ``setup()``. Weights for the teacher MLP are loaded
from a standalone PPO-actor checkpoint and frozen; the student MLP is trained
against teacher actions (DAgger) plus PPO advantages via the critic.

Obs layout (matches the distillation rollout storage):
- ``actor_obs``: proprioception only, consumed by the student.
- ``depth_obs``: raw depth image, consumed by the depth CNN backbone whose
  latent is concatenated onto ``actor_obs`` before the student MLP.
- ``teacher_obs``: privileged obs seen by the frozen teacher (same layout as
  what the teacher's original PPO training saw in its actor group). It is raw;
  each teacher normalizes it with the frozen statistics of its own PPO
  ``EmpiricalNormalization``, if it was trained with one.
- ``critic_obs``: privileged obs seen by the critic.
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger
from torch import nn
from torch.distributions import Normal

MIN_ACTION_STD = 1.0e-6

# PPO builds its EmpiricalNormalization with the default eps, which its state
# dict does not store.
PPO_OBS_NORMALIZER_EPS = 1.0e-2


class TeacherObsNormalizer(nn.Module):
    """Frozen copy of a teacher's PPO observation normalizer.

    Computes ``(obs - mean) / (std + eps)`` exactly as PPO's
    ``EmpiricalNormalization`` does at inference. The statistics are loaded with
    the teacher and never updated, so train/eval mode does not affect it. The
    default buffers make it the identity, for teachers trained without one.
    """

    def __init__(self, num_obs: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, num_obs))
        self.register_buffer("std", torch.ones(1, num_obs))
        self.register_buffer("eps", torch.zeros(()))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / (self.std + self.eps)

    @torch.no_grad()
    def load_ppo_state_dict(self, state: dict[str, torch.Tensor] | None) -> None:
        """Load PPO's ``actor_obs_normalizer_state_dict``; ``None`` resets to identity."""
        if state is None:
            self.mean.zero_()
            self.std.fill_(1.0)
            self.eps.zero_()
            return
        if state["_mean"].shape != self.mean.shape or state["_std"].shape != self.std.shape:
            raise ValueError(
                f"Teacher obs normalizer has shape {tuple(state['_mean'].shape)}, "
                f"but teacher_obs has shape {tuple(self.mean.shape)}."
            )
        self.mean.copy_(state["_mean"])
        self.std.copy_(state["_std"])
        self.eps.fill_(PPO_OBS_NORMALIZER_EPS)


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
    """Build a plain MLP with the requested hidden layers and activation."""
    act_cls = getattr(nn, activation)
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class StudentTeacher(nn.Module):
    """Proprio-only student + one or more frozen privileged teachers.

    Student consumes ``actor_obs`` (+ optional depth latent from a subclass).
    Each teacher consumes the full ``teacher_obs`` and has its weights loaded
    via :meth:`load_teacher_state_dict`, then frozen (``eval()`` + no grad).

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
        teacher_hidden_dims: list[int],
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
        # Each teacher MLP sees the whole teacher_obs. An application that
        # smuggles routing metadata through the observation is responsible for
        # stripping it before calling teacher_act, and for declaring
        # num_teacher_obs as the width its teachers actually expect.
        self._teacher_in_dim = num_teacher_obs

        self.student = _build_mlp(num_actor_obs, student_hidden_dims, num_actions, activation)
        self.teachers = nn.ModuleList(
            [
                _build_mlp(self._teacher_in_dim, teacher_hidden_dims, num_actions, activation)
                for _ in range(num_teachers)
            ]
        )
        self.teacher_obs_normalizers = nn.ModuleList(
            [TeacherObsNormalizer(self._teacher_in_dim) for _ in range(num_teachers)]
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)
        self.project_action_std()

        self._loaded_teacher = [False] * num_teachers
        for t in self.teachers:
            t.eval()
            for p in t.parameters():
                p.requires_grad_(False)

        print(f"Student MLP: {self.student}")
        print(f"Teachers ({num_teachers}x): {self.teachers[0]}")

    @property
    def loaded_teacher(self) -> bool:
        return all(self._loaded_teacher)

    @property
    def num_teachers(self) -> int:
        return len(self.teachers)

    def load_training_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:
        """Load a full policy state dict, upgrading legacy single-teacher keys
        and checkpoints saved before teacher observation normalization."""
        legacy_keys = [key for key in state_dict if key.startswith("teacher.")]
        current_keys = [key for key in state_dict if key.startswith("teachers.")]
        if legacy_keys and current_keys:
            raise ValueError("Policy state dict mixes legacy 'teacher.*' and current 'teachers.*' keys.")
        if legacy_keys:
            state_dict = {
                f"teachers.0.{key[len('teacher.') :]}" if key.startswith("teacher.") else key: value
                for key, value in state_dict.items()
            }
        if not any(key.startswith("teacher_obs_normalizers.") for key in state_dict):
            # Those checkpoints ran their teachers on raw teacher_obs; keep doing so.
            logger.warning(
                "Checkpoint predates teacher observation normalization; its teachers run on raw "
                "teacher_obs. Provide training.teacher_checkpoint to reload them with their PPO normalizer."
            )
            for normalizer in self.teacher_obs_normalizers:
                normalizer.load_ppo_state_dict(None)
            state_dict = {**self.teacher_obs_normalizers.state_dict(prefix="teacher_obs_normalizers."), **state_dict}
        return super().load_state_dict(state_dict, strict=strict)

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

    def load_teacher_state_dict(self, ckpt: dict[str, Any], strict: bool = True, teacher_index: int = 0) -> None:
        """Load a teacher-PPO checkpoint into ``self.teachers[teacher_index]``.

        Accepts either a raw state dict or a holosoma algo checkpoint dict.
        Renames ``actor.*`` / ``actor_module.module.*`` keys to ``<bare>`` so
        they match a plain ``nn.Sequential`` layout. A checkpoint's
        ``actor_obs_normalizer_state_dict`` (PPO with ``empirical_normalization``)
        becomes the teacher's frozen observation normalizer; without one the
        teacher sees raw ``teacher_obs``.
        """
        if teacher_index < 0 or teacher_index >= len(self.teachers):
            raise IndexError(f"teacher_index={teacher_index} out of range for {len(self.teachers)} teachers")

        self.teacher_obs_normalizers[teacher_index].load_ppo_state_dict(ckpt.get("actor_obs_normalizer_state_dict"))

        state: Any = ckpt
        # unwrap common holosoma/rsl_rl ckpt wrappers. holosoma PPO saves as
        # {'actor_model_state_dict': ..., 'critic_model_state_dict': ...}; other
        # tooling stores a nested 'model_state_dict' / 'model' / 'state_dict'.
        for key in (
            "actor_model_state_dict",
            "model_state_dict",
            "model",
            "state_dict",
        ):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break

        # collect prefix candidates, longest first
        prefixes = (
            "actor_module.module.",
            "actor_module.",
            "actor.",
        )
        renamed: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            new_k = k
            for p in prefixes:
                if new_k.startswith(p):
                    new_k = new_k[len(p) :]
                    break
            renamed[new_k] = v

        target = self.teachers[teacher_index]
        teacher_keys = set(dict(target.named_parameters()).keys()) | set(dict(target.named_buffers()).keys())
        filtered = {k: v for k, v in renamed.items() if k in teacher_keys}
        missing = sorted(k for k in teacher_keys if k not in filtered)
        if missing and strict:
            raise RuntimeError(
                f"load_teacher_state_dict[{teacher_index}]: missing keys after renaming: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
                f"Available (post-rename): {list(renamed.keys())[:10]}"
            )
        target.load_state_dict(filtered, strict=strict)
        target.eval()
        for param in target.parameters():
            param.requires_grad_(False)
        self._loaded_teacher[teacher_index] = not missing

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
    def teacher_act(self, teacher_obs: torch.Tensor, teacher_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Deterministic teacher forward. Used to generate DAgger labels.

        ``teacher_obs`` is the raw privileged observation; each teacher MLP sees
        it through its own frozen observation normalizer (identity unless its
        checkpoint had one). ``teacher_idx`` optionally selects a teacher per env:

        - ``None`` (default): every env goes to ``self.teachers[0]``.
        - an integer tensor of shape ``[N]``: env ``n`` is routed to
          ``self.teachers[teacher_idx[n]]``. Negative entries mean "no teacher
          applies" and yield a zero action, which callers may use to mark
          samples to exclude from the behaviour-cloning loss.

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
            return self.teachers[0](self.teacher_obs_normalizers[0](teacher_obs))

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
                out[mask] = self.teachers[i](self.teacher_obs_normalizers[i](teacher_obs[mask]))
        return out

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
        teacher_hidden_dims: list[int],
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
            teacher_hidden_dims=teacher_hidden_dims,
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
        depth_latent = self.depth_backbone(depth_obs)
        return torch.cat([actor_obs, depth_latent], dim=-1)


class StudentTeacherCritic(StudentTeacher):
    """Student-teacher with a privileged critic for PPO updates."""

    def __init__(
        self,
        num_actor_obs: int,
        num_teacher_obs: int,
        num_critic_obs: int,
        num_actions: int,
        student_hidden_dims: list[int],
        teacher_hidden_dims: list[int],
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
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            num_teachers=num_teachers,
        )
        self.num_critic_obs = num_critic_obs
        self.critic = _build_mlp(num_critic_obs, critic_hidden_dims, 1, activation)
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
        teacher_hidden_dims: list[int],
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
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            num_teachers=num_teachers,
        )
        self.num_critic_obs = num_critic_obs
        self.critic = _build_mlp(num_critic_obs, critic_hidden_dims, 1, activation)
        print(f"Critic MLP: {self.critic}")

    def evaluate(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(policy_state_dict["critic_obs"])
