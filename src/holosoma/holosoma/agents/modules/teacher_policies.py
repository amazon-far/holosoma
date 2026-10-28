"""Frozen teacher policies for distillation, each rebuilt from its own checkpoint.

A teacher is the policy that produced its checkpoint, reconstructed from the
construction config saved with it and loaded strictly, so no learned component
can be dropped on the way. :func:`teacher_from_checkpoint` reads a teacher's own
training checkpoint; :meth:`TeacherPolicy.construction_config` and
:func:`build_teacher` let a student checkpoint rebuild its teachers without the
original files.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn

from holosoma.config_types.algo import ModuleConfig

PPO_OBS_NORMALIZER_EPS = 1.0e-2
"""``EmpiricalNormalization``'s ``eps`` in holosoma PPO."""


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
    def load_ppo_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load PPO's ``actor_obs_normalizer_state_dict``."""
        if state["_mean"].shape != self.mean.shape or state["_std"].shape != self.std.shape:
            raise ValueError(
                f"Teacher obs normalizer has shape {tuple(state['_mean'].shape)}, "
                f"but the teacher reads observations of shape {tuple(self.mean.shape)}."
            )
        self.mean.copy_(state["_mean"])
        self.std.copy_(state["_std"])
        self.eps.fill_(PPO_OBS_NORMALIZER_EPS)


def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
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


class TeacherPolicy(nn.Module):
    """A frozen policy that labels DAgger targets: ``forward(teacher_obs) -> actions``.

    Subclasses register a ``TYPE`` in :data:`TEACHER_POLICY_TYPES`, describe how to
    rebuild themselves with :meth:`construction_config`, and accept that config back
    in :meth:`from_construction_config`.
    """

    TYPE: ClassVar[str]

    def __init__(self, num_obs: int, num_actions: int) -> None:
        super().__init__()
        self.num_obs = num_obs
        self.num_actions = num_actions

    def construction_config(self) -> dict[str, Any]:
        """Serializable description that :func:`build_teacher` rebuilds this teacher from."""
        raise NotImplementedError

    @classmethod
    def from_construction_config(cls, config: dict[str, Any]) -> TeacherPolicy:
        raise NotImplementedError


class PPOTeacher(TeacherPolicy):
    """A holosoma PPO actor and its observation normalizer, as PPO ran them at inference."""

    TYPE = "holosoma_ppo"

    def __init__(
        self,
        actor_config: dict[str, Any],
        obs_dim_dict: dict[str, Any],
        history_length: dict[str, int],
        num_actions: int,
        normalized: bool,
    ) -> None:
        # Imported here: module_utils imports the student-teacher modules, which import this module.
        from holosoma.agents.modules.module_utils import setup_ppo_actor_module

        # PPO's actor resolves its module config in place, so it gets a fresh one.
        actor = setup_ppo_actor_module(
            obs_dim_dict=dict(obs_dim_dict),
            module_config=ModuleConfig(**actor_config),
            num_actions=num_actions,
            init_noise_std=1.0,
            device="cpu",
            history_length=dict(history_length),
        )
        if actor.actor_module.output_dim != num_actions:
            raise ValueError(f"Teacher actor outputs {actor.actor_module.output_dim} actions, expected {num_actions}.")
        super().__init__(num_obs=actor.actor_module.input_dim, num_actions=num_actions)
        self.actor = actor
        self.obs_normalizer = TeacherObsNormalizer(self.num_obs)
        self.normalized = normalized
        self._actor_config = actor_config
        self._obs_dim_dict = dict(obs_dim_dict)
        self._history_length = dict(history_length)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.act_inference({"actor_obs": self.obs_normalizer(obs)})

    def construction_config(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "actor_config": self._actor_config,
            "obs_dim_dict": self._obs_dim_dict,
            "history_length": self._history_length,
            "num_actions": self.num_actions,
            "normalized": self.normalized,
        }

    @classmethod
    def from_construction_config(cls, config: dict[str, Any]) -> PPOTeacher:
        return cls(
            actor_config=config["actor_config"],
            obs_dim_dict=config["obs_dim_dict"],
            history_length=config["history_length"],
            num_actions=config["num_actions"],
            normalized=config["normalized"],
        )

    @staticmethod
    def matches(checkpoint: dict[str, Any]) -> bool:
        """Whether ``checkpoint`` is a holosoma PPO training checkpoint."""
        return "actor_model_state_dict" in checkpoint and "experiment_config" in checkpoint

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any], num_obs: int, num_actions: int) -> PPOTeacher:
        """Rebuild the actor a PPO checkpoint was saved from and load its weights strictly.

        The actor is built from the checkpoint's own ``experiment_config``. Checkpoints that
        record ``obs_dim_dict`` give the actor its exact input layout; older ones are accepted
        only for actors that read a single observation group, which then spans all ``num_obs``
        teacher observations.
        """
        algo_config = checkpoint["experiment_config"]["algo"]["config"]
        actor_config = algo_config["module_dict"]["actor"]
        obs_dim_dict = checkpoint.get("obs_dim_dict")
        history_length = checkpoint.get("history_length_dict")
        if obs_dim_dict is None:
            groups = list(actor_config["input_dim"])
            if len(groups) != 1:
                raise ValueError(
                    f"The teacher actor reads observation groups {groups}, but its checkpoint predates "
                    "recorded observation dims (obs_dim_dict), so their split of teacher_obs is unknown."
                )
            obs_dim_dict = {groups[0]: num_obs}
            obs_groups = checkpoint["experiment_config"].get("observation", {}).get("groups", {})
            history_length = {groups[0]: int(obs_groups.get(groups[0], {}).get("history_length", 1))}

        normalizer_state = checkpoint.get("actor_obs_normalizer_state_dict")
        if bool(algo_config.get("empirical_normalization")) != (normalizer_state is not None):
            raise ValueError(
                "The teacher checkpoint's empirical_normalization setting does not match whether it "
                "saved an actor observation normalizer."
            )
        teacher = cls(
            actor_config=actor_config,
            obs_dim_dict=obs_dim_dict,
            history_length=history_length or {},
            num_actions=num_actions,
            normalized=normalizer_state is not None,
        )
        if teacher.num_obs != num_obs:
            raise ValueError(f"The teacher actor reads {teacher.num_obs} observations, but teacher_obs has {num_obs}.")
        teacher.actor.load_state_dict(checkpoint["actor_model_state_dict"], strict=True)
        if normalizer_state is not None:
            teacher.obs_normalizer.load_ppo_state_dict(normalizer_state)
        return teacher


class MLPTeacher(TeacherPolicy):
    """A plain MLP teacher and its observation normalizer.

    Rebuilds the teachers of student checkpoints saved before teachers recorded their
    construction, which built every teacher as this MLP; see
    :func:`holosoma.agents.distillation.checkpoint.convert_legacy_teacher_state`.
    """

    TYPE = "mlp"

    def __init__(self, num_obs: int, hidden_dims: list[int], num_actions: int, activation: str) -> None:
        super().__init__(num_obs=num_obs, num_actions=num_actions)
        self.hidden_dims = list(hidden_dims)
        self.activation = activation
        self.mlp = build_mlp(num_obs, self.hidden_dims, num_actions, activation)
        self.obs_normalizer = TeacherObsNormalizer(num_obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(obs))

    def construction_config(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "num_obs": self.num_obs,
            "hidden_dims": self.hidden_dims,
            "num_actions": self.num_actions,
            "activation": self.activation,
        }

    @classmethod
    def from_construction_config(cls, config: dict[str, Any]) -> MLPTeacher:
        return cls(config["num_obs"], config["hidden_dims"], config["num_actions"], config["activation"])


TEACHER_POLICY_TYPES: dict[str, type[TeacherPolicy]] = {cls.TYPE: cls for cls in (PPOTeacher, MLPTeacher)}
"""Teacher types by :attr:`TeacherPolicy.TYPE`, for rebuilding from a construction config."""


def build_teacher(config: dict[str, Any]) -> TeacherPolicy:
    """Rebuild a teacher's structure from :meth:`TeacherPolicy.construction_config`; weights are not loaded.

    The global RNG is left untouched: the initial weights are about to be replaced, and loading a
    teacher should not change the random stream training continues with.
    """
    teacher_type = config.get("type")
    if teacher_type not in TEACHER_POLICY_TYPES:
        raise ValueError(f"Unknown teacher type {teacher_type!r}; known types: {sorted(TEACHER_POLICY_TYPES)}.")
    with torch.random.fork_rng(devices=[]):
        return TEACHER_POLICY_TYPES[teacher_type].from_construction_config(config)


def teacher_from_checkpoint(checkpoint: dict[str, Any], num_obs: int, num_actions: int) -> TeacherPolicy:
    """Rebuild the teacher a training checkpoint was saved from, weights loaded strictly.

    Like :func:`build_teacher`, this leaves the global RNG untouched.
    """
    if PPOTeacher.matches(checkpoint):
        with torch.random.fork_rng(devices=[]):
            return PPOTeacher.from_checkpoint(checkpoint, num_obs=num_obs, num_actions=num_actions)
    raise ValueError(
        "Unsupported teacher checkpoint: expected a holosoma PPO checkpoint with actor_model_state_dict "
        f"and experiment_config. Got keys {sorted(checkpoint)[:10]}."
    )
