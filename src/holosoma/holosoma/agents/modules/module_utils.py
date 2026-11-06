from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from holosoma.agents.modules.ppo_modules import PPOActor, PPOActorEncoder, PPOCritic, PPOCriticEncoder
from holosoma.agents.modules.student_teacher_modules import (
    DepthStudentTeacher,
    DepthStudentTeacherCritic,
    StudentTeacher,
    StudentTeacherCritic,
)
from holosoma.config_types.algo import ModuleConfig, StudentTeacherModuleConfig
from holosoma.utils.helpers import get_class


def setup_ppo_actor_module(
    obs_dim_dict: dict[str, Any],
    module_config: ModuleConfig,
    num_actions: int,
    init_noise_std: float,
    device: torch.device | str,
    history_length: dict[str, int],
) -> PPOActor | PPOActorEncoder:
    module_type = module_config.type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return cast(
            "PPOActorEncoder",
            PPOActorEncoder(
                obs_dim_dict=obs_dim_dict,
                module_config_dict=module_config,
                num_actions=num_actions,
                init_noise_std=init_noise_std,
                history_length=history_length,
            ).to(device),
        )
    if module_type == "MLP":
        return cast(
            "PPOActor",
            PPOActor(
                obs_dim_dict=obs_dim_dict,
                module_config_dict=module_config,
                num_actions=num_actions,
                init_noise_std=init_noise_std,
                history_length=history_length,
            ).to(device),
        )

    raise ValueError(f"Invalid actor type: {module_type}")


def setup_ppo_critic_module(
    obs_dim_dict: dict[str, Any],
    module_config: ModuleConfig,
    device: torch.device | str,
    history_length: dict[str, int],
) -> PPOCritic | PPOCriticEncoder:
    module_type = module_config.type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return cast(
            "PPOCriticEncoder",
            PPOCriticEncoder(
                obs_dim_dict=obs_dim_dict,
                module_config_dict=module_config,
                history_length=history_length,
            ).to(device),
        )
    if module_type == "MLP":
        return cast(
            "PPOCritic",
            PPOCritic(
                obs_dim_dict=obs_dim_dict,
                module_config_dict=module_config,
                history_length=history_length,
            ).to(device),
        )
    raise ValueError(f"Invalid critic type: {module_type}")


def _setup_depth_backbone(module_config: StudentTeacherModuleConfig) -> nn.Module | None:
    """Instantiate ``module_config.depth_backbone`` as ``cls(depth_output_dim)``, if set."""
    if module_config.depth_backbone is None:
        return None
    return get_class(module_config.depth_backbone)(module_config.depth_output_dim)


def setup_student_teacher_module(
    num_actor_obs: int,
    num_teacher_obs: int,
    num_critic_obs: int,
    num_actions: int,
    module_config: StudentTeacherModuleConfig,
    device: torch.device | str,
    init_noise_std: float,
    num_teachers: int = 1,
    module_cls: type | None = None,
) -> StudentTeacherCritic | DepthStudentTeacherCritic:
    """Build the student/teacher/critic module used by :class:`DistillationPPO`.

    Builds a :class:`DepthStudentTeacherCritic` when ``module_config.depth_backbone``
    is set, and a proprioception-only :class:`StudentTeacherCritic` otherwise.
    ``num_teachers`` sets the number of teacher slots, filled when teachers are loaded.

    ``module_cls`` lets an application substitute a subclass -- e.g. one that
    derives the per-env teacher index from the observation rather than taking it
    as an argument -- without restating this construction.
    """
    depth_backbone = _setup_depth_backbone(module_config)
    kwargs: dict[str, Any] = {
        "num_actor_obs": num_actor_obs,
        "num_teacher_obs": num_teacher_obs,
        "num_critic_obs": num_critic_obs,
        "num_actions": num_actions,
        "student_hidden_dims": list(module_config.student_hidden_dims),
        "critic_hidden_dims": list(module_config.critic_hidden_dims),
        "activation": module_config.activation,
        "init_noise_std": init_noise_std,
        "num_teachers": num_teachers,
    }
    if depth_backbone is None:
        cls = module_cls if module_cls is not None else StudentTeacherCritic
    else:
        cls = module_cls if module_cls is not None else DepthStudentTeacherCritic
        kwargs.update(depth_backbone=depth_backbone, depth_output_dim=module_config.depth_output_dim)
    return cast("StudentTeacherCritic | DepthStudentTeacherCritic", cls(**kwargs).to(device))


def setup_depth_student_teacher_module(
    num_actor_obs: int,
    num_teacher_obs: int,
    num_actions: int,
    module_config: StudentTeacherModuleConfig,
    device: torch.device | str,
    init_noise_std: float,
    num_teachers: int = 1,
    module_cls: type | None = None,
) -> StudentTeacher | DepthStudentTeacher:
    """Build the student/teacher module (no critic) used by :class:`Distillation`.

    Builds a :class:`DepthStudentTeacher` when ``module_config.depth_backbone`` is
    set, and a proprioception-only :class:`StudentTeacher` otherwise.
    ``critic_hidden_dims`` on the config is ignored. ``module_cls`` substitutes a
    drop-in subclass, as in :func:`setup_student_teacher_module`.
    """
    depth_backbone = _setup_depth_backbone(module_config)
    kwargs: dict[str, Any] = {
        "num_actor_obs": num_actor_obs,
        "num_teacher_obs": num_teacher_obs,
        "num_actions": num_actions,
        "student_hidden_dims": list(module_config.student_hidden_dims),
        "activation": module_config.activation,
        "init_noise_std": init_noise_std,
        "num_teachers": num_teachers,
    }
    if depth_backbone is None:
        cls = module_cls if module_cls is not None else StudentTeacher
    else:
        cls = module_cls if module_cls is not None else DepthStudentTeacher
        kwargs.update(depth_backbone=depth_backbone, depth_output_dim=module_config.depth_output_dim)
    return cast("StudentTeacher | DepthStudentTeacher", cls(**kwargs).to(device))
