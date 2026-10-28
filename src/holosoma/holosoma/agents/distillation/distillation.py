"""Pure-DAgger (behavior-cloning) distillation algorithm.

Parallels :class:`~holosoma.agents.distillation_ppo.distillation_ppo.DistillationPPO`
but strips away every PPO-only concern: no critic, no value loss, no surrogate
/ advantage computation, no action-noise std param group. The loss is just
``loss_fn(student_mean, teacher_action)`` and the optimizer is Adam over the
student (and its depth backbone, if it has one).

This mirrors an internal FAR ``DepthDistillation`` and exists as a first-class
algorithm so a pure-DAgger preset doesn't have to pretend-via-huge-schedule
inside ``DistillationPPO``. That upstream is itself built on rsl_rl's
:class:`Distillation` (``rsl_rl/algorithms/distillation.py``, BSD-3-Clause; see
``THIRD_PARTY_LICENSES``), which is where the behavior-cloning-against-teacher-
actions formulation and the ``"behavior"`` loss key come from. Every reference to
"the reference" below means that internal FAR implementation.
"""

from __future__ import annotations

from typing import Any, TypedDict, cast

import torch
from torch import nn
from typing_extensions import NotRequired

from holosoma.agents.distillation.base import DistillationBase
from holosoma.agents.distillation.rollout_storage import RolloutStorageDagger
from holosoma.agents.modules.module_utils import setup_depth_student_teacher_module
from holosoma.agents.modules.student_teacher_modules import StudentTeacher
from holosoma.config_types.algo import DistillationConfig


class DaggerMinibatch(TypedDict):
    actor_obs: torch.Tensor
    teacher_obs: torch.Tensor
    depth_obs: NotRequired[torch.Tensor]
    actions: torch.Tensor
    teacher_actions: torch.Tensor
    teacher_valid: torch.Tensor
    dones: torch.Tensor


class Distillation(DistillationBase):
    """Pure-DAgger trainer for a student/teacher module.

    Same public surface as :class:`DistillationPPO` (``setup``/``learn``/
    ``save``/``load``/``load_teacher``/``get_inference_policy``/
    ``evaluate_policy``) so ``train_agent.py`` can dispatch to either without
    branching. There is no critic and no PPO loss.
    """

    config: DistillationConfig
    policy: StudentTeacher

    def _build_policy(self) -> None:
        self.policy = setup_depth_student_teacher_module(
            num_actor_obs=self.actor_obs_dim,
            num_teacher_obs=self.teacher_obs_dim,
            num_actions=self.num_act,
            module_config=self.config.module,
            device=self.device,
            init_noise_std=self.config.init_noise_std,
            num_teachers=self.num_teachers,
        )

    def _build_optimizer(self) -> None:
        """Single Adam over the student MLP and its depth backbone. Teachers are
        frozen; ``self.policy.std`` is untrained in pure DAgger so it is excluded."""
        trainable_params = list(self.policy.student.parameters())
        if self.uses_depth:
            trainable_params += list(self.policy.depth_backbone.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.config.learning_rate)
        self.learning_rate = self.config.learning_rate

    def _build_storage(self) -> None:
        self.storage = RolloutStorageDagger(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.config.num_steps_per_env,
            actor_obs_dim=self.actor_obs_dim,
            teacher_obs_dim=self.teacher_obs_dim,
            depth_shape=self.depth_shape,
            num_actions=self.num_act,
            device=self.device,
        )

    # ------------------------------------------------------------- rollout step

    def _rollout_step(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                obs = self._extract_obs(obs_dict)

                # Student samples an action; teacher provides the DAgger target.
                actions = self.policy.act(obs)
                teacher = self.policy.teacher_act(obs["teacher_obs"])
                teacher_actions = teacher.actions.detach()

                # Warm-up: follow teacher actions so the student sees expert
                # trajectories before its policy is competent. Still record the
                # student's sampled action so the update step's distribution
                # state is consistent with what would have been taken.
                step_actions = self._step_actions(actions, teacher_actions, teacher.valid, self._warmup_active)
                next_obs, rewards, dones, infos = self.env.step({"actions": step_actions})
                next_obs = self._obs_to_device(next_obs)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.storage.add(
                    **obs,
                    actions=actions,
                    teacher_actions=teacher_actions,
                    teacher_valid=teacher.valid.view(-1, 1),
                    dones=dones.view(-1, 1),
                )
                self.policy.reset(dones)

                if self.log_dir is not None:
                    self.logging_helper.update_episode_stats(rewards, dones, infos)

                obs_dict = next_obs
        return obs_dict

    # ------------------------------------------------------------ training step

    def _training_step(self) -> dict[str, float]:
        generator = self.storage.mini_batch_generator(self.config.num_mini_batches, self.config.num_learning_epochs)
        loss_accum = 0.0
        num_updates = 0

        mini_batch: DaggerMinibatch
        for raw_mini_batch in generator:
            mini_batch = cast("DaggerMinibatch", raw_mini_batch)
            actor_obs = mini_batch["actor_obs"]
            depth_obs = mini_batch.get("depth_obs")
            teacher_actions = mini_batch["teacher_actions"]

            # Deterministic student forward (distribution mean); this is what
            # the reference implementation uses for behavior cloning, not a
            # sampled action. Rows without a teacher are masked out.
            student_mean = self.policy.student(self.policy._student_input(actor_obs, depth_obs))
            loss = self._dagger_loss(student_mean, teacher_actions, mini_batch["teacher_valid"])

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self._reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            loss_accum += loss.detach().item()
            num_updates += 1

        loss_dict: dict[str, float] = {"warmup_active": float(self._warmup_active)}
        if num_updates > 0:
            # Logged as "behavior" to match the reference DepthDistillation
            # loss-dict key, keeping wandb plot axes consistent across the two
            # codebases.
            loss_dict["behavior"] = loss_accum / num_updates

        self.storage.clear()
        return loss_dict

    # ------------------------------------------------------------------- logging

    def _post_epoch_logging(self, it: int, loss_dict: dict[str, float]) -> None:
        extra_log_dicts = {
            "Policy": {
                "mean_noise_std": self.policy.std.mean().item(),
            },
            "LR": {
                "student_backbone_lr": self.learning_rate,
            },
        }
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)
