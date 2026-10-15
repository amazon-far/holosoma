"""PPO + DAgger (distillation) combined-loss algorithm.

Holosoma-native port of an internal FAR ``DepthDistillationPPO``, which is in
turn built on rsl_rl's :class:`Distillation` algorithm
(``rsl_rl/algorithms/distillation.py``, BSD-3-Clause; see
``THIRD_PARTY_LICENSES``). The lineage is visible in the shared design: behavior
cloning against the teacher's *actions* rather than against a latent, and the
``"behavior"`` loss-dict key reproduced below to keep the wandb schema aligned.

Built around the
:class:`~holosoma.agents.modules.student_teacher_modules.StudentTeacherCritic`
policy, or its depth-backbone variant. Interface-compatible with
:class:`~holosoma.agents.ppo.ppo.PPO` so it can be dispatched via
``tyro_config.algo._target_``.

The training loop walks a schedule that interpolates between pure DAgger (PPO
coefficient 0) and nearly-pure PPO (coefficient plateau ``max_ppo_coef``, 0.9 by
default). Under the hood we build a single :class:`torch.optim.AdamW` with parameter groups for the
student, critic, depth backbone (if any), and action ``std``. This lets the depth
encoder use its own regularization and lets the ``std`` learning rate be zeroed
while the DAgger phase is active, preventing the action-noise scale from drifting
before PPO takes over.
"""

from __future__ import annotations

import os
from typing import Any, TypedDict, cast

import torch
from torch import nn
from typing_extensions import NotRequired

from holosoma.agents.distillation.base import CRITIC_GROUP, DistillationBase
from holosoma.agents.distillation_ppo.rollout_storage import RolloutStorageDistillation
from holosoma.agents.modules.advantages import compute_gae_returns, normalize_advantages
from holosoma.agents.modules.module_utils import setup_student_teacher_module
from holosoma.agents.modules.student_teacher_modules import StudentTeacherCritic
from holosoma.config_types.algo import DistillationPPOConfig
from holosoma.envs.base_task.base_task import BaseTask


class DistillationMinibatch(TypedDict):
    """Structure of a minibatch yielded by
    :class:`RolloutStorageDistillation.mini_batch_generator`. Matches
    :class:`holosoma.agents.ppo.ppo.Minibatch` plus the three distillation
    fields.
    """

    actor_obs: torch.Tensor
    teacher_obs: torch.Tensor
    critic_obs: torch.Tensor
    depth_obs: NotRequired[torch.Tensor]
    actions: torch.Tensor
    teacher_actions: torch.Tensor
    teacher_valid: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    actions_log_prob: torch.Tensor
    action_mean: torch.Tensor
    action_sigma: torch.Tensor
    ppo_gate: torch.Tensor


class DistillationPPO(DistillationBase):
    """PPO + DAgger combined-loss trainer for a student/teacher/critic module.

    Mirrors :class:`~holosoma.agents.ppo.ppo.PPO`'s public surface
    (``setup``/``learn``/``save``/``load``/``get_inference_policy``/
    ``evaluate_policy``) so the training entrypoint is agnostic to which
    algorithm is being instantiated.

    Multi-GPU training mirrors holosoma's PPO and the reference
    DepthDistillationPPO. Under torchrun, each rank runs its own env shard; the
    optimizer step is kept in sync by (a) broadcasting initial weights from rank
    0, (b) all-reducing gradients before optimizer.step, (c) all-reducing the KL
    estimate before the adaptive-LR schedule, (d) broadcasting the LR decision
    from rank 0 so every rank advances its optimizer in lockstep, and (e)
    synchronizing curriculum state before every rollout.
    """

    config: DistillationPPOConfig
    policy: StudentTeacherCritic

    uses_critic = True

    def __init__(
        self,
        env: BaseTask,
        config: DistillationPPOConfig,
        log_dir: str | os.PathLike[str],
        device: str = "cpu",
        multi_gpu_cfg: dict[str, int] | None = None,
    ):
        super().__init__(env, config, log_dir, device, multi_gpu_cfg)
        # PPO/DAgger schedule state -- filled in by adjust_ppo_dagger_coeff.
        self.ppo_coef = 0.0

    def _build_policy(self) -> None:
        self.policy = setup_student_teacher_module(
            num_actor_obs=self.actor_obs_dim,
            num_teacher_obs=self.teacher_obs_dim,
            num_critic_obs=self.critic_obs_dim,
            num_actions=self.num_act,
            module_config=self.config.module,
            device=self.device,
            init_noise_std=self.config.init_noise_std,
            num_teachers=self.num_teachers,
        )

    def _build_optimizer(self) -> None:
        """Build the student, critic, depth-backbone, and std AdamW groups.

        The depth backbone is isolated so it can use the stronger regularization
        from the reference distillation recipe without also decaying the
        student MLP. The ``std`` group remains separate so its learning rate can
        be toggled while PPO warms up.
        """
        lr, weight_decay = self.config.learning_rate, self.config.weight_decay
        param_groups: list[dict[str, Any]] = [
            {"params": list(self.policy.student.parameters()), "lr": lr, "weight_decay": weight_decay},
            {"params": list(self.policy.critic.parameters()), "lr": lr, "weight_decay": weight_decay},
        ]
        if self.uses_depth:
            param_groups.append(
                {
                    "params": list(self.policy.depth_backbone.parameters()),
                    "lr": lr,
                    "weight_decay": self.config.depth_weight_decay,
                }
            )
        param_groups.append({"params": [self.policy.std], "lr": lr, "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(param_groups)
        # _set_std_lr_from_ppo_coef / _update_learning_rate address the std group by index.
        self._STD_GROUP_IDX = len(param_groups) - 1

    def _build_storage(self) -> None:
        self.storage = RolloutStorageDistillation(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.config.num_steps_per_env,
            actor_obs_dim=self.actor_obs_dim,
            teacher_obs_dim=self.teacher_obs_dim,
            critic_obs_dim=self.critic_obs_dim,
            depth_shape=self.depth_shape,
            num_actions=self.num_act,
            device=self.device,
        )

    def _on_iteration_start(self, iteration: int) -> None:
        # Advance the DAgger -> PPO schedule and gate the std-param-group LR.
        self.adjust_ppo_dagger_coeff(iteration)
        self._set_std_lr_from_ppo_coef()

    # ---------------------------------------------------------- ppo/dagger mix

    def adjust_ppo_dagger_coeff(self, iteration: int) -> None:
        """Ramp the PPO coefficient from 0 to ``max_ppo_coef`` between
        ``ppo_start_epoch`` and ``dagger_end_epoch``."""
        start = self.config.ppo_start_epoch
        end = self.config.dagger_end_epoch
        if iteration < start:
            self.ppo_coef = 0.0
        elif iteration >= end:
            self.ppo_coef = self.config.max_ppo_coef
        else:
            self.ppo_coef = min((iteration - start) / max(end - start, 1), self.config.max_ppo_coef)

    def _set_std_lr_from_ppo_coef(self) -> None:
        """Zero the std param group's LR while DAgger dominates the loss.

        When PPO is active, the std group follows the *current* decayed
        ``self.learning_rate`` (mirrors the reference implementation).
        Using ``self.config.learning_rate`` instead would insulate the std
        parameter from the KL-driven adaptive LR decay and let it drift to
        a higher steady-state value than the reference's.
        """
        std_lr = 0.0 if self.ppo_coef <= self.config.ppo_active_threshold else self.learning_rate
        self.optimizer.param_groups[self._STD_GROUP_IDX]["lr"] = std_lr

    # ------------------------------------------------------------- rollout step

    def _rollout_step(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                obs = self._extract_obs(obs_dict)

                # Sample student action and compute value / teacher label.
                actions = self.policy.act(obs)
                values = self.policy.evaluate(obs).detach()
                teacher = self.policy.teacher_act(obs["teacher_obs"])
                teacher_actions = teacher.actions.detach()

                actions_log_prob = self.policy.get_actions_log_prob(actions).detach().unsqueeze(1)
                action_mean = self.policy.action_mean.detach()
                action_sigma = self.policy.action_std.detach()
                # Read the gate before ``env.step``: it describes the state the
                # action was taken in, and stepping advances the motion phase.
                ppo_gate = self._get_ppo_gate().detach()
                # Let an application accumulate per-motion-frame statistics while
                # the frame index is still live. Done here rather than in the
                # update loop because the rollout storage carries no frame index,
                # and a one-step lag is irrelevant against the EMA's time constant.
                self._update_ppo_gate_stats(action_mean, teacher_actions, teacher.valid)

                # During warmup the teacher drives the environment. Student
                # distribution fields are still stored because the common
                # rollout storage schema requires them; updates are DAgger-only.
                step_actions = self._step_actions(actions, teacher_actions, teacher.valid, self._warmup_active)
                next_obs, rewards, dones, infos = self.env.step({"actions": step_actions})
                next_obs = self._obs_to_device(next_obs)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # Bootstrap value at timeouts (matches PPO exactly).
                final_rewards = torch.zeros_like(rewards)
                if infos.get("time_outs") is not None and infos["time_outs"].any():
                    final_critic_obs = infos["final_observations"].get(CRITIC_GROUP)
                    if final_critic_obs is not None:
                        final_values = self.policy.evaluate({"critic_obs": final_critic_obs}).detach()
                        final_rewards += self.config.gamma * torch.squeeze(
                            final_values * infos["time_outs"].unsqueeze(1).to(self.device), 1
                        )

                self.storage.add(
                    **obs,
                    actions=actions,
                    teacher_actions=teacher_actions,
                    teacher_valid=teacher.valid.view(-1, 1),
                    values=values,
                    actions_log_prob=actions_log_prob,
                    action_mean=action_mean,
                    action_sigma=action_sigma,
                    ppo_gate=ppo_gate.view(-1, 1),
                    rewards=(rewards + final_rewards).view(-1, 1),
                    dones=dones.view(-1, 1),
                )

                self.policy.reset(dones)

                if self.log_dir is not None:
                    self.logging_helper.update_episode_stats(rewards, dones, infos)

                obs_dict = next_obs

            # GAE / returns.
            last_values = self.policy.evaluate(self._extract_obs(obs_dict)).detach().to(self.device)
            returns, advantages = self._compute_returns_and_advantages(
                last_values,
                self.storage["values"].to(self.device),
                self.storage["dones"].to(self.device),
                self.storage["rewards"].to(self.device),
            )
            self.storage["returns"] = returns
            self.storage["advantages"] = advantages

        return obs_dict

    def _compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        returns = compute_gae_returns(last_values, values, dones, rewards, self.config.gamma, self.config.lam)
        advantages = normalize_advantages(returns - values, self.gpu_world_size if self.is_multi_gpu else None)
        return returns, advantages

    # ------------------------------------------------------------ training step

    def _get_ppo_gate(self) -> torch.Tensor:
        """Per-env PPO gate in ``[0, 1]``, shape ``(num_envs, 1)``.

        Scales how much of the PPO/DAgger split applies to *this* transition:
        the effective coefficient becomes ``gate * ppo_coef``, so a gate of 1
        reproduces the scalar behaviour exactly and a gate of 0 makes the
        transition pure DAgger.

        Defaults to all-ones, i.e. inert. Override to concentrate the reward
        pressure on the part of a clip that needs it — reward and imitation do
        not cost the same everywhere, so a skill that only imitation can't hold
        (a climb, say) can keep its PPO term while phases that imitation handles
        better (flat locomotion) run pure DAgger.
        """
        return torch.ones(self.env.num_envs, 1, device=self.device)

    def _update_ppo_gate_stats(
        self, student_mean: torch.Tensor, teacher_actions: torch.Tensor, teacher_valid: torch.Tensor
    ) -> None:
        """Hook for accumulating per-motion-frame statistics during rollout.

        ``teacher_valid`` (``[N]`` bool) marks the envs a teacher produced a label for.

        No-op by default. Override alongside :meth:`_get_ppo_gate` to make the
        gate adapt to where imitation is actually falling short, rather than to a
        hand-specified window.
        """
        return

    def _dagger_loss_per_sample(
        self, student_mean: torch.Tensor, teacher_actions: torch.Tensor, teacher_valid: torch.Tensor
    ) -> torch.Tensor:
        """Behaviour-cloning loss per sample, shape ``(batch,)``; zero where ``teacher_valid`` is false.

        This is the extension point the combined loss uses; :meth:`_dagger_loss`
        is its batch mean and exists for logging and for callers that want a
        scalar.
        """
        target = self._dagger_target(student_mean, teacher_actions, teacher_valid)
        return self.distill_loss_fn(student_mean, target, reduction="none").mean(dim=-1)

    def _dagger_loss(
        self, student_mean: torch.Tensor, teacher_actions: torch.Tensor, teacher_valid: torch.Tensor
    ) -> torch.Tensor:
        """Batch-mean behaviour-cloning loss. See :meth:`_dagger_loss_per_sample`."""
        return self._dagger_loss_per_sample(student_mean, teacher_actions, teacher_valid).mean()

    def _training_step(self) -> dict[str, float]:
        generator = self.storage.mini_batch_generator(self.config.num_mini_batches, self.config.num_learning_epochs)
        loss_accum: dict[str, torch.Tensor] = {}
        num_updates = 0

        mini_batch: DistillationMinibatch
        for raw_mini_batch in generator:
            mini_batch = cast("DistillationMinibatch", raw_mini_batch)
            step_losses = self._update_step(mini_batch)
            num_updates += 1
            for k, v in step_losses.items():
                val = v.detach() if torch.is_tensor(v) else torch.tensor(v, device=self.device)
                if k not in loss_accum:
                    loss_accum[k] = val.clone()
                else:
                    loss_accum[k] += val

        loss_dict: dict[str, float] = {
            "ppo_coef": float(self.ppo_coef),
            "warmup_active": float(self._warmup_active),
            # Mirror the rsl_rl runner's `Loss/learning_rate` (emitted from
            # its on_policy_runner) so the wandb schema matches the reference
            # implementation.
            "learning_rate": float(self.learning_rate),
        }
        if num_updates > 0:
            for k, tensor in loss_accum.items():
                loss_dict[k] = (tensor / num_updates).item()

        self.storage.clear()
        return loss_dict

    def _dagger_only_update(
        self,
        actor_obs: torch.Tensor,
        depth_obs: torch.Tensor | None,
        teacher_actions: torch.Tensor,
        teacher_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        student_mean = self.policy.student(self.policy._student_input(actor_obs, depth_obs))
        dagger_loss = self._dagger_loss(student_mean, teacher_actions, teacher_valid)
        total_loss = self.config.dagger_loss_coef * dagger_loss
        self._optimizer_step(total_loss)
        zero = dagger_loss.new_zeros(())
        return {
            "total_loss": total_loss.detach(),
            "surrogate_loss": zero,
            "value_loss": zero,
            "behavior": dagger_loss.detach(),
            "entropy_mean": zero,
            "mean_kl": zero,
        }

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        """Apply one synchronized, clipped optimizer update."""
        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
            self._reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.policy.project_action_std()

    def _update_step(self, mini_batch: DistillationMinibatch) -> dict[str, torch.Tensor]:
        actor_obs = mini_batch["actor_obs"]
        depth_obs = mini_batch.get("depth_obs")
        critic_obs = mini_batch["critic_obs"]
        actions = mini_batch["actions"]
        teacher_actions = mini_batch["teacher_actions"]
        teacher_valid = mini_batch["teacher_valid"]
        if self._warmup_active:
            return self._dagger_only_update(actor_obs, depth_obs, teacher_actions, teacher_valid)

        target_values = mini_batch["values"]
        returns = mini_batch["returns"]
        advantages = mini_batch["advantages"]
        old_log_prob = mini_batch["actions_log_prob"]
        old_mu = mini_batch["action_mean"]
        old_sigma = mini_batch["action_sigma"]

        # Re-run the student distribution for this minibatch.
        self.policy.act({"actor_obs": actor_obs, "depth_obs": depth_obs})
        log_prob = self.policy.get_actions_log_prob(actions)
        mu = self.policy.action_mean
        sigma = self.policy.action_std
        entropy = self.policy.entropy

        student_mean = mu  # distribution mean is already the student MLP output

        # Critic forward.
        value = self.policy.evaluate({"critic_obs": critic_obs})

        # Adaptive LR update from approximate KL -- skipped during pure-DAgger phase.
        if (
            self.config.desired_kl is not None  # type: ignore[redundant-expr]
            and self.config.schedule == "adaptive"
            and self.ppo_coef > self.config.ppo_active_threshold
        ):
            # Weight the KL by the gate: with per-sample gating the LR schedule
            # must be driven by the samples PPO actually trains on, otherwise a
            # mostly-ungated batch reports a KL for transitions the surrogate
            # never touched and the schedule chases the wrong signal.
            kl_mean = self._compute_kl_div(
                old_mu, old_sigma, mu, sigma, weights=torch.squeeze(mini_batch["ppo_gate"], -1)
            )
            self._update_learning_rate(kl_mean)
        else:
            kl_mean = torch.tensor(0.0, device=self.device)

        # Surrogate loss.
        ratio = torch.exp(log_prob - torch.squeeze(old_log_prob))
        surrogate = -torch.squeeze(advantages) * ratio
        surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
            ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
        )
        surrogate_per_sample = torch.max(surrogate, surrogate_clipped)
        surrogate_loss = surrogate_per_sample.mean()

        # Clipped value loss.
        value_clipped = target_values + (value - target_values).clamp(-self.config.clip_param, self.config.clip_param)
        value_losses = (value - returns).pow(2)
        value_losses_clipped = (value_clipped - returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        entropy_mean = entropy.mean()

        dagger_per_sample = self._dagger_loss_per_sample(student_mean, teacher_actions, teacher_valid)

        # Combined loss. Matches the reference DepthDistillationPPO formula:
        # value loss is **always active**
        # (ungated by ppo_coef) so the critic and the shared depth backbone
        # keep training from iteration 0; only the PPO surrogate + entropy
        # term is gated by ppo_coef. The DAgger term is independently scaled
        # by (1 - ppo_coef).
        #
        # The split is applied **per sample**: the effective coefficient is
        # ``gate * ppo_coef``, so a transition with gate 0 is pure DAgger and one
        # with gate 1 behaves exactly as the scalar formula did. With the default
        # all-ones gate this reduces to the previous expression identically.
        #
        # Deliberately a plain batch mean rather than normalising each term by
        # its own weight mass. A gated sample then keeps *exactly* the PPO/DAgger
        # balance it had under the scalar formula, which is what makes a gated run
        # comparable to an ungated one. The total PPO gradient does shrink with
        # the gated fraction, but Adam is invariant to a uniform rescaling of the
        # gradient, so this costs far less than it looks; normalising instead
        # would silently redefine what ``ppo_coef`` means.
        gate = torch.squeeze(mini_batch["ppo_gate"], -1)
        coef = gate * self.ppo_coef
        ppo_loss = (
            self.config.value_loss_coef * value_loss
            + (coef * (surrogate_per_sample - self.config.entropy_coef * entropy)).mean()
        )
        total_loss = ppo_loss + ((1.0 - coef) * self.config.dagger_loss_coef * dagger_per_sample).mean()

        self._optimizer_step(total_loss)

        return {
            "total_loss": total_loss.detach(),
            "surrogate_loss": surrogate_loss.detach(),
            "value_loss": value_loss.detach(),
            # Logged as "behavior" to match the reference distillation loss
            # dict key. The reference bakes (1 - ppo_coef) into the tensor
            # before .item(); we mirror that here so the plot scales line up
            # across the two codebases. The actual BC gradient magnitude is
            # unchanged — the (1 - ppo_coef) factor is already applied to
            # `total_loss` above. With per-sample gating that weight varies, so
            # use its batch mean rather than the scalar ``1 - ppo_coef`` (which
            # would misreport whenever the gate is not all-ones).
            "behavior": ((1.0 - coef) * dagger_per_sample).mean().detach(),
            "entropy_mean": entropy_mean.detach(),
            "mean_kl": kl_mean.detach(),
            # Fraction of the batch the PPO term is active on. Flat at 1.0 unless
            # ``_get_ppo_gate`` is overridden -- worth logging because a gate
            # that silently reads 0 turns the run into pure DAgger.
            "ppo_gate_mean": gate.mean().detach(),
        }

    def _compute_kl_div(
        self,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Closed-form KL for diagonal Gaussians; matches the reference
        # implementation's adaptive-LR formula.
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma / old_sigma + 1.0e-5)
                + (torch.square(old_sigma) + torch.square(old_mu - mu)) / (2.0 * torch.square(sigma))
                - 0.5,
                dim=-1,
            )
            if weights is None:
                kl_mean = torch.mean(kl)
                # Average the KL estimate across ranks so every rank makes the
                # same adaptive-LR decision. Ranks hold equal batches, so this is
                # the global mean. Matches holosoma PPO and the reference
                # implementation.
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= self.gpu_world_size
                return kl_mean
            # Weighted mean, not a weighted sum: the threshold comparison in
            # ``_update_learning_rate`` is against an absolute KL, so the scale
            # must stay independent of how much of the batch is gated. Gates give
            # ranks different total weights, so reduce the weighted sum and the
            # weight separately and divide once.
            totals = torch.stack(((kl * weights).sum(), weights.sum()))
            if self.is_multi_gpu:
                torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
            return totals[0] / totals[1].clamp(min=1e-6)

    def _update_learning_rate(self, kl_mean: torch.Tensor) -> None:
        # Only rank 0 computes the LR update; broadcast the decision so
        # every rank's optimizer ticks in lockstep. Without this, tiny
        # float-comparison mismatches can push ranks onto different LRs
        # and the all-reduced grads become inconsistent.
        if not self.is_multi_gpu or self.gpu_global_rank == 0:
            if kl_mean > self.config.desired_kl * 2.0:
                self.learning_rate = max(self.config.min_learning_rate, self.learning_rate / 1.5)
            elif kl_mean < self.config.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(self.config.max_learning_rate, self.learning_rate * 1.5)

        if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()

        # Push the decayed LR to all groups, including the std group when it
        # is active (ppo_coef > ppo_active_threshold). Matches the reference
        # implementation, which updates every param group.
        for idx, group in enumerate(self.optimizer.param_groups):
            if idx == self._STD_GROUP_IDX and self.ppo_coef <= self.config.ppo_active_threshold:
                # Still gated to 0 while DAgger dominates; don't overwrite.
                continue
            group["lr"] = self.learning_rate

    # ------------------------------------------------------------------- logging

    def _post_epoch_logging(self, it: int, loss_dict: dict[str, float]) -> None:
        extra_log_dicts = {
            "Policy": {
                "mean_noise_std": self.policy.std.mean().item(),
                "ppo_coef": self.ppo_coef,
            },
            "LR": {
                "student_critic_lr": self.learning_rate,
                "std_lr": self.optimizer.param_groups[self._STD_GROUP_IDX]["lr"],
            },
        }
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

    # ---------------------------------------------------------- save/load

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {"ppo_coef": self.ppo_coef}

    def _restore_extra_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        if "ppo_coef" in checkpoint:
            self.ppo_coef = float(checkpoint["ppo_coef"])
