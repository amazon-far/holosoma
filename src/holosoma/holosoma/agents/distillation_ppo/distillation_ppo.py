"""PPO + DAgger (distillation) combined-loss algorithm.

Holosoma-native port of ``DistillationPPO`` from far-tracking, built around the
:class:`~holosoma.agents.modules.student_teacher_modules.DepthStudentTeacherCritic`
policy. Interface-compatible with :class:`~holosoma.agents.ppo.ppo.PPO` so it
can be dispatched via ``tyro_config.algo._target_``.

The training loop walks a schedule that interpolates between pure DAgger (PPO
coefficient 0) and nearly-pure PPO (coefficient plateau at 0.9). Under the
hood we build a single :class:`torch.optim.AdamW` with four parameter groups:
student, critic, depth backbone, and action ``std``. This lets the depth encoder
use its own regularization and lets the ``std`` learning rate be zeroed while
the DAgger phase is active, preventing the action-noise scale from drifting
before PPO takes over.
"""

from __future__ import annotations

import itertools
import os
from typing import Any, Callable, TypedDict

import torch
import torch.nn.functional as F
from loguru import logger
from rich.console import Console
from torch import nn
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.distillation_ppo.rollout_storage import RolloutStorageDistillation
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.modules.module_utils import setup_student_teacher_module
from holosoma.agents.modules.student_teacher_modules import DepthStudentTeacherCritic
from holosoma.config_types.algo import DistillationPPOConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.helpers import instantiate

console = Console()


# Observation group name constants. Kept top-level for clarity; override by
# subclassing if a preset wires in different group names. Matches holosoma's
# ``actor_obs`` / ``critic_obs`` convention used by the teacher presets.
ACTOR_GROUP = "actor_obs"
TEACHER_GROUP = "teacher_obs"
CRITIC_GROUP = "critic_obs"
DEPTH_GROUP = "depth_camera"
DEPTH_TERM = "depth_cam"


class DistillationMinibatch(TypedDict):
    """Structure of a minibatch yielded by
    :class:`RolloutStorageDistillation.mini_batch_generator`. Matches
    :class:`holosoma.agents.ppo.ppo.Minibatch` plus the three distillation
    fields.
    """

    actor_obs: torch.Tensor
    teacher_obs: torch.Tensor
    critic_obs: torch.Tensor
    depth_obs: torch.Tensor
    actions: torch.Tensor
    teacher_actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    actions_log_prob: torch.Tensor
    action_mean: torch.Tensor
    action_sigma: torch.Tensor
    ppo_gate: torch.Tensor


class DistillationPPO(BaseAlgo):
    """PPO + DAgger combined-loss trainer for a student/teacher/critic module.

    Mirrors :class:`~holosoma.agents.ppo.ppo.PPO`'s public surface
    (``setup``/``learn``/``save``/``load``/``get_inference_policy``/
    ``evaluate_policy``) so the training entrypoint is agnostic to which
    algorithm is being instantiated.
    """

    config: DistillationPPOConfig
    policy: DepthStudentTeacherCritic

    def __init__(
        self,
        env: BaseTask,
        config: DistillationPPOConfig,
        log_dir: str | os.PathLike,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(env, config, device, multi_gpu_cfg)
        # Multi-GPU training: mirrors holosoma's PPO
        # (ppo.py:267-268 + _reduce_parameters / _synchronize_model_weights)
        # and far-tracking's DepthDistillationPPO (my_distillation.py:529-554,
        # 1087-1088, 1101-1117). Under torchrun, each rank runs its own env
        # shard; the optimizer step is kept in sync by (a) broadcasting
        # initial weights from rank 0, (b) all-reducing gradients before
        # optimizer.step, (c) all-reducing the KL estimate before the
        # adaptive-LR schedule, and (d) broadcasting the LR decision from
        # rank 0 so every rank advances its optimizer in lockstep.

        self.log_dir = str(log_dir)
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=self.config.num_steps_per_env,
            num_learning_iterations=self.config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )

        if self.config.empirical_normalization:
            # Not supported: our obs pipeline currently assumes the observation
            # manager already applies any normalization/clipping.
            raise ValueError("empirical_normalization is not supported by DistillationPPO.")

        self.current_learning_iteration = 0
        self.eval_callbacks: list[RLEvalCallback] = []

        # Number of frozen teacher MLPs to build in ``_build_policy``. Set by
        # ``train_agent.py`` from the comma count in ``training.teacher_checkpoint``
        # before ``setup()``; defaults to 1. ``teacher_obs`` must carry the
        # routing motion_idx in its leading column when this is > 1.
        self.num_teachers: int = 1

        # PPO/DAgger schedule state -- filled in by adjust_ppo_dagger_coeff.
        self.ppo_coef = 0.0
        self.learning_rate = self.config.learning_rate
        # Warm-up window toggle; flipped per iter inside ``learn()``. When
        # ``True``, rollouts step the env with teacher actions instead of the
        # student's sampled action.
        self._warmup_active = False

        # Loss function for the behavior-cloning term. Resolved eagerly so a
        # bad config surfaces during construction, not deep inside update().
        if self.config.distill_loss_type == "mse":
            self.distill_loss_fn = F.mse_loss
        elif self.config.distill_loss_type == "huber":
            self.distill_loss_fn = F.huber_loss
        else:
            raise ValueError(f"Unknown distill_loss_type: {self.config.distill_loss_type!r}")

        _ = self.env.reset_all()

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        logger.info("Setting up DistillationPPO")
        self._resolve_obs_dims()
        self._build_policy()
        # Broadcast initial weights from rank 0 so every rank starts from the
        # same random init. Must happen after the policy is built but before
        # the optimizer is built (so the optimizer sees synchronized params).
        if self.is_multi_gpu:
            self._synchronize_model_weights()
        self._build_optimizer()
        self._build_storage()

    def _resolve_obs_dims(self) -> None:
        """Pull observation dims for actor/teacher/critic/depth out of the obs manager."""
        assert self.env.observation_manager is not None
        obs_dims = self.env.observation_manager.get_obs_dims()

        for grp in (ACTOR_GROUP, TEACHER_GROUP, CRITIC_GROUP):
            if grp not in obs_dims:
                raise KeyError(f"DistillationPPO expects observation group {grp!r} to be present.")
        if DEPTH_GROUP not in obs_dims:
            raise KeyError(
                f"DistillationPPO expects observation group {DEPTH_GROUP!r} to be present for depth input."
            )

        actor_dim = obs_dims[ACTOR_GROUP]
        teacher_dim = obs_dims[TEACHER_GROUP]
        critic_dim = obs_dims[CRITIC_GROUP]
        if not (isinstance(actor_dim, int) and isinstance(teacher_dim, int) and isinstance(critic_dim, int)):
            raise TypeError(
                "actor/teacher/critic obs groups must concatenate into a flat int dimension; "
                "got non-int entry in observation_manager.get_obs_dims()."
            )

        # The depth group does NOT concatenate (images are 3D). get_obs_dims()
        # returns a dict keyed by term name in that case.
        depth_entry = obs_dims[DEPTH_GROUP]
        if isinstance(depth_entry, dict):
            if DEPTH_TERM not in depth_entry:
                raise KeyError(
                    f"depth obs group {DEPTH_GROUP!r} missing term {DEPTH_TERM!r}; found {list(depth_entry.keys())}"
                )
            # depth_entry[DEPTH_TERM] is the flat dim after reshape in manager.get_obs_dims;
            # for a genuine image shape we need to sample the tensor.
            depth_tensor = self.env.observation_manager.compute_group(DEPTH_GROUP, modify_history=False)
            if not isinstance(depth_tensor, dict):
                raise TypeError(
                    f"expected compute_group({DEPTH_GROUP!r}) to return a dict when concatenate=False."
                )
            depth_shape = tuple(depth_tensor[DEPTH_TERM].shape[1:])
        elif isinstance(depth_entry, int):
            # concatenate=True; dims are flat already but we treat it as (dim,).
            depth_shape = (depth_entry,)
        else:
            raise TypeError(f"Unexpected type for depth obs dims: {type(depth_entry)}")

        self.num_act = self.env.robot_config.actions_dim
        self.actor_obs_dim = actor_dim
        self.teacher_obs_dim = teacher_dim
        self.critic_obs_dim = critic_dim
        self.depth_shape = depth_shape

        logger.info(
            f"DistillationPPO obs dims -- actor: {actor_dim}, teacher: {teacher_dim}, "
            f"critic: {critic_dim}, depth: {depth_shape}, actions: {self.num_act}"
        )

    def _build_policy(self) -> None:
        self.policy = setup_student_teacher_module(
            num_actor_obs=self.actor_obs_dim,
            num_teacher_obs=self.teacher_obs_dim,
            num_critic_obs=self.critic_obs_dim,
            num_actions=self.num_act,
            module_config=self.config.module,
            device=self.device,
            num_teachers=self.num_teachers,
        )

    def _build_optimizer(self) -> None:
        """Build the student, critic, depth-backbone, and std AdamW groups.

        The depth backbone is isolated so it can use the stronger regularization
        from the far-tracking distillation recipe without also decaying the
        student MLP. The ``std`` group remains separate so its learning rate can
        be toggled while PPO warms up.
        """
        student_params = list(self.policy.student.parameters())
        critic_params = list(self.policy.critic.parameters())
        depth_params = list(self.policy.depth_backbone.parameters())

        self.optimizer = torch.optim.AdamW(
            [
                {"params": student_params, "lr": self.config.learning_rate, "weight_decay": self.config.weight_decay},
                {"params": critic_params, "lr": self.config.learning_rate, "weight_decay": self.config.weight_decay},
                {
                    "params": depth_params,
                    "lr": self.config.learning_rate,
                    "weight_decay": self.config.depth_weight_decay,
                },
                {"params": [self.policy.std], "lr": self.config.learning_rate, "weight_decay": 0.0},
            ]
        )
        # Indices are hardcoded; _set_std_lr / _update_learning_rate rely on them.
        self._STD_GROUP_IDX = 3

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

    # --------------------------------------------------------------- obs helpers

    def _extract_groups(self, obs_dict: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pull (actor_obs, teacher_obs, critic_obs, depth_obs) out of a
        manager-produced obs dict. Handles both concat- and non-concat depth
        groups transparently."""
        actor_obs = obs_dict[ACTOR_GROUP]
        teacher_obs = obs_dict[TEACHER_GROUP]
        critic_obs = obs_dict[CRITIC_GROUP]
        depth_entry = obs_dict[DEPTH_GROUP]
        if isinstance(depth_entry, dict):
            depth_obs = depth_entry[DEPTH_TERM]
        else:
            depth_obs = depth_entry

        # Flatten the depth obs from (N, H, W) -> (N, H*W) when the policy
        # stores a flat dim. This mirrors how the backbone expects its input;
        # DepthOnlyFCBackbone flattens internally, so passing the raw 3D is
        # also fine. We hand off the tensor in its native shape.
        return actor_obs, teacher_obs, critic_obs, depth_obs

    # --------------------------------------------------------------- train mode

    def _train_mode(self) -> None:
        self.policy.train()
        # Teacher MLP always stays in eval; StudentTeacher enforces this at
        # init but a user-triggered .train() on the composite flips every
        # submodule, so restore it explicitly.
        self.policy.teacher.eval()

    def _eval_mode(self) -> None:
        self.policy.eval()

    # -------------------------------------------------------------------- learn

    def learn(self, num_learning_iterations: int | None = None) -> None:
        self._train_mode()

        obs_dict = self.env.reset_all()
        for k in obs_dict:
            if isinstance(obs_dict[k], torch.Tensor):
                obs_dict[k] = obs_dict[k].to(self.device)
            elif isinstance(obs_dict[k], dict):
                obs_dict[k] = {kk: vv.to(self.device) for kk, vv in obs_dict[k].items()}

        total_iters = num_learning_iterations if num_learning_iterations is not None else self.config.num_learning_iterations
        start_iter = self.current_learning_iteration
        stop_iter = start_iter + total_iters

        for it in range(start_iter, stop_iter):
            self.current_learning_iteration = it
            # Warm-up window: step env with teacher action so the student sees
            # expert trajectories before its own policy is competent. Counted
            # from the start of *this* learn() call so resumes behave sensibly.
            self._warmup_active = (it - start_iter) < self.config.distillation_warmup_steps

            # Advance the DAgger -> PPO schedule and gate the std-param-group LR.
            self.adjust_ppo_dagger_coeff(it)
            self._set_std_lr_from_ppo_coef()

            with self.logging_helper.record_collection_time():
                obs_dict = self._rollout_step(obs_dict)

            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()

            if self.is_main_process:
                self._post_epoch_logging(it, loss_dict)

            if it % self.config.save_interval == 0 and self.is_main_process:
                self.save(os.path.join(self.log_dir, f"model_{it:05d}.pt"))

        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"))

    # ---------------------------------------------------------- ppo/dagger mix

    def adjust_ppo_dagger_coeff(self, iteration: int) -> None:
        """Ramp the PPO coefficient from 0 to 0.9 between
        ``ppo_start_epoch`` and ``dagger_end_epoch``."""
        start = self.config.ppo_start_epoch
        end = self.config.dagger_end_epoch
        if iteration < start:
            self.ppo_coef = 0.0
        elif iteration >= end:
            self.ppo_coef = 0.9
        else:
            self.ppo_coef = min((iteration - start) / max(end - start, 1), 0.9)

    def _set_std_lr_from_ppo_coef(self) -> None:
        """Zero the std param group's LR while DAgger dominates the loss.

        When PPO is active, the std group follows the *current* decayed
        ``self.learning_rate`` (mirrors far-tracking my_distillation.py:977-979).
        Using ``self.config.learning_rate`` instead would insulate the std
        parameter from the KL-driven adaptive LR decay and let it drift to
        a higher steady-state value than far-tracking's.
        """
        std_lr = 0.0 if self.ppo_coef <= 0.1 else self.learning_rate
        self.optimizer.param_groups[self._STD_GROUP_IDX]["lr"] = std_lr

    # ------------------------------------------------------------- rollout step

    def _rollout_step(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                actor_obs, teacher_obs, critic_obs, depth_obs = self._extract_groups(obs_dict)

                # Sample student action and compute value / teacher label.
                actions = self.policy.act({"actor_obs": actor_obs, "depth_obs": depth_obs})
                values = self.policy.evaluate({"critic_obs": critic_obs}).detach()
                teacher_actions = self.policy.teacher_act(teacher_obs).detach()

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
                self._update_ppo_gate_stats(action_mean, teacher_actions)

                # During the distillation warmup window the env follows the
                # teacher's action (expert trajectories). We still record the
                # student's (actions, log_prob, mean, sigma) so PPO + DAgger
                # updates train the student head; the only thing that changes
                # is which policy the physics follows this step.
                step_actions = teacher_actions if self._warmup_active else actions
                next_obs, rewards, dones, infos = self.env.step({"actions": step_actions})
                for k in next_obs:
                    if isinstance(next_obs[k], torch.Tensor):
                        next_obs[k] = next_obs[k].to(self.device)
                    elif isinstance(next_obs[k], dict):
                        next_obs[k] = {kk: vv.to(self.device) for kk, vv in next_obs[k].items()}
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
                    actor_obs=actor_obs,
                    teacher_obs=teacher_obs,
                    critic_obs=critic_obs,
                    depth_obs=depth_obs,
                    actions=actions,
                    teacher_actions=teacher_actions,
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
            last_actor_obs, _last_teacher_obs, last_critic_obs, _last_depth_obs = self._extract_groups(obs_dict)
            last_values = self.policy.evaluate({"critic_obs": last_critic_obs}).detach().to(self.device)
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
        advantage = 0.0
        returns = torch.zeros_like(values)
        num_steps = returns.shape[0]
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.config.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.config.gamma * self.config.lam * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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
        self, student_mean: torch.Tensor, teacher_actions: torch.Tensor
    ) -> None:
        """Hook for accumulating per-motion-frame statistics during rollout.

        No-op by default. Override alongside :meth:`_get_ppo_gate` to make the
        gate adapt to where imitation is actually falling short, rather than to a
        hand-specified window.
        """
        return

    def _dagger_loss_per_sample(
        self, student_mean: torch.Tensor, teacher_actions: torch.Tensor
    ) -> torch.Tensor:
        """Behaviour-cloning loss per sample, shape ``(batch,)``.

        Override to drop samples — e.g. an application whose ``teacher_act``
        marks "no teacher applies" with a zero action can mask those rows here.
        This is the extension point the combined loss uses; :meth:`_dagger_loss`
        is its batch mean and exists for logging and for callers that want a
        scalar.
        """
        return self.distill_loss_fn(student_mean, teacher_actions, reduction="none").mean(dim=-1)

    def _dagger_loss(
        self, student_mean: torch.Tensor, teacher_actions: torch.Tensor
    ) -> torch.Tensor:
        """Batch-mean behaviour-cloning loss. See :meth:`_dagger_loss_per_sample`."""
        return self._dagger_loss_per_sample(student_mean, teacher_actions).mean()

    def _training_step(self) -> dict[str, float]:
        generator = self.storage.mini_batch_generator(self.config.num_mini_batches, self.config.num_learning_epochs)
        loss_accum: dict[str, torch.Tensor] = {}
        num_updates = 0

        mini_batch: DistillationMinibatch
        for mini_batch in generator:
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
            # Mirror far-tracking's rsl_rl runner `Loss/learning_rate`
            # (emitted from their on_policy_runner). FAR-pi also logs the
            # same value under `LR/student_critic_lr`; this key is here
            # purely to match the far-tracking wandb schema.
            "learning_rate": float(self.learning_rate),
        }
        if num_updates > 0:
            for k, tensor in loss_accum.items():
                loss_dict[k] = (tensor / num_updates).item()

        self.storage.clear()
        return loss_dict

    def _update_step(self, mini_batch: DistillationMinibatch) -> dict[str, torch.Tensor]:
        actor_obs = mini_batch["actor_obs"]
        depth_obs = mini_batch["depth_obs"]
        critic_obs = mini_batch["critic_obs"]
        actions = mini_batch["actions"]
        teacher_actions = mini_batch["teacher_actions"]
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
            self.config.desired_kl is not None
            and self.config.schedule == "adaptive"
            and self.ppo_coef > 0.1
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
        value_clipped = target_values + (value - target_values).clamp(
            -self.config.clip_param, self.config.clip_param
        )
        value_losses = (value - returns).pow(2)
        value_losses_clipped = (value_clipped - returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        entropy_mean = entropy.mean()

        dagger_per_sample = self._dagger_loss_per_sample(student_mean, teacher_actions)
        dagger_loss = dagger_per_sample.mean()

        # Combined loss. Matches far-tracking's DepthDistillationPPO formula at
        # my_distillation.py:978-985, 1037: value loss is **always active**
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
        ppo_loss = self.config.value_loss_coef * value_loss + (
            coef * (surrogate_per_sample - self.config.entropy_coef * entropy)
        ).mean()
        total_loss = ppo_loss + (
            (1.0 - coef) * self.config.dagger_loss_coef * dagger_per_sample
        ).mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        # Average gradients across ranks so every rank's optimizer.step()
        # applies the same update. Clip-grad-norm operates on the averaged
        # gradients (matches holosoma PPO._update_algo_step at
        # ppo.py:505-510 and far-tracking my_distillation.py:1057-1063).
        if self.is_multi_gpu:
            self._reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "total_loss": total_loss.detach(),
            "surrogate_loss": surrogate_loss.detach(),
            "value_loss": value_loss.detach(),
            # Logged as "behavior" to match far-tracking's distillation loss
            # dict key (my_distillation.py:238, 1073). far-tracking bakes
            # (1 - ppo_coeff) into the tensor before .item() at
            # my_distillation.py:1021, 1073; we mirror that here so the
            # plot scales line up across the two codebases. The actual
            # BC gradient magnitude is unchanged — the (1 - ppo_coef)
            # factor is already applied to `total_loss` at line 553.
            # far-tracking bakes the DAgger weight into the logged value; with
            # per-sample gating the weight varies, so use its batch mean rather
            # than the scalar ``1 - ppo_coef`` (which would misreport whenever
            # the gate is not all-ones).
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
        # Closed-form KL for diagonal Gaussians; matches the far-tracking
        # reference's adaptive-LR formula.
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma / old_sigma + 1.0e-5)
                + (torch.square(old_sigma) + torch.square(old_mu - mu)) / (2.0 * torch.square(sigma))
                - 0.5,
                dim=-1,
            )
            if weights is None:
                kl_mean = torch.mean(kl)
            else:
                # Weighted mean, not a weighted sum: the threshold comparison in
                # ``_update_learning_rate`` is against an absolute KL, so the
                # scale must stay independent of how much of the batch is gated.
                kl_mean = (kl * weights).sum() / weights.sum().clamp(min=1e-6)
            # Average the KL estimate across ranks so every rank makes the
            # same adaptive-LR decision. Matches holosoma PPO at
            # ppo.py:640-642 and far-tracking my_distillation.py:529-530.
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
            return kl_mean

    def _update_learning_rate(self, kl_mean: torch.Tensor) -> None:
        # Only rank 0 computes the LR update; broadcast the decision so
        # every rank's optimizer ticks in lockstep. Without this, tiny
        # float-comparison mismatches can push ranks onto different LRs
        # and the all-reduced grads become inconsistent.
        # Mirror of far-tracking my_distillation.py:533-543, 940-950, 1323-1334.
        if not self.is_multi_gpu or self.gpu_global_rank == 0:
            if kl_mean > self.config.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.config.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

        if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()

        # Push the decayed LR to all groups, including the std group when it
        # is active (ppo_coef > 0.1). Matches far-tracking
        # my_distillation.py:953-954 which updates every param group.
        for idx, group in enumerate(self.optimizer.param_groups):
            if idx == self._STD_GROUP_IDX and self.ppo_coef <= 0.1:
                # Still gated to 0 while DAgger dominates; don't overwrite.
                continue
            group["lr"] = self.learning_rate

    # --------------------------------------------------------------- multi-gpu

    def _reduce_parameters(self) -> None:
        """All-reduce gradients across ranks and scale by world size.

        Mirrors holosoma PPO._reduce_parameters (ppo.py:773-793) but covers
        every trainable group managed by the single DistillationPPO
        optimizer: student MLP, depth backbone, critic, and the `std`
        parameter. Teacher params are frozen (eval mode, requires_grad=False)
        so their grads are None and are skipped naturally.
        """
        grads = [p.grad.view(-1) for p in self.policy.parameters() if p.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for p in self.policy.parameters():
            if p.grad is None:
                continue
            numel = p.numel()
            p.grad.data.copy_(all_grads[offset : offset + numel].view_as(p.grad))
            offset += numel

    def _synchronize_model_weights(self) -> None:
        """Broadcast initial policy weights from rank 0 to every other rank.

        Called once at setup so every rank starts training from the same
        random initialization. Matches holosoma PPO._synchronize_model_weights
        (ppo.py:795-805).
        """
        for p in self.policy.parameters():
            torch.distributed.broadcast(p.data, src=0)
        logger.info(f"Synchronized DistillationPPO model weights across {self.gpu_world_size} GPUs")

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

    # ---------------------------------------------------------- save/load/teacher

    def save(self, path: str | None = None, name: str = "last.ckpt") -> None:
        if path is None:
            path = os.path.join(self.log_dir, name)
        checkpoint_dict: dict[str, Any] = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "learning_rate": self.learning_rate,
            "ppo_coef": self.ppo_coef,
        }
        checkpoint_dict.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        env_state = self._collect_env_state()
        if env_state:
            checkpoint_dict["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(checkpoint_dict, path)

    def load(self, path: str | None) -> dict | None:
        if path is None:
            return None
        logger.info(f"Loading DistillationPPO checkpoint from {path}")
        loaded = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(loaded["model_state_dict"])
        if "optimizer_state_dict" in loaded:
            try:
                self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
            except Exception as exc:  # optimizer state has no public signature to validate
                logger.warning(f"Could not load optimizer state: {exc}")
        self.current_learning_iteration = int(loaded.get("iter", 0))
        if "learning_rate" in loaded:
            self.learning_rate = float(loaded["learning_rate"])
        if "ppo_coef" in loaded:
            self.ppo_coef = float(loaded["ppo_coef"])
        self._restore_env_state(loaded.get("env_state"))
        # Teacher weights inside policy.state_dict are assumed to already match
        # a previously-loaded teacher; re-freeze defensively.
        self.policy.teacher.eval()
        for p in self.policy.teacher.parameters():
            p.requires_grad_(False)
        return loaded.get("infos")

    def load_teacher(self, paths: str | list[str]) -> None:
        """Load one or more teacher-only checkpoints into ``policy.teachers``.

        Accepts a single path or a list of paths (one per teacher slot,
        in the same order as ``policy.teachers``). Each path may be a
        holosoma PPO algo checkpoint (contains ``actor_model_state_dict`` or
        ``model_state_dict``) or a raw state_dict. Delegates renaming to
        :meth:`DepthStudentTeacherCritic.load_teacher_state_dict`.
        """
        if isinstance(paths, str):
            paths = [paths]
        if len(paths) != self.policy.num_teachers:
            raise RuntimeError(
                f"load_teacher: got {len(paths)} paths but policy has "
                f"{self.policy.num_teachers} teacher slots. Rebuild the policy "
                f"with matching num_teachers or pass the correct number of paths."
            )
        for i, path in enumerate(paths):
            logger.info(f"Loading teacher[{i}] weights from {path}")
            loaded = torch.load(path, map_location=self.device)
            # Prefer the PPO actor state dict if present; fall back to the
            # generic bucket; else assume the top-level is already a raw
            # state dict.
            if isinstance(loaded, dict) and "actor_model_state_dict" in loaded:
                teacher_ckpt = {"model_state_dict": loaded["actor_model_state_dict"]}
            else:
                teacher_ckpt = loaded
            self.policy.load_teacher_state_dict(teacher_ckpt, strict=True, teacher_index=i)
        for t in self.policy.teachers:
            t.eval()
            for p in t.parameters():
                p.requires_grad_(False)

    # --------------------------------------------------------- inference / eval

    @property
    def inference_model(self) -> DepthStudentTeacherCritic:
        return self.policy

    @property
    def actor_onnx_wrapper(self):
        # ONNX export for the depth student will need a custom wrapper that
        # takes both proprio and depth obs; not implemented in this phase.
        raise NotImplementedError("ONNX export for DistillationPPO will be wired up in a later phase.")

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        self.policy.eval()
        if device is not None:
            self.policy.to(device)

        @torch.no_grad()
        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            return self.policy.act_inference(
                {
                    "actor_obs": obs["actor_obs"],
                    "depth_obs": obs.get("depth_obs"),
                }
            )

        return policy_fn

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None) -> None:
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state: dict[str, Any] = {"done_indices": [], "stop": False}
        eval_policy = self.get_inference_policy()

        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})

        actor_obs, _teacher_obs, critic_obs, depth_obs = self._extract_groups(obs_dict)
        actor_state["obs"]["actor_obs"] = actor_obs
        actor_state["obs"]["critic_obs"] = critic_obs
        actor_state["obs"]["depth_obs"] = depth_obs

        for step in itertools.islice(itertools.count(), max_eval_steps):
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state, eval_policy)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)

        self._post_evaluate_policy()

    def _create_eval_callbacks(self) -> None:
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env: bool = True) -> None:
        self._eval_mode()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()
        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self) -> None:
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict, eval_policy: Callable) -> dict:
        actor_obs, _teacher_obs, _critic_obs, depth_obs = self._extract_groups(actor_state["obs"])
        actions = eval_policy({"actor_obs": actor_obs, "depth_obs": depth_obs})
        actor_state["actions"] = actions
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state: dict) -> dict:
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    # Override env_step so the returned actor_state keeps the same shape as PPO's.
    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update({"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras})
        return actor_state
