from __future__ import annotations

import itertools
import os
from typing import TypedDict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from rich.console import Console
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.data_utils import RolloutStorage
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.modules.module_utils import (
    setup_ppo_actor_module,
    setup_ppo_critic_module,
)
from holosoma.config_types.algo import PPOConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_motion_and_policy_as_onnx,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)

console = Console()


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, device, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}")

        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return

        if dist.is_available() and dist.is_initialized():
            local_batch_size = x.shape[0]
            world_size = dist.get_world_size()
            global_batch_size = world_size * local_batch_size

            x_shifted = x - self._mean
            local_sum_shifted = torch.sum(x_shifted, dim=0, keepdim=True)
            local_sum_sq_shifted = torch.sum(x_shifted.pow(2), dim=0, keepdim=True)

            stats_to_sync = torch.cat([local_sum_shifted, local_sum_sq_shifted], dim=0)
            dist.all_reduce(stats_to_sync, op=dist.ReduceOp.SUM)
            global_sum_shifted, global_sum_sq_shifted = stats_to_sync

            batch_mean_shifted = global_sum_shifted / global_batch_size
            batch_var = global_sum_sq_shifted / global_batch_size - batch_mean_shifted.pow(2)
            batch_mean = batch_mean_shifted + self._mean
        else:
            global_batch_size = x.shape[0]
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        new_count = self.count + global_batch_size

        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (global_batch_size / new_count))

        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * global_batch_size
        M2 = m_a + m_b + delta2.pow(2) * (self.count * global_batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)


class Minibatch(TypedDict):
    """A minibatch of data for training a PPO agent."""

    actor_obs: torch.Tensor
    """The observation of the actor.

    Shape: (mini_batch_size, actor_obs_dim), dtype: torch.float32
    """

    critic_obs: torch.Tensor
    """The observation of the critic.

    Shape: (mini_batch_size, critic_obs_dim), dtype: torch.float32
    """

    actions: torch.Tensor
    """The actions taken by the agent.

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """

    rewards: torch.Tensor
    """The rewards received from the environment.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    dones: torch.Tensor
    """Whether each episode is done after taking the action.

    Shape: (mini_batch_size, 1), dtype: torch.bool
    """

    values: torch.Tensor
    """The value estimates from the critic.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    returns: torch.Tensor
    """The computed (unnormalized) returns for each step.

    The returns are computed following Generalized Advantage Estimation (GAE).

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    advantages: torch.Tensor
    """The computed (normalized) advantages for each step.

    The advantages are computed following Generalized Advantage Estimation (GAE).

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    actions_log_prob: torch.Tensor
    """The log probabilities of the actions.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    action_mean: torch.Tensor
    """The mean of the action distribution (assuming Gaussian distribution).

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """

    action_sigma: torch.Tensor
    """The standard deviation of the action distribution (assuming Gaussian distribution).

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """


class PPO(BaseAlgo):
    config: PPOConfig

    def __init__(self, env: BaseTask, config: PPOConfig, log_dir, device="cpu", multi_gpu_cfg: dict | None = None):
        super().__init__(env, config, device, multi_gpu_cfg)
        self.log_dir = log_dir
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

        self._init_config()

        self.current_learning_iteration = 0
        self.eval_callbacks: list[RLEvalCallback] = []
        _ = self.env.reset_all()

    def _init_config(self) -> None:
        self.algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()

        # Observation manager system - history is defined per-module in module_dict
        assert self.env.observation_manager is not None
        self.algo_history_length_dict = {
            "actor_obs": self.env.observation_manager.cfg.groups["actor_obs"].history_length,
            "critic_obs": self.env.observation_manager.cfg.groups["critic_obs"].history_length,
        }

        self.num_act = self.env.robot_config.actions_dim

        self.actor_learning_rate = self.config.actor_learning_rate
        self.max_actor_learning_rate = self.config.max_actor_learning_rate or max(self.actor_learning_rate, 1e-2)
        self.min_actor_learning_rate = self.config.min_actor_learning_rate or min(self.actor_learning_rate, 1e-5)
        self.critic_learning_rate = self.config.critic_learning_rate
        self.max_critic_learning_rate = self.config.max_critic_learning_rate or max(self.critic_learning_rate, 1e-2)
        self.min_critic_learning_rate = self.config.min_critic_learning_rate or min(self.critic_learning_rate, 1e-5)

        # Observation related Config
        self.use_symmetry = self.config.use_symmetry
        self.empirical_normalization = self.config.empirical_normalization
        self._init_obs_keys()

    def _init_obs_keys(self):
        self.actor_obs_keys = self.config.module_dict.actor.input_dim
        self.critic_obs_keys = self.config.module_dict.critic.input_dim

    def setup(self):
        logger.info("Setting up PPO")
        self._setup_models_and_optimizer()
        logger.info("Setting up Storage")
        self._setup_storage()

        # Log curriculum synchronization status for multi-GPU training
        if self.is_multi_gpu:
            if self.has_curricula_enabled():
                logger.info(f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs")

    def _setup_models_and_optimizer(self):
        self.actor = setup_ppo_actor_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )
        self.critic = setup_ppo_critic_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.module_dict.critic,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )

        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(shape=actor_obs_dim, device=self.device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(shape=critic_obs_dim, device=self.device)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        if self.use_symmetry:
            self.symmetry_utils = SymmetryUtils(self.env)

        # Synchronize model weights across GPUs after initialization
        if self.is_multi_gpu:
            self._synchronize_model_weights()

        self.actor_optimizer = instantiate(
            self.config.actor_optimizer, params=self.actor.parameters(), lr=self.actor_learning_rate
        )
        self.critic_optimizer = instantiate(
            self.config.critic_optimizer, params=self.critic.parameters(), lr=self.critic_learning_rate
        )

    def _get_obs_dim(self, obs_keys: list[str]) -> int:
        """Compute total observation dimension for given observation keys."""
        obs_dim = 0
        for obs_key in obs_keys:
            key_dim = self.algo_obs_dim_dict[obs_key]
            assert isinstance(key_dim, int), f"Observation dimension for {obs_key} is not an integer: {key_dim}"
            # Note: algo_obs_dim_dict from observation_manager.get_obs_dims() already includes history
            obs_dim += key_dim
        return obs_dim

    def _get_zero_input(self):
        """
        Create a dummy (all-zero) input for the actor.

        During training, we cannot use the logic in `self.get_example_obs()`, since it resets environments mid-rollout.
        """
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        return torch.zeros(1, actor_obs_dim, device=self.device)

    def _normalize_actor_obs(self, actor_obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if self.empirical_normalization:
            return self.actor_obs_normalizer(actor_obs, update=update)
        return actor_obs

    def _normalize_critic_obs(self, critic_obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if self.empirical_normalization:
            return self.critic_obs_normalizer(critic_obs, update=update)
        return critic_obs

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.config.num_steps_per_env, device=self.device)
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        print(f"Registering key: actor_obs with shape: {actor_obs_dim}")
        self.storage.register("actor_obs", shape=(actor_obs_dim,), dtype=torch.float)

        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        print(f"Registering key: critic_obs with shape: {critic_obs_dim}")
        self.storage.register("critic_obs", shape=(critic_obs_dim,), dtype=torch.float)

        # Register others based on Minibatch structure
        minibatch_keys = [
            ("actions", (self.num_act,), torch.float),
            ("rewards", (1,), torch.float),
            ("dones", (1,), torch.bool),
            ("values", (1,), torch.float),
            ("returns", (1,), torch.float),
            ("advantages", (1,), torch.float),
            ("actions_log_prob", (1,), torch.float),
            ("action_mean", (self.num_act,), torch.float),
            ("action_sigma", (self.num_act,), torch.float),
        ]
        for key, shape, dtype in minibatch_keys:
            self.storage.register(key, shape=shape, dtype=dtype)

    def _eval_mode(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_obs_normalizer.eval()
        self.critic_obs_normalizer.eval()

    def _train_mode(self):
        self.actor.train()
        self.critic.train()
        self.actor_obs_normalizer.train()
        self.critic_obs_normalizer.train()

    def learn(self):
        self._train_mode()

        obs_dict = self.env.reset_all()

        # Initialize environments with different episode length buffers
        # Must happen AFTER reset_all() to avoid being overwritten by reset
        if self.config.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        for obs_key in obs_dict:
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        for it in range(
            self.current_learning_iteration,
            self.current_learning_iteration + self.config.num_learning_iterations,
        ):
            self.current_learning_iteration = it

            # Synchronize curriculum metrics across GPUs before rollout
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            with self.logging_helper.record_collection_time():
                obs_dict = self._rollout_step(obs_dict)

            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()

            if self.is_main_process:
                self._post_epoch_logging(it, loss_dict)

            if it % self.config.save_interval == 0 and self.is_main_process:
                self.save(os.path.join(self.log_dir, f"model_{it:05d}.pt"))
                self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{it:05d}.onnx"))

        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"))
            self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.onnx"))

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                # Environment step
                actor_obs_raw = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                critic_obs_raw = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
                actor_obs = self._normalize_actor_obs(actor_obs_raw)
                critic_obs = self._normalize_critic_obs(critic_obs_raw)

                actions = self.actor.act({"actor_obs": actor_obs})
                values = self.critic.evaluate({"critic_obs": critic_obs}).detach()

                obs_dict, rewards, dones, infos = self.env.step({"actions": actions})

                for obs_key in obs_dict:
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # Compute bootstrap value for timeouts
                final_rewards = torch.zeros_like(rewards)
                if infos["time_outs"].any():
                    final_critic_obs = torch.cat([infos["final_observations"][k] for k in self.critic_obs_keys], dim=1)
                    final_critic_obs = self._normalize_critic_obs(final_critic_obs, update=False)
                    final_values = self.critic.evaluate({"critic_obs": final_critic_obs}).detach()
                    final_rewards += self.config.gamma * torch.squeeze(
                        final_values * infos["time_outs"].unsqueeze(1).to(self.device), 1
                    )

                # Add transition to storage
                self.storage.add(
                    actor_obs=actor_obs,
                    critic_obs=critic_obs,
                    actions=actions,
                    values=values,
                    actions_log_prob=self.actor.get_actions_log_prob(actions).detach().unsqueeze(1),
                    action_mean=self.actor.action_mean.detach(),
                    action_sigma=self.actor.action_std.detach(),
                    rewards=(rewards + final_rewards).view(-1, 1),
                    dones=dones.view(-1, 1),
                )

                # Reset actor and critic for completed envs
                self.actor.reset(dones)
                self.critic.reset(dones)

                if self.log_dir is not None:
                    # Update episode stats using logging helper
                    self.logging_helper.update_episode_stats(rewards, dones, infos)

            # Return / Advantage computation
            last_critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
            last_critic_obs = self._normalize_critic_obs(last_critic_obs, update=False)
            last_values = self.critic.evaluate({"critic_obs": last_critic_obs}).detach().to(self.device)
            returns, advantages = self._compute_returns_and_advantages(
                last_values,
                self.storage["values"].to(self.device),
                self.storage["dones"].to(self.device),
                self.storage["rewards"].to(self.device),
            )

            self.storage["returns"] = returns
            self.storage["advantages"] = advantages

        return obs_dict

    def _compute_returns_and_advantages(self, last_values, values, dones, rewards):
        advantage = 0
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

        if self.is_multi_gpu:
            advantages = self._normalize_advantages_multi_gpu(advantages)
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def _training_step(self) -> dict[str, float]:
        generator = self.storage.mini_batch_generator(self.config.num_mini_batches, self.config.num_learning_epochs)

        minibatch: Minibatch
        loss_dict = {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0}
        for minibatch in generator:
            loss_dict = self._update_algo_step(minibatch, loss_dict)

        num_updates = self.config.num_learning_epochs * self.config.num_mini_batches
        for key in loss_dict:
            loss_dict[key] /= num_updates
        self.storage.clear()
        return loss_dict

    def _update_algo_step(self, minibatch: Minibatch, loss_dict: dict[str, float]):
        ppo_loss_dict = self._compute_ppo_loss(minibatch)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        ppo_loss = ppo_loss_dict["actor_loss"] + ppo_loss_dict["critic_loss"]
        ppo_loss.backward()

        if self.is_multi_gpu:
            self._reduce_parameters()

        # Gradient step
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict["Value"] += ppo_loss_dict.pop("value_loss").item()
        loss_dict["Surrogate"] += ppo_loss_dict.pop("surrogate_loss").item()
        loss_dict["Entropy"] += ppo_loss_dict.pop("entropy_loss").item()
        loss_dict["KL"] += ppo_loss_dict.pop("kl_mean").item()
        for key, loss in ppo_loss_dict.items():
            if key not in loss_dict:
                loss_dict[key] = 0.0
            loss_value = loss.item() if torch.is_tensor(loss) else loss
            loss_dict[key] += loss_value
        return loss_dict

    def _compute_ppo_loss(self, minibatch: Minibatch):
        actions_batch = minibatch["actions"]
        target_values_batch = minibatch["values"]
        advantages_batch = minibatch["advantages"]
        returns_batch = minibatch["returns"]
        old_actions_log_prob_batch = minibatch["actions_log_prob"]
        old_mu_batch = minibatch["action_mean"]
        old_sigma_batch = minibatch["action_sigma"]

        # Symmetry augmentation
        original_batch_size = actions_batch.shape[0]
        if self.use_symmetry:
            actor_obs = self.symmetry_utils.augment_observations(
                obs=minibatch["actor_obs"],
                env=self.env,
                obs_list=self.actor_obs_keys,
            )
            critic_obs = self.symmetry_utils.augment_observations(
                obs=minibatch["critic_obs"],
                env=self.env,
                obs_list=self.critic_obs_keys,
            )
            actions_batch = self.symmetry_utils.augment_actions(
                actions=actions_batch,
            )
            num_aug = int(actor_obs.shape[0] / original_batch_size)
            old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
            target_values_batch = target_values_batch.repeat(num_aug, 1)
            advantages_batch = advantages_batch.repeat(num_aug, 1)
            returns_batch = returns_batch.repeat(num_aug, 1)
        else:
            actor_obs = minibatch["actor_obs"]
            critic_obs = minibatch["critic_obs"]

        self.actor.act({"actor_obs": actor_obs})
        value_batch = self.critic.evaluate({"critic_obs": critic_obs})
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        mu_batch = self.actor.action_mean[:original_batch_size]
        sigma_batch = self.actor.action_std[:original_batch_size]
        entropy_batch = self.actor.entropy[:original_batch_size]

        if self.config.desired_kl is not None and self.config.schedule == "adaptive":
            # Compute the KL divergence between the old and new action distributions
            kl_mean = self._compute_kl_div(old_mu_batch, old_sigma_batch, mu_batch, sigma_batch)
            self._update_learning_rate(kl_mean)

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
            -self.config.clip_param, self.config.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        if self.use_symmetry and (self.config.symmetry_actor_coef > 0.0 or self.config.symmetry_critic_coef > 0.0):
            mean_actions_batch = self.actor.act_inference({"actor_obs": actor_obs.detach().clone()})
            mean_actions_for_original_batch, mean_actions_for_symmetry_batch = (
                mean_actions_batch[:original_batch_size],
                mean_actions_batch[original_batch_size:],
            )
            mean_symmetry_actions_batch = self.symmetry_utils.augment_actions(
                actions=mean_actions_for_original_batch,
            )[original_batch_size:]
            symmetry_actor_loss = F.mse_loss(
                mean_actions_for_symmetry_batch,
                mean_symmetry_actions_batch,
            )

            # Symmetry critic loss
            symmetry_critic_loss = F.mse_loss(
                value_batch[:original_batch_size],
                value_batch[original_batch_size:],
            )
        else:
            symmetry_actor_loss = torch.tensor(0.0, device=self.device)
            symmetry_critic_loss = torch.tensor(0.0, device=self.device)

        entropy_loss = entropy_batch.mean()
        actor_loss = (
            surrogate_loss
            - self.config.entropy_coef * entropy_loss
            + self.config.symmetry_actor_coef * symmetry_actor_loss
        )

        critic_loss = self.config.value_loss_coef * value_loss + self.config.symmetry_critic_coef * symmetry_critic_loss

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "symmetry_actor_loss": symmetry_actor_loss,
            "symmetry_critic_loss": symmetry_critic_loss,
            "value_loss": value_loss,
            "surrogate_loss": surrogate_loss,
            "entropy_loss": entropy_loss,
            "kl_mean": kl_mean,
        }

    def _compute_kl_div(self, old_mu_batch, old_sigma_batch, mu_batch, sigma_batch) -> torch.Tensor:
        with torch.inference_mode():
            # Compute the KL divergence between the old and new action distributions
            old_dist = Normal(old_mu_batch, old_sigma_batch)
            new_dist = Normal(mu_batch, sigma_batch)
            kl = kl_divergence(old_dist, new_dist).sum(-1)
            kl_mean = torch.mean(kl)

            # Reduce the KL divergence across all GPUs
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
        return kl_mean

    def _update_learning_rate(self, kl_mean: torch.Tensor):
        if kl_mean > self.config.desired_kl * 2.0:
            self.actor_learning_rate = max(self.min_actor_learning_rate, self.actor_learning_rate / 1.5)
            self.critic_learning_rate = max(self.min_critic_learning_rate, self.critic_learning_rate / 1.5)
        elif kl_mean < self.config.desired_kl / 2.0 and kl_mean > 0.0:
            self.actor_learning_rate = min(self.max_actor_learning_rate, self.actor_learning_rate * 1.5)
            self.critic_learning_rate = min(self.max_critic_learning_rate, self.critic_learning_rate * 1.5)

        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.actor_learning_rate
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = self.critic_learning_rate

    def load(self, ckpt_path: str | None) -> dict | None:
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            if self.empirical_normalization and loaded_dict.get("actor_obs_normalizer_state_dict") is not None:
                self.actor_obs_normalizer.load_state_dict(loaded_dict["actor_obs_normalizer_state_dict"])
            if self.empirical_normalization and loaded_dict.get("critic_obs_normalizer_state_dict") is not None:
                self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_normalizer_state_dict"])
            if self.config.load_optimizer:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                self.actor_learning_rate = loaded_dict["actor_optimizer_state_dict"]["param_groups"][0]["lr"]
                self.critic_learning_rate = loaded_dict["critic_optimizer_state_dict"]["param_groups"][0]["lr"]
                logger.info("Optimizer loaded from checkpoint")
            self.current_learning_iteration = loaded_dict["iter"]
            self._restore_env_state(loaded_dict.get("env_state"))
            return loaded_dict.get("infos")
        return None

    def save(self, path, infos=None):
        checkpoint_dict = {
            "actor_model_state_dict": self.actor.state_dict(),
            "critic_model_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "actor_obs_normalizer_state_dict": (
                self.actor_obs_normalizer.state_dict() if self.empirical_normalization else None
            ),
            "critic_obs_normalizer_state_dict": (
                self.critic_obs_normalizer.state_dict() if self.empirical_normalization else None
            ),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        checkpoint_dict.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        env_state = self._collect_env_state()
        if env_state:
            checkpoint_dict["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(checkpoint_dict, path)

    def export(self, onnx_file_path: str):
        """Export the `.onnx` of the policy to & save it to `path`.

        This is intended to enable deployment, but not resuming training.
        For storing checkpoints to resume training, see `PPO.save()`
        """
        # Save current training state
        was_training = self.actor.training

        # Set model to evaluation mode for export so we don't affect gradients mid-rollout
        self._eval_mode()

        # Save the .onnx file to filesystem
        motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is not None:
            export_motion_and_policy_as_onnx(
                self.actor_onnx_wrapper,
                motion_command,
                onnx_file_path,
                self.device,
            )
        else:
            export_policy_as_onnx(
                wrapper=self.actor_onnx_wrapper,
                onnx_file_path=onnx_file_path,
                example_obs_dict={"actor_obs": self._get_zero_input()},
            )

        # Extract control gains and velocity limits & attach to onnx as metadata
        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.env)
        action_scales = getattr(self.env, "action_scales", None)
        if action_scales is None:
            action_scale_metadata: float | list[float] = float(self.env.robot_config.control.action_scale)
        else:
            action_scale_metadata = action_scales.detach().cpu().tolist()
        # Extract URDF text from the robot config
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "action_scale": action_scale_metadata,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
        }
        metadata.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))

        attach_onnx_metadata(
            onnx_path=onnx_file_path,
            metadata=metadata,
        )

        # Upload the .onnx file to wandb
        self.logging_helper.save_to_wandb(onnx_file_path)

        # Restore original training state
        if was_training:
            self._train_mode()

    def _post_epoch_logging(self, it, loss_dict):
        extra_log_dicts = {
            "Policy": {
                "mean_noise_std": self.actor.std.mean().item(),
            },
        }
        loss_dict["actor_learning_rate"] = self.actor_learning_rate
        loss_dict["critic_learning_rate"] = self.critic_learning_rate
        # Use logging helper
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

    def _reduce_parameters(self):
        grads = [
            param.grad.view(-1)
            for model in [self.actor, self.critic]
            for param in model.parameters()
            if param.grad is not None
        ]
        if not grads:
            return
        all_grads = torch.cat(grads)

        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for model in [self.actor, self.critic]:
            for param in model.parameters():
                if param.grad is not None:
                    numel = param.numel()
                    param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad))
                    offset += numel

    def _synchronize_model_weights(self):
        """Synchronize actor and critic weights across all GPUs."""
        # Broadcast actor weights from rank 0 to all other ranks
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast critic weights from rank 0 to all other ranks
        for param in self.critic.parameters():
            torch.distributed.broadcast(param.data, src=0)

        logger.info(f"Synchronized model weights across {self.gpu_world_size} GPUs")

    def _normalize_advantages_multi_gpu(self, advantages):
        local_stats = torch.stack(
            [
                advantages.mean(),
                (advantages**2).mean(),
            ]
        )
        torch.distributed.all_reduce(local_stats, op=torch.distributed.ReduceOp.SUM)

        global_mean = local_stats[0] / self.gpu_world_size
        global_sq_mean = local_stats[1] / self.gpu_world_size
        global_variance = global_sq_mean - global_mean**2
        global_std = torch.sqrt(global_variance + 1e-8)

        return (advantages - global_mean) / global_std

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################

    @property
    def actor_onnx_wrapper(self):
        class ActorWrapper(nn.Module):
            def __init__(self, actor, actor_obs_normalizer, empirical_normalization):
                super().__init__()
                self.actor = actor
                self.actor_obs_normalizer = actor_obs_normalizer
                self.empirical_normalization = empirical_normalization

            def forward(self, actor_obs):
                if self.empirical_normalization:
                    actor_obs = self.actor_obs_normalizer(actor_obs, update=False)
                return self.actor.act_inference({"actor_obs": actor_obs})

        return ActorWrapper(self.actor, self.actor_obs_normalizer, self.empirical_normalization)

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update({"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras})
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        """Used for exporting policy as onnx."""
        obs_dict = self.env.reset_all()
        return {
            "actor_obs": torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1),
            "critic_obs": torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1),
        }

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        self.eval_policy = self.get_inference_policy()

        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})

        critic_obs = torch.cat([actor_state["obs"][k] for k in self.critic_obs_keys], dim=1)
        actor_state["obs"]["critic_obs"] = critic_obs

        actor_state = self._pre_eval_env_step(actor_state)

        for step in itertools.islice(itertools.count(), max_eval_steps):
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)

        self._post_evaluate_policy()

    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env=True):
        self._eval_mode()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict):
        actor_obs = torch.cat([actor_state["obs"][k] for k in self.actor_obs_keys], dim=1)
        actions = self.eval_policy({"actor_obs": actor_obs})
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def get_inference_policy(self, device=None):
        self.actor.eval()  # switch to evaluation mode (dropout for example)
        self.actor_obs_normalizer.eval()
        if device is not None:
            self.actor.to(device)
            self.actor_obs_normalizer.to(device)

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            actor_obs = self._normalize_actor_obs(obs["actor_obs"], update=False)
            return self.actor.act_inference({"actor_obs": actor_obs})

        return policy_fn
