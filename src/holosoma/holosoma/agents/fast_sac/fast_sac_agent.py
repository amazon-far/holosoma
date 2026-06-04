from __future__ import annotations

import copy
import itertools
import math
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Sequence

import tqdm
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.fast_sac.fast_sac import Actor, CNNActor, CNNCritic, Critic
from holosoma.agents.fast_sac.fast_sac_utils import (
    EmpiricalNormalization,
    SimpleReplayBuffer,
    save_params,
)
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.config_types.algo import FastSACConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.average_meters import TensorAverageMeterDict
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_motion_and_policy_as_onnx,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)
from holosoma.utils.safe_torch_import import (
    F,
    GradScaler,
    TensorboardSummaryWriter,
    TensorDict,
    autocast,
    nn,
    optim,
    torch,
)

torch.set_float32_matmul_precision("high")

#把holosoma.envs.base_task.base_task.BaseTask传入FastSACEnv中，进行包装，
# 使其适配FastSAC算法的输入输出要求
class FastSACEnv:
    def __init__(
        self,
        env: BaseTask,
        actor_obs_keys: Sequence[str],
        critic_obs_keys: Sequence[str],
    ):
        self._env = env
        self._actor_obs_keys = actor_obs_keys
        self._critic_obs_keys = critic_obs_keys

        # Initialize per-joint action boundaries for proper tanh scaling
        self._action_boundaries = self._compute_action_boundaries()

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self._env, name)

    def reset(self) -> torch.Tensor:
        obs_dict = self._env.reset_all()
        return torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)

    def reset_with_critic_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict = self._env.reset_all()
        actor_obs = torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)
        critic_obs = torch.cat([obs_dict[k] for k in self._critic_obs_keys], dim=1)
        return actor_obs, critic_obs
    
  
  

    
#在step函数中，智能体与环境进行交互，传入动作并获取下一个状态、奖励、done标志和其他信息。原始 env.step() 返回的是 observation 字典
# 然后根据配置的actor_obs_keys和critic_obs_keys，从环境返回的观测字典中提取相应的观测，并将它们拼接成适合actor和critic输入的格式。
# 最后，函数返回actor_obs、奖励、done标志和一个包含额外信息的字典。

#FastSAC 想要的是拼好的 actor_obs / critic_obs tensor
#所以这里负责转换格式，并额外保存 final_observations。

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        #得到环境返回的观测字典、奖励、done标志和其他信息，然后根据配置的actor_obs_keys和critic_obs_keys，从环境返回的观测字典中提取相应的观测，并将它们拼接成适合actor和critic输入的格式。最后，函数返回actor_obs、奖励、done标志和一个包含额外信息的字典。
        obs_dict, rew_buf, reset_buf, info_dict = self._env.step({"actions": actions})  # type: ignore[attr-defined]
        actor_obs = torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)
        critic_obs = torch.cat([obs_dict[k] for k in self._critic_obs_keys], dim=1)
        #如果 info_dict 中包含 "final_observations" 键，表示环境提供了最终的观测信息，
        # 那么就使用这些观测信息来构建 final_actor_obs 和 final_critic_obs。
        # 否则，就使用当前的 actor_obs 和 critic_obs 作为最终的观测信息。
        # 这些最终的观测信息将被包含在 extras 字典中返回，以供后续的学习更新使用。
        if "final_observations" in info_dict:
            # Use true final observations when available
            final_actor_obs = torch.cat([info_dict["final_observations"][k] for k in self._actor_obs_keys], dim=1)
            final_critic_obs = torch.cat([info_dict["final_observations"][k] for k in self._critic_obs_keys], dim=1)
        else:
            final_actor_obs = actor_obs
            final_critic_obs = critic_obs
        extras = {
            "time_outs": info_dict["time_outs"],#表示环境是否因为达到时间限制而终止
            # "observations" 包含了当前的观测信息，分为 "critic" 和 "final" 两部分。
            # "critic" 部分包含了供 critic 网络使用的观测，而 "final" 部分包含了供 actor 和 critic 使用的最终观测。
            # 这些观测是根据配置的 actor_obs_keys 和 critic_obs_keys 从环境返回的观测字典中提取并拼接而成的。
            "observations": {
                "critic": critic_obs,
                "final": {
                    "actor_obs": final_actor_obs,
                    "critic_obs": final_critic_obs,
                },
            },
            "episode": info_dict["episode"],
            "episode_all": info_dict["episode_all"],
            "raw_episode": info_dict.get("raw_episode", {}),
            "raw_episode_all": info_dict.get("raw_episode_all", {}),
            "to_log": info_dict["to_log"],
        }
        return actor_obs, rew_buf, reset_buf, extras
        """
        Compute per-joint action scaling factors based on robot configuration.
        Returns tensor of shape (num_dof,) containing the scaling factor for each joint.

        The scaling factor is the maximum difference between default and joint limits,
        ensuring that action=0 corresponds to default position and action=±1 reaches
        the furthest limit from default.
        """
        #计算每个关节的动作缩放因子，基于机器人配置中的默认位置和关节限制。
        # 这个函数确保了当动作为0时，机器人处于默认位置，而当动作为±1时，机器人能够达到距离默认位置最远的限制位置。
        # 这种缩放方式有助于稳定训练过程，并确保动作空间的合理范围。
    def _compute_action_boundaries(self) -> torch.Tensor:

        robot_config = self._env.robot_config

        # 这里我们直接使用机器人配置中的关节位置限制来计算动作缩放因子。
        # 对于每个关节，我们计算默认位置与上下限之间的最大距离，并将其作为缩放因子的一部分。
        # 这样可以确保动作空间的中心（动作=0）对应于默认位置，而动作的范围（±1）能够覆盖从默认位置到最远限制的位置。这种方法有助于稳定训练过程，并确保智能体能够有效地探索动作空间。
        dof_pos_lower_limits = torch.tensor(robot_config.dof_pos_lower_limit_list, device=self._env.device)
        dof_pos_upper_limits = torch.tensor(robot_config.dof_pos_upper_limit_list, device=self._env.device)

        # 得到每个关节的默认位置，如果机器人配置中没有提供默认位置，则默认为0
        # 。这些默认位置将用于计算动作缩放因子，确保动作空间的中心对应于默认位置，并且动作范围能够覆盖从默认位置到最远限制的位置。
        default_joint_angles = torch.zeros(len(robot_config.dof_names), device=self._env.device)
        for i, joint_name in enumerate(robot_config.dof_names):
            if joint_name in robot_config.init_state.default_joint_angles:
                default_joint_angles[i] = robot_config.init_state.default_joint_angles[joint_name]

        # Get action scale from robot config
        action_scale = robot_config.control.action_scale

        # Compute maximum range from default to either limit for each joint
        # 这里我们计算每个关节的动作缩放因子，基于机器人配置中的默认位置和关节限制。
        range_to_lower = torch.abs(dof_pos_lower_limits - default_joint_angles)
        range_to_upper = torch.abs(dof_pos_upper_limits - default_joint_angles)
        max_range = torch.maximum(range_to_lower, range_to_upper)

        # Account for action_scale: the environment applies actions_scaled = actions * action_scale
        #所以我们需要将 max_range 除以 action_scale 来得到最终的动作缩放因子，
        # 这样才能确保当智能体输出动作时，经过环境的缩放后能够正确地映射到关节的实际运动范围。
        action_scaling_factors = max_range / action_scale

        logger.info(f"Computed action scaling factors for {len(robot_config.dof_names)} DOFs")
        logger.info(f"Action scale: {action_scale}")
        logger.info(f"Scaling: {action_scaling_factors}")

        return action_scaling_factors

#初始化FastSACAgent，传入环境、配置、设备、日志目录等参数，并进行必要的设置和初始化
class FastSACAgent(BaseAlgo):
    """
    FastSAC is an efficient variant of Soft Actor-Critic (SAC) tuned for
    large-scale training with massively parallel simulation.
    See https://arxiv.org/abs/2505.22642 for more details about FastTD3.
    Detailed technical report for FastSAC will be available soon.
    """

    config: FastSACConfig
    env: FastSACEnv  # type: ignore[assignment]
    actor: Actor
    qnet: Critic
#初始化FastSACAgent，传入环境、配置、设备、日志目录等参数，并进行必要的设置和初始化。
# 这包括创建actor和critic网络、目标网络、优化器、经验回放缓冲区，以及设置日志记录工具和其他辅助组件。
# 同时，如果启用了多GPU训练，还会进行模型参数的同步，以确保所有GPU上的模型初始化一致。
    def __init__(
        self, env: BaseTask, config: FastSACConfig, device: str, log_dir: str, multi_gpu_cfg: dict | None = None
    ):
        wrapped_env = FastSACEnv(env, config.actor_obs_keys, config.critic_obs_keys)

        super().__init__(wrapped_env, config, device, multi_gpu_cfg)  # type: ignore[arg-type]
        self.unwrapped_env = env
        self.log_dir = log_dir
        self.global_step = 0
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=config.logging_interval,
            num_learning_iterations=config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )

        self.training_metrics = TensorAverageMeterDict()
        self.eval_callbacks: list[RLEvalCallback] = []
#根据配置文件中的参数，设置FastSAC算法的各种组件，包括actor、critic、优化器、学习率调度器、经验回放缓冲区等。同时，如果启用了多GPU训练，还会进行模型参数的同步。
    def setup(self) -> None:
        logger.info("Setting up FastSAC")

        # Log curriculum synchronization status for multi-GPU training
        if self.is_multi_gpu:
            if self.has_curricula_enabled():
                logger.info(f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs")

        args = self.config
        device = self.device
        env = self.env

        algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()

        algo_history_length_dict: Dict[str, int] = {}

        for group_cfg in self.env.observation_manager.cfg.groups.values():
            history_len = getattr(group_cfg, "history_length", 1)
            for term_name in group_cfg.terms:
                algo_history_length_dict[term_name] = history_len

        actor_obs_keys = self.config.actor_obs_keys
        critic_obs_keys = self.config.critic_obs_keys

        n_act = self.env.robot_config.actions_dim

        # Compute actor observation dimensions and store indices
        actor_obs_dim = 0
        self.actor_obs_indices = {}
        for obs_key in actor_obs_keys:
            history_len = algo_history_length_dict.get(obs_key, 1)
            obs_size = algo_obs_dim_dict[obs_key] * history_len

            # Store start and end indices for this observation key
            self.actor_obs_indices[obs_key] = {
                "start": actor_obs_dim,
                "end": actor_obs_dim + obs_size,
                "size": obs_size,
            }
            actor_obs_dim += obs_size

        self.actor_obs_dim = actor_obs_dim

        # Compute critic observation dimensions and store indices
        critic_obs_dim = 0
        self.critic_obs_indices = {}
        for obs_key in critic_obs_keys:
            history_len = algo_history_length_dict.get(obs_key, 1)
            obs_size = algo_obs_dim_dict[obs_key] * history_len

            # Store start and end indices for this observation key
            self.critic_obs_indices[obs_key] = {
                "start": critic_obs_dim,
                "end": critic_obs_dim + obs_size,
                "size": obs_size,
            }
            critic_obs_dim += obs_size

        self.scaler = GradScaler(enabled=args.amp)

        self.obs_normalization = args.obs_normalization
        if args.obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(shape=actor_obs_dim, device=device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(shape=critic_obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # 得到每个关节的动作缩放因子，基于机器人配置中的默认位置和关节限制。
        action_scale = env._action_boundaries if args.use_tanh else torch.ones(n_act, device=device)
        action_bias = torch.zeros(n_act, device=device)  # Assuming zero bias for now

        # Handle CNN actor/critic
        if args.use_cnn_encoder:
            # We assume that MLP doesn't take raw encoder observations
            actor_mlp_obs_keys = [k for k in actor_obs_keys if k != args.encoder_obs_key]
            critic_mlp_obs_keys = [k for k in critic_obs_keys if k != args.encoder_obs_key]
        else:
            actor_mlp_obs_keys = list(actor_obs_keys)
            critic_mlp_obs_keys = list(critic_obs_keys)
        actor_cls, critic_cls = (CNNActor, CNNCritic) if args.use_cnn_encoder else (Actor, Critic)

        self.actor = actor_cls(
            obs_indices=self.actor_obs_indices,
            obs_keys=actor_mlp_obs_keys,
            n_act=n_act,
            num_envs=env.num_envs,
            device=device,
            hidden_dim=args.actor_hidden_dim,
            log_std_max=args.log_std_max,
            log_std_min=args.log_std_min,
            use_tanh=args.use_tanh,
            use_layer_norm=args.use_layer_norm,
            action_scale=action_scale,
            action_bias=action_bias,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )
        self.qnet = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )

        print(self.actor)
        print(self.qnet)

        self.log_alpha = torch.tensor([math.log(args.alpha_init)], requires_grad=True, device=device)
        self.policy = self.actor.explore

        self.qnet_target = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.q_optimizer = optim.AdamW(
            list(self.qnet.parameters()),
            lr=args.critic_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.actor_optimizer = optim.AdamW(
            list(self.actor.parameters()),
            lr=args.actor_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )

        self.target_entropy = -n_act * args.target_entropy_ratio
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=args.alpha_learning_rate, fused=True, betas=(0.9, 0.95))

        logger.info(f"actor_obs_dim: {actor_obs_dim}, critic_obs_dim: {critic_obs_dim}")

        self.rb = SimpleReplayBuffer(
            n_env=env.num_envs,
            buffer_size=args.buffer_size,
            n_obs=actor_obs_dim,
            n_act=n_act,
            n_critic_obs=critic_obs_dim,
            n_steps=args.num_steps,
            gamma=args.gamma,
            device=device,
        )

        if args.use_symmetry:
            # using env._env is not really ideal..
            self.symmetry_utils = SymmetryUtils(env._env)

        # Synchronize model parameters across GPUs for consistent initialization
        if self.is_multi_gpu:
            self._synchronize_model_parameters()

    @contextmanager
    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.config.amp):
            yield

    def _synchronize_model_parameters(self):
        """Synchronize actor, qnet, and log_alpha parameters across all GPUs."""
        # Broadcast actor weights from rank 0 to all other ranks
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast qnet weights from rank 0 to all other ranks
        for param in self.qnet.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast log_alpha parameter from rank 0 to all other ranks
        torch.distributed.broadcast(self.log_alpha.data, src=0)

        # Load qnet_target weights from synced qnet
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        logger.info(f"Synchronized model parameters across {self.gpu_world_size} GPUs")

    def _all_reduce_model_grads(self, model: nn.Module) -> None:
        """Batches and all-reduces gradients across GPUs to reduce NCCL call count.

        This flattens all existing parameter gradients into a single contiguous
        tensor, performs one all_reduce, averages by world size, and then
        scatters the reduced values back into the original gradient tensors.
        """
        if not self.is_multi_gpu:
            return
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
                offset += n

    def _update_main(
        self, data: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        args = self.config

        scaler = self.scaler
        actor = self.actor
        qnet = self.qnet
        qnet_target = self.qnet_target
        q_optimizer = self.q_optimizer
        alpha_optimizer = self.alpha_optimizer

        with self._maybe_amp():
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            with torch.no_grad():
                next_state_actions, next_state_log_probs = actor.get_actions_and_log_probs(next_observations)
                discount = args.gamma ** data["next"]["effective_n_steps"]

                target_distributions = qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards - discount * bootstrap * self.log_alpha.exp() * next_state_log_probs,
                    bootstrap,
                    discount,
                )
                target_values = qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = qnet(critic_observations, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(qnet)

        scaler.unscale_(q_optimizer)
        if args.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(q_optimizer)
        scaler.update()
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.config.use_autotune:
            alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_loss = (-self.log_alpha.exp() * (next_state_log_probs.detach() + self.target_entropy)).mean()

            scaler.scale(alpha_loss).backward()

            if self.is_multi_gpu:
                if self.log_alpha.grad is not None:
                    torch.distributed.all_reduce(self.log_alpha.grad.data, op=torch.distributed.ReduceOp.SUM)
                    self.log_alpha.grad.data.copy_(self.log_alpha.grad.data / self.gpu_world_size)

            scaler.unscale_(alpha_optimizer)

            scaler.step(alpha_optimizer)
            scaler.update()

        return (
            rewards.mean(),
            critic_grad_norm.detach(),
            qf_loss.detach(),
            target_value_max.detach(),
            target_value_min.detach(),
            alpha_loss.detach(),
        )

    def _update_pol(self, data: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor = self.actor
        qnet = self.qnet
        actor_optimizer = self.actor_optimizer
        scaler = self.scaler
        args = self.config

        with self._maybe_amp():
            critic_observations = data["critic_observations"]

            actions, log_probs = actor.get_actions_and_log_probs(data["observations"])
            # For logging, this is a bit wasteful though, but could be useful
            with torch.no_grad():
                _, _, log_std = actor(data["observations"])
                action_std = log_std.exp().mean()
                # Compute policy entropy (negative log probability)
                policy_entropy = -log_probs.mean()

            q_outputs = qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(actor)

        scaler.unscale_(actor_optimizer)

        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(actor_optimizer)
        scaler.update()
        return (
            actor_grad_norm.detach(),
            actor_loss.detach(),
            policy_entropy.detach(),
            action_std.detach(),
        )

    def _sample_and_prepare_batches(
        self, batch_size: int, num_updates: int, normalize_obs, normalize_critic_obs
    ) -> list[TensorDict]:
        """
        Sample a large batch once and split it into smaller batches for each update.
        This reduces sampling overhead by `num_updates` and normalization overhead by `num_updates`.
        """
        # Sample a large batch (batch_size * num_updates)
        large_batch_size = batch_size * num_updates
        large_data = self.rb.sample(large_batch_size)
        samples_per_update = batch_size * self.env.num_envs

        if self.config.use_symmetry:
            samples_per_update *= 2

            augmented_large_data: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {"next": {}}

            augmented_large_data["observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_large_data["actions"] = self.symmetry_utils.augment_actions(actions=large_data["actions"])
            assert isinstance(augmented_large_data["next"], dict)
            augmented_large_data["next"]["observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["next"]["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_large_data["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )
            augmented_large_data["next"]["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["next"]["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )

            # Calculate augmentation factor and repeat non-augmented data
            observations_tensor = augmented_large_data["observations"]
            assert isinstance(observations_tensor, torch.Tensor), (
                "observations should be a Tensor after data augmentation"
            )
            num_aug = int(observations_tensor.shape[0] / large_data["next"]["rewards"].shape[0])
            augmented_large_data["next"]["rewards"] = large_data["next"]["rewards"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["dones"] = large_data["next"]["dones"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["truncations"] = large_data["next"]["truncations"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["effective_n_steps"] = large_data["next"]["effective_n_steps"].repeat(num_aug)  # type: ignore[index]

            # Override large_data
            large_data = augmented_large_data

        # Normalize all data once
        large_data["observations"] = normalize_obs(large_data["observations"])
        large_data["next"]["observations"] = normalize_obs(large_data["next"]["observations"])
        large_data["critic_observations"] = normalize_critic_obs(large_data["critic_observations"])
        large_data["next"]["critic_observations"] = normalize_critic_obs(large_data["next"]["critic_observations"])

        # Split into smaller batches
        prepared_batches = []

        for i in range(num_updates):
            start_idx = i * samples_per_update
            end_idx = (i + 1) * samples_per_update

            # Create a slice of the large batch
            batch_data = TensorDict(
                {
                    "observations": large_data["observations"][start_idx:end_idx],
                    "actions": large_data["actions"][start_idx:end_idx],
                    "next": {
                        "rewards": large_data["next"]["rewards"][start_idx:end_idx],
                        "dones": large_data["next"]["dones"][start_idx:end_idx],
                        "truncations": large_data["next"]["truncations"][start_idx:end_idx],
                        "observations": large_data["next"]["observations"][start_idx:end_idx],
                        "effective_n_steps": large_data["next"]["effective_n_steps"][start_idx:end_idx],
                    },
                    "critic_observations": large_data["critic_observations"][start_idx:end_idx],
                },
                batch_size=samples_per_update,
            )
            batch_data["next"]["critic_observations"] = large_data["next"]["critic_observations"][start_idx:end_idx]

            prepared_batches.append(batch_data)

        return prepared_batches

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return
        # Load checkpoint if specified
        torch_checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Handle DDP-wrapped models
        actor_state_dict = torch_checkpoint["actor_state_dict"]
        qnet_state_dict = torch_checkpoint["qnet_state_dict"]

        self.actor.load_state_dict(actor_state_dict)
        self.qnet.load_state_dict(qnet_state_dict)

        self.obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        self.critic_obs_normalizer.load_state_dict(torch_checkpoint["critic_obs_normalizer_state"])
        self.qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        self.log_alpha.data.copy_(torch_checkpoint["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(torch_checkpoint["actor_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(torch_checkpoint["q_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(torch_checkpoint["alpha_optimizer_state_dict"])
        self.scaler.load_state_dict(torch_checkpoint["grad_scaler_state_dict"])
        self.global_step = torch_checkpoint["global_step"]
        self._restore_env_state(torch_checkpoint.get("env_state"))
#真正的学习循环，在每个学习迭代中，智能体首先与环境进行交互，收集数据并存储在经验回放缓冲区中。然后，根据配置的更新频率和批量大小，从缓冲区中采样数据，并调用更新函数来优化actor和critic网络的参数。同时，还会进行目标网络的软更新，并记录各种训练指标以供分析和调试。
    def learn(self) -> None:
        args = self.config
        device = self.device
        #决定是否使用torch.compile来加速更新函数和策略函数的执行。
        # 根据配置中的compile参数，如果启用编译，则将_update_main、_update_pol、policy、obs_normalizer.forward和critic_obs_normalizer.forward函数进行编译，
        # 以提高它们的执行效率。这可以显著减少每次更新的计算时间，特别是在大规模训练中，从而加快整体学习过程。
        if args.compile:
            update_main = torch.compile(self._update_main)
            update_pol = torch.compile(self._update_pol)
            policy = torch.compile(self.policy)
            normalize_obs = torch.compile(self.obs_normalizer.forward)
            normalize_critic_obs = torch.compile(self.critic_obs_normalizer.forward)
        else:
            update_main = self._update_main
            update_pol = self._update_pol
            policy = self.policy
            normalize_obs = self.obs_normalizer.forward
            normalize_critic_obs = self.critic_obs_normalizer.forward
        qnet = self.qnet
        qnet_target = self.qnet_target
        env = self.env
        rb = self.rb
        #重置环境并获取初始观测，特别是对于critic_obs，因为critic网络需要特定格式的观测输入。
        # 将这些观测转换为适当的张量格式，并准备好进行后续的交互和学习更新。
        obs, critic_obs = env.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
        #初始化统计信息和进度条，准备进入主学习循环。在循环中，智能体将与环境进行交互，收集数据，并根据配置的更新频率进行学习更新。
        # 同时，还会记录各种训练指标，并在指定的间隔保存模型检查点。
        dones = None
        # Initialize metrics that might not be updated every step
        policy_entropy = torch.tensor(0.0, device=device)
        action_std = torch.tensor(0.0, device=device)
        actor_loss = torch.tensor(0.0, device=device)
        actor_grad_norm = torch.tensor(0.0, device=device)
        pbar = tqdm.tqdm(total=args.num_learning_iterations, initial=self.global_step)
        #主循环，在每次迭代中，智能体首先与环境进行交互，收集数据并存储在经验回放缓冲区中。
        # 然后，根据配置的更新频率和批量大小，从缓冲区中采样数据，并调用更新函数来优化actor和critic网络的参数。
        # 同时，还会进行目标网络的软更新，并记录各种训练指标以供分析和调试。
        
        #这个循环每走一步，global_step就会增加一次，直到达到预设的num_learning_iterations为止。
        # 在每次迭代中，智能体会与环境进行交互，收集数据，并根据配置的更新频率进行学习更新。
        # 同时，还会记录各种训练指标，并在指定的间隔保存模型检查点。
        while self.global_step <= args.num_learning_iterations:
            # Synchronize curriculum metrics across GPUs before rollout
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()
            # 使用 logging helper 来记录交互时间，这包括智能体与环境的交互、数据收集和存储等过程。
            # 通过在这个上下文管理器中执行交互代码，我们可以准确地测量和记录这些操作的时间，从而更好地分析和优化训练过程中的性能瓶颈。
            with self.logging_helper.record_collection_time():
                #用当前的策略选择动作，并与环境进行交互，获取下一个观测、奖励、done标志和其他信息。
                # 然后，将这些数据存储在经验回放缓冲区中，以供后续的学习更新使用。
                with torch.no_grad(), self._maybe_amp():
                    norm_obs = normalize_obs(obs, update=False)
                    actions = policy(obs=norm_obs, dones=dones)

                #真正让机器人动起来的地方！智能体根据当前的策略选择动作，并与环境进行交互，获取下一个观测、奖励、done标志和其他信息。
                next_obs, rewards, dones, infos = env.step(actions.float())
                #处理环境返回的truncation信息，这些信息指示了哪些环境实例由于达到时间限制而被截断。
                # 我们需要将这些信息存储在经验回放缓冲区中，以便在学习更新时正确处理这些情况。
                truncations = infos["time_outs"]
                # Update episode stats using logging helper
                self.logging_helper.update_episode_stats(rewards, dones, infos)

                next_critic_obs = infos["observations"]["critic"]

                # Compute 'true' next_obs and next_critic_obs for saving
                true_next_obs = torch.where(
                    truncations[:, None] > 0, infos["observations"]["final"]["actor_obs"], next_obs
                )
                true_next_critic_obs = torch.where(
                    truncations[:, None] > 0,
                    infos["observations"]["final"]["critic_obs"],
                    next_critic_obs,
                )
                #构造一个包含当前观测、动作、下一个观测、奖励、done标志和truncation信息的transition字典，
                # 并将其存储在经验回放缓冲区中。（也是强化学习里面最基础的数据）
                transition = TensorDict(
                    {
                        "observations": obs,
                        "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                        "next": {
                            "observations": true_next_obs,
                            "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float),
                            "truncations": truncations.long(),
                            "dones": dones.long(),
                        },
                    },
                    batch_size=(env.num_envs,),
                    device=device,
                )
                #额外地，我们还将critic_obs和next_critic_obs添加到transition字典中，以便在学习更新时使用。
                # 这些观测是专门为critic网络设计的，可能包含与actor观测不同的信息，因此需要单独存储和处理。
                transition["critic_observations"] = critic_obs
                transition["next"]["critic_observations"] = true_next_critic_obs
                #更新当前的观测和critic观测，以便在下一次迭代中使用。
                # 这些更新确保了智能体在与环境交互时能够正确地跟踪状态信息，并为后续的学习更新提供准确的数据。
                
                #值得注意的是，FsatSAC是一个off-policy算法，这意味着它在学习更新时使用的是经验回放缓冲区中存储的过去的交互数据，而不是当前的交互数据。
                obs = next_obs
                critic_obs = next_critic_obs

                rb.extend(transition)

            # 根据不同的更新频率和批量大小配置，计算每次学习更新需要使用的数据量（base_batch_size），并从经验回放缓冲区中采样数据进行学习更新。
            batch_size = max(args.batch_size // env.num_envs // self.gpu_world_size, 1)
            # 只有当global_step超过预设的learning_starts值时，才会进行学习更新。这是为了确保智能体在开始学习之前有足够的交互数据。
            if self.global_step > args.learning_starts:
                with self.logging_helper.record_learn_time():
                    #采样数据并准备批次进行学习更新。这里我们一次性采样一个大批次（batch_size * num_updates），然后将其分割成多个小批次，
                    # 以供每次更新使用。
                    prepared_batches = self._sample_and_prepare_batches(
                        batch_size, args.num_updates, normalize_obs, normalize_critic_obs
                    )
                    for i, data in enumerate(prepared_batches):
                        # Data is already normalized, just run the updates
                        (
                            buffer_rewards,
                            critic_grad_norm,
                            qf_loss,
                            qf_max,
                            qf_min,
                            alpha_loss,
                        ) = update_main(data)#更新critic网络的参数，并计算相关的训练指标，如critic梯度范数、critic损失、目标值的最大最小值等。
                        #按照配置的更新频率，更新actor网络的参数，并计算相关的训练指标，如actor梯度范数、actor损失、策略熵和动作标准差等。
                        if args.num_updates > 1:
                            if i % args.policy_frequency == 1:
                                actor_grad_norm, actor_loss, policy_entropy, action_std = update_pol(data)
                        elif self.global_step % args.policy_frequency == 0:
                            actor_grad_norm, actor_loss, policy_entropy, action_std = update_pol(data)

                        # 记录当前的训练指标到training_metrics中，以便后续的日志记录和分析。
                        current_metrics = {
                            "actor_loss": actor_loss,
                            "qf_loss": qf_loss,
                            "qf_max": qf_max,
                            "qf_min": qf_min,
                            "actor_grad_norm": actor_grad_norm,
                            "critic_grad_norm": critic_grad_norm,
                            "buffer_rewards": buffer_rewards,
                            "alpha_loss": alpha_loss,
                            "alpha_value": self.log_alpha.exp().detach().mean(),
                            "policy_entropy": policy_entropy,
                            "action_std": action_std,
                        }
                        self.training_metrics.add(current_metrics)
                        #软更新目标网络的参数，使用当前critic网络的参数和预设的tau值进行更新。
                        # 这种软更新有助于稳定训练过程，避免目标网络参数变化过快导致训练不稳定。
                        with torch.no_grad():
                            src_ps = [p.data for p in qnet.parameters()]
                            tgt_ps = [p.data for p in qnet_target.parameters()]
                            torch._foreach_mul_(tgt_ps, 1.0 - args.tau)
                            torch._foreach_add_(tgt_ps, src_ps, alpha=args.tau)
                #定期记录训练指标和保存模型检查点。根据预设的logging_interval和save_interval，智能体会定期记录当前的训练指标，并在指定的间隔保存模型检查点，以便后续的分析和恢复训练。
                if self.global_step % args.logging_interval == 0:
                    with torch.no_grad():
                        # Use accumulated training metrics for smoother logging (reduces noise)
                        accumulated_metrics = self.training_metrics.mean_and_clear()

                        # Convert tensor values to float for logging
                        loss_dict = {}
                        for key, value in accumulated_metrics.items():
                            if isinstance(value, torch.Tensor):
                                loss_dict[key] = value.item()
                            else:
                                loss_dict[key] = float(value)

                        # Add current env rewards (not part of training loop accumulation)
                        loss_dict["env_rewards"] = rewards.mean().item()

                    # Use logging helper
                    self.logging_helper.post_epoch_logging(it=self.global_step, loss_dict=loss_dict, extra_log_dicts={})
                    #定期保存模型检查点。根据预设的save_interval，智能体会定期保存当前的模型检查点，以便后续的分析和恢复训练。
                if args.save_interval > 0 and self.global_step > 0 and self.global_step % args.save_interval == 0:
                    if self.is_main_process:
                        logger.info(f"Saving model at global step {self.global_step}")
                        self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
                        self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

            # Avoid global_step being incremented beyond args.num_learning_iterations, so that the final checkpoint is
            # saved at exactly args.num_learning_iterations. In the `while` condition, we check for self.global_step <=
            # args.num_learning_iterations, so that we have complete logging data at the final step too (assuming
            # `args.num_learning_iterations` is a multiple of `args.logging_interval`).
            #如果global_step已经达到预设的num_learning_iterations，我们就跳出循环，确保最终的检查点是在正确的迭代次数保存的，并且日志记录也是完整的。
            if self.global_step >= args.num_learning_iterations:
                break
            self.global_step += 1
            pbar.update(1)
        # 在完成所有学习迭代后，智能体会进行最终的模型保存和ONNX导出，以便后续的分析、部署或恢复训练。
        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
            self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

    def save(self, path: str) -> None:  # type: ignore[override]
        env_state = self._collect_env_state()
        save_params(
            self.global_step,
            self.actor,
            self.qnet,
            self.qnet_target,
            self.log_alpha,
            self.obs_normalizer,
            self.critic_obs_normalizer,
            self.actor_optimizer,
            self.q_optimizer,
            self.alpha_optimizer,
            self.scaler,
            self.config,
            path,
            save_fn=self.logging_helper.save_checkpoint_artifact,
            env_state=env_state or None,
            metadata=self._checkpoint_metadata(iteration=self.global_step),
        )

    @torch.no_grad()
    def get_example_obs(self):
        """Used for exporting policy as onnx."""
        obs_dict = self.unwrapped_env.reset_all()
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return {
            "actor_obs": torch.cat([obs_dict[k] for k in self.config.actor_obs_keys], dim=1),
            "critic_obs": torch.cat([obs_dict[k] for k in self.config.critic_obs_keys], dim=1),
        }

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        device = device or self.device
        # Use the underlying module for inference
        policy = self.actor.to(device)
        obs_normalizer = self.obs_normalizer.to(device)
        policy.eval()
        obs_normalizer.eval()

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            if self.obs_normalization:
                normalized_obs = obs_normalizer(obs["actor_obs"], update=False)
            else:
                normalized_obs = obs["actor_obs"]
            # Actions are already scaled by the actor
            return policy(normalized_obs)[0]

        return policy_fn

    @property
    def actor_onnx_wrapper(self):
        # Use the underlying module for ONNX export
        actor = copy.deepcopy(self.actor).to("cpu")
        obs_normalizer = copy.deepcopy(self.obs_normalizer).to("cpu")

        class ActorWrapper(nn.Module):
            def __init__(self, actor, obs_normalizer):
                super().__init__()
                self.actor = actor
                self.obs_normalizer = obs_normalizer

            def forward(self, actor_obs):
                if self.obs_normalizer is not None:
                    normalized_obs = self.obs_normalizer(actor_obs, update=False)
                else:
                    normalized_obs = actor_obs
                # Actions are already scaled by the actor
                return self.actor(normalized_obs)[0]

        return ActorWrapper(actor, obs_normalizer if self.obs_normalization else None)

    def extract_actor_obs(self, obs: torch.Tensor, obs_key: str) -> torch.Tensor:
        """
        Extract a specific observation component from the flattened actor observation tensor.

        Args:
            obs: Flattened actor observation tensor of shape [batch_size, actor_obs_dim]
            obs_key: The observation key to extract (e.g., 'perception_obs', 'actor_state_obs')

        Returns:
            Extracted observation tensor of shape [batch_size, obs_size]
        """
        if obs_key not in self.actor_obs_indices:
            raise ValueError(
                f"Observation key '{obs_key}' not found in actor observations. "
                f"Available keys: {list(self.actor_obs_indices.keys())}"
            )

        indices = self.actor_obs_indices[obs_key]
        return obs[..., indices["start"] : indices["end"]]

    def extract_critic_obs(self, obs: torch.Tensor, obs_key: str) -> torch.Tensor:
        """
        Extract a specific observation component from the flattened critic observation tensor.

        Args:
            obs: Flattened critic observation tensor of shape [batch_size, critic_obs_dim]
            obs_key: The observation key to extract (e.g., 'perception_obs', 'critic_state_obs')

        Returns:
            Extracted observation tensor of shape [batch_size, obs_size]
        """
        if obs_key not in self.critic_obs_indices:
            raise ValueError(
                f"Observation key '{obs_key}' not found in critic observations. "
                f"Available keys: {list(self.critic_obs_indices.keys())}"
            )

        indices = self.critic_obs_indices[obs_key]
        return obs[..., indices["start"] : indices["end"]]

    def get_actor_obs_info(self) -> dict[str, dict[str, int]]:
        """
        Get information about actor observation indices.

        Returns:
            Dictionary with obs_key -> {'start': int, 'end': int, 'size': int}
        """
        return self.actor_obs_indices.copy()

    def get_critic_obs_info(self) -> dict[str, dict[str, int]]:
        """
        Get information about critic observation indices.

        Returns:
            Dictionary with obs_key -> {'start': int, 'end': int, 'size': int}
        """
        return self.critic_obs_indices.copy()

    def export(self, onnx_file_path: str) -> None:
        """Export the `.onnx` of the policy to & save it to `path`.

        This is intended to enable deployment, but not resuming training.
        For storing checkpoints to resume training, see `FastSACAgent.save()`
        """
        # Save current training state
        was_training = self.actor.training

        # Set model to evaluation mode for export so we don't affect gradients mid-rollout
        self.actor.eval()
        if self.obs_normalization:
            self.obs_normalizer.eval()

        # Create dummy all-zero input for ONNX tracing.
        example_input_list = torch.zeros(1, self.actor_obs_dim, device="cpu")

        motion_command = self.unwrapped_env.command_manager.get_state("motion_command")
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
                example_obs_dict={"actor_obs": example_input_list},
            )

        # Extract control gains and velocity limits & attach to onnx as metadata
        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.unwrapped_env)
        action_scales = getattr(self.unwrapped_env, "action_scales", None)
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
        metadata.update(self._checkpoint_metadata(iteration=self.global_step))

        attach_onnx_metadata(
            onnx_path=onnx_file_path,
            metadata=metadata,
        )

        self.logging_helper.save_to_wandb(onnx_file_path)

        # Restore original training state
        if was_training:
            self.actor.train()
            if self.obs_normalization:
                self.obs_normalizer.train()

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()

        obs = self.env.reset()

        for step in itertools.islice(itertools.count(), max_eval_steps):
            if self.obs_normalization:
                normalized_obs = self.obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs
            # Actions are already scaled by the actor
            actions = self.actor(normalized_obs)[0]

            actor_state = {"step": step, "actions": actions, "obs": obs}
            actor_state = self._pre_eval_env_step(actor_state)

            obs, _, _, _ = self.env.step(actor_state["actions"])
            actor_state["obs"] = obs
            actor_state = self._post_eval_env_step(actor_state)

        self._post_evaluate_policy()

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb_name in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb_name], training_loop=self))

    def _pre_evaluate_policy(self):
        self.env.set_is_evaluating()
        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict) -> dict:
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state: dict) -> dict:
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state
