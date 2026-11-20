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
from holosoma.agents.fast_sac.fast_sac import (
    Actor,
    CNNActor,
    CNNCritic,
    Critic,
)
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

    def step(self, actions: torch.Tensor, action_mean: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Actions are now already scaled by the actor, so pass them directly to the environment
        actor_state = {"actions": actions}
        if action_mean is not None:
            actor_state["action_mean"] = action_mean
        obs_dict, rew_buf, reset_buf, info_dict = self._env.step(actor_state)  # type: ignore[attr-defined]
        actor_obs = torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)
        critic_obs = torch.cat([obs_dict[k] for k in self._critic_obs_keys], dim=1)
        if "final_observations" in info_dict:
            # Use true final observations when available
            final_actor_obs = torch.cat([info_dict["final_observations"][k] for k in self._actor_obs_keys], dim=1)
            final_critic_obs = torch.cat([info_dict["final_observations"][k] for k in self._critic_obs_keys], dim=1)
        else:
            final_actor_obs = actor_obs
            final_critic_obs = critic_obs

        extras = {
            "time_outs": info_dict["time_outs"],
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

    def _compute_action_boundaries(self) -> torch.Tensor:
        """
        Compute per-policy-action scaling factors based on robot configuration.
        Returns tensor of shape (num_actions,) containing scaling for policy-controlled DOFs.

        The scaling factor is the maximum difference between default and joint limits,
        ensuring that action=0 corresponds to default position and action=±1 reaches
        the furthest limit from default.
        """
        robot_config = self._env.robot_config

        # Get joint limits and default positions
        dof_pos_lower_limits = torch.tensor(robot_config.dof_pos_lower_limit_list, device=self._env.device)
        dof_pos_upper_limits = torch.tensor(robot_config.dof_pos_upper_limit_list, device=self._env.device)

        # Get default joint angles
        default_joint_angles = torch.zeros(len(robot_config.dof_names), device=self._env.device)
        for i, joint_name in enumerate(robot_config.dof_names):
            if joint_name in robot_config.init_state.default_joint_angles:
                default_joint_angles[i] = robot_config.init_state.default_joint_angles[joint_name]

        # Get action scale from robot config
        action_scale = robot_config.control.action_scale

        # Compute maximum range from default to either limit for each joint
        # This ensures symmetric scaling where action=0 -> default position
        range_to_lower = torch.abs(dof_pos_lower_limits - default_joint_angles)
        range_to_upper = torch.abs(dof_pos_upper_limits - default_joint_angles)
        max_range = torch.maximum(range_to_lower, range_to_upper)

        # Account for action_scale: the environment applies actions_scaled = actions * action_scale
        # So our scaling factor should be: max_range / action_scale
        action_scaling_factors = max_range / action_scale
        policy_dof_indices = self._get_policy_dof_indices()
        policy_scaling_factors = action_scaling_factors[policy_dof_indices]

        logger.info(f"Computed action scaling factors for {policy_scaling_factors.numel()} policy actions")
        logger.info(f"Action scale: {action_scale}")
        logger.info(f"Policy DOF indices: {policy_dof_indices.tolist()}")
        logger.info(f"Scaling: {policy_scaling_factors}")

        return policy_scaling_factors

    def _get_policy_dof_indices(self) -> torch.Tensor:
        """Resolve robot DOF indices controlled by policy actions.

        Raises:
            ValueError: If ``policy_dim != num_dof`` and no action term
                provides ``policy_dof_indices``, or if the provided indices
                have a length that does not match ``policy_dim``.
        """
        action_manager = getattr(self._env, "action_manager", None)
        policy_dim = self._env.dim_actions
        device = self._env.device

        if action_manager is None:
            return torch.arange(policy_dim, device=device, dtype=torch.long)

        for _, term in action_manager.iter_terms():
            policy_dof_indices = getattr(term, "policy_dof_indices", None)
            if policy_dof_indices is None:
                continue
            policy_dof_indices = policy_dof_indices.to(device=device, dtype=torch.long)
            if policy_dof_indices.numel() != policy_dim:
                raise ValueError(
                    "Action term policy_dof_indices size must match policy action size: "
                    f"len(policy_dof_indices)={policy_dof_indices.numel()}, actions_dim={policy_dim}"
                )
            return policy_dof_indices

        if policy_dim == self._env.num_dof:
            return torch.arange(self._env.num_dof, device=device, dtype=torch.long)

        raise ValueError(
            f"policy_dim ({policy_dim}) differs from num_dof ({self._env.num_dof}) "
            "but no action term provides policy_dof_indices. "
            "Reduced-action-space configurations require explicit policy DOF indices."
        )


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

    def setup(self) -> None:
        logger.info("Setting up FastSAC")

        # Log curriculum synchronization status for multi-GPU training
        if self.is_multi_gpu:
            if self.has_curricula_enabled():
                logger.info(f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs")

        args = self.config
        device = self.device
        env = self.env

        if args.use_symmetry_loss and not args.use_symmetry:
            raise ValueError(
                "use_symmetry_loss=True requires use_symmetry=True. "
                "The symmetry regularization loss is only meaningful alongside the "
                "mirrored data-doubling augmentation. Set use_symmetry=True to enable "
                "use_symmetry_loss, or set use_symmetry_loss=False. "
                "use_symmetry_loss is never auto-derived from use_symmetry."
            )
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

        if args.obs_normalization:
            for normalizer, terms, keys, indices, dim in (
                (self.obs_normalizer, args.fixed_scale_actor_obs_terms, args.actor_obs_keys,
                 self.actor_obs_indices, actor_obs_dim),
                (self.critic_obs_normalizer, args.fixed_scale_critic_obs_terms, critic_obs_keys,
                 self.critic_obs_indices, critic_obs_dim),
            ):
                if not terms:
                    continue
                slices = self._obs_term_slices(keys, indices)
                missing = set(terms) - set(slices)
                if missing:
                    raise ValueError(f"fixed-scale terms not found in the observation: {sorted(missing)}")
                mask = torch.zeros(dim, dtype=torch.bool, device=device)
                for t in terms:
                    for a, b in slices[t]:
                        mask[a:b] = True
                normalizer.fix_dims(mask)
                logger.info(f"Fixed-scale (not running-normalized) obs terms {list(terms)}: {int(mask.sum())} dims")

        # Get action scaling parameters from the environment
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

        if args.use_cnn_encoder:
            actor_cls, critic_cls = CNNActor, CNNCritic
        else:
            actor_cls, critic_cls = Actor, Critic

        actor_kwargs: dict[str, Any] = dict(
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
        self.actor = actor_cls(**actor_kwargs)
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

        if args.use_symmetry or args.use_symmetry_loss:
            # using env._env is not really ideal..
            # The `or use_symmetry_loss` clause is redundant in practice (the
            # Requirement 10 validation above guarantees use_symmetry=True
            # whenever use_symmetry_loss=True) but is kept as harmless
            # defense-in-depth against future refactors.
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

    def _mean_over_ranks(self, log_dict: dict[str, Any], key: str) -> dict[str, Any]:
        """``log_dict`` with ``key`` replaced by its mean over all ranks.

        Each rank's metric covers only its own environments, so checkers fed
        local values can disagree, and a rank that stops alone leaves the others
        waiting in the next all-reduce. Fed the same mean, every rank's checker
        makes the same decision on the same iteration, from all environments as
        on one GPU. Every rank joins the all-reduce even when it lacks the key;
        the key is kept only if every rank has it.
        """
        value = log_dict.get(key)
        present = value is not None
        packed = torch.tensor(
            [float(value) if present else 0.0, 1.0 if present else 0.0], device=self.device, dtype=torch.float64
        )
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        out = {k: v for k, v in log_dict.items() if k != key}
        if int(packed[1].item()) == self.gpu_world_size:
            out[key] = float(packed[0].item()) / self.gpu_world_size
        return out

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

            # Symmetry critic invariance regularizer (gated solely on
            # use_symmetry_loss). Computed inside the same autocast block so its
            # dtype/grad-scaling matches the base critic loss. The raw term is
            # always computed under the flag (even when the coef is 0.0) so it
            # can be logged for calibration.
            if args.use_symmetry_loss:
                sym_critic_loss = self._compute_symmetry_critic_loss(critic_observations, actions)
                qf_loss = qf_loss + args.symmetry_critic_coef * sym_critic_loss
            else:
                sym_critic_loss = torch.zeros((), device=self.device)

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
            sym_critic_loss.detach(),
        )

    def _compute_symmetry_actor_loss(self, observations: torch.Tensor) -> torch.Tensor:
        """Actor equivariance regularizer: MSE( actor_mean(M·o), M·actor_mean(o) ).

        Encourages the deterministic policy to be equivariant under the robot's
        sagittal-plane mirror ``M`` — mirroring the observation and then running
        the policy should equal running the policy and then mirroring the action.

        Gradients flow through BOTH branches (no detach on the mirrored/target
        side), matching the PPO reference. The only ``.detach()`` is a defensive
        one on the observation INPUT used to build the mirror pair: FastSAC's
        minibatch obs are already detached replay-buffer tensors, so this is a
        no-op today, but it makes the intent explicit and is robust to future
        encoder-in-the-loop changes.
        """
        assert hasattr(self, "symmetry_utils"), (
            "use_symmetry_loss requires symmetry_utils to be initialized. "
            "symmetry_utils is created in setup() when use_symmetry or "
            "use_symmetry_loss is True; ensure the agent was configured with "
            "use_symmetry=True (the required prerequisite for use_symmetry_loss)."
        )
        # Deterministic mean action = Actor.forward(...)[0] (tanh-squashed, scaled).
        mean_action = self.actor(observations)[0]
        mirror_obs = self.symmetry_utils.mirror_xz_plane(observations.detach(), self.env, self.config.actor_obs_keys)
        mean_action_on_mirror = self.actor(mirror_obs)[0]
        mirror_of_mean_action = self.symmetry_utils.mirror_action_xz_plane(mean_action)
        # No detach on the target side: gradients flow through both branches (matches PPO).
        return F.mse_loss(mean_action_on_mirror, mirror_of_mean_action)

    def _compute_symmetry_critic_loss(
        self, critic_observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Critic invariance regularizer: MSE( value(o, a), value(M·o, M·a) ).

        Encourages the distributional critic's SCALAR value estimate to be
        invariant under the robot's sagittal-plane mirror ``M`` — mirroring the
        critic observation and the action should leave the scalar value
        unchanged. The loss is computed on the scalar ``get_value`` output
        (expected value over the softmaxed atom distribution), NOT on the raw
        atom logits, matching the value path already used in ``_update_pol``.

        Gradients flow through BOTH branches (no detach on the mirrored side),
        matching the PPO reference. The only ``.detach()`` is a defensive one on
        the observation/action INPUTS used to build the mirror pair: FastSAC's
        minibatch tensors are already detached replay-buffer tensors, so this is
        a no-op today, but it makes the intent explicit and is robust to future
        encoder-in-the-loop changes.
        """
        assert hasattr(self, "symmetry_utils"), (
            "use_symmetry_loss requires symmetry_utils to be initialized. "
            "symmetry_utils is created in setup() when use_symmetry or "
            "use_symmetry_loss is True; ensure the agent was configured with "
            "use_symmetry=True (the required prerequisite for use_symmetry_loss)."
        )
        qnet = self.qnet
        # Scalar value on the original (obs, action) pair. get_value returns a
        # [num_q_networks, batch] tensor, so the MSE averages across both critics.
        value = qnet.get_value(F.softmax(qnet(critic_observations, actions), dim=-1))
        # Mirror the critic obs (critic keys) and the actions to build the pair.
        mirror_cobs = self.symmetry_utils.mirror_xz_plane(
            critic_observations.detach(), self.env, self.config.critic_obs_keys
        )
        mirror_actions = self.symmetry_utils.mirror_action_xz_plane(actions.detach())
        value_mirror = qnet.get_value(F.softmax(qnet(mirror_cobs, mirror_actions), dim=-1))
        # No detach on the mirrored side: gradients flow through both branches (matches PPO).
        return F.mse_loss(value, value_mirror)

    def _update_pol(
        self, data: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

            # Symmetry actor equivariance regularizer (gated solely on
            # use_symmetry_loss). Computed inside the same autocast block so its
            # dtype/grad-scaling matches the base actor loss.
            if args.use_symmetry_loss:
                sym_actor_loss = self._compute_symmetry_actor_loss(data["observations"])
                actor_loss = actor_loss + args.symmetry_actor_coef * sym_actor_loss
            else:
                sym_actor_loss = torch.zeros((), device=self.device)

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
            sym_actor_loss.detach(),
        )

    def _obs_term_slices(self, obs_keys, obs_indices) -> dict[str, list[tuple[int, int]]]:
        """Where each observation term sits in a concatenated observation vector.

        The ObservationManager concatenates a group's enabled terms in sorted
        order, each term stacked over the group's history (term-major), so a
        term occupies one contiguous block of ``term_dim * history_length``.
        """
        return {
            name: [(start, start + dim * history) for start, dim, history, _ in blocks]
            for name, blocks in self._obs_term_layout(obs_keys, obs_indices).items()
        }

    def _obs_term_layout(self, obs_keys, obs_indices) -> dict[str, list[tuple[int, int, int, str]]]:
        """Per term: (block start, per-frame dim, history length, group) for each group holding it.

        Within a block, frames are oldest first, so the current frame is the
        last ``dim`` entries.
        """
        obs_manager = self.env.observation_manager
        out: dict[str, list[tuple[int, int, int, str]]] = {}
        for key in obs_keys:
            group_cfg = obs_manager.cfg.groups[key]
            history = max(int(getattr(group_cfg, "history_length", 1) or 1), 1)
            offset = obs_indices[key]["start"]
            for name in sorted(n for n, t in group_cfg.terms.items() if t.enabled):
                dim = obs_manager._compute_term(key, name, group_cfg.terms[name]).shape[1]
                out.setdefault(name, []).append((offset, dim, history, key))
                offset += dim * history
            if offset != obs_indices[key]["end"]:
                raise RuntimeError(
                    f"obs group {key!r}: terms span {offset - obs_indices[key]['start']} dims, "
                    f"the group is {obs_indices[key]['size']}"
                )
        return out

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

    def _iter_prepared_batches(
        self, batch_size: int, num_updates: int, normalize_obs, normalize_critic_obs
    ):
        """Yield the ``num_updates`` batches of one learning step.

        With ``sample_chunk_updates`` off (0) this is exactly
        ``_sample_and_prepare_batches`` -- one draw for all updates. When on,
        it draws ``sample_chunk_updates`` updates' worth at a time, so only one
        chunk's block is alive at once. The caller must drop its reference to
        each batch before advancing, or the previous chunk stays alive through
        that view while the next one is drawn.
        """
        chunk = self.config.sample_chunk_updates
        prepare = lambda n: self._sample_and_prepare_batches(  # noqa: E731
            batch_size, n, normalize_obs, normalize_critic_obs
        )
        if self.config.use_symmetry:
            # One update per draw. A draw is mirrored as one block,
            # [originals | mirrors], and split into updates, so a draw of
            # several updates would give each update originals only or mirrors
            # only. Mirrored samples are only as good as the physics' mirror
            # symmetry, so every update gets a one-update draw: [originals | mirrors].
            chunk = 1
        if chunk <= 0 or chunk >= num_updates:
            yield from prepare(num_updates)
            return
        remaining = num_updates
        while remaining > 0:
            n = min(chunk, remaining)
            batches = prepare(n)
            while batches:
                yield batches.pop(0)
            remaining -= n

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

    def learn(self) -> None:
        args = self.config
        device = self.device
        if args.compile:
            # Compiling _update_main destabilizes multi-GPU training when the
            # compile cache is cold (mainline holosoma, #93).
            update_main = self._update_main if self.is_multi_gpu else torch.compile(self._update_main)
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

        obs, critic_obs = env.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)

        dones = None
        # Initialize metrics that might not be updated every step
        policy_entropy = torch.tensor(0.0, device=device)
        action_std = torch.tensor(0.0, device=device)
        actor_loss = torch.tensor(0.0, device=device)
        actor_grad_norm = torch.tensor(0.0, device=device)
        # sym_actor_loss is produced by the policy-frequency actor update; seed
        # it so logging has a bound value on iterations before the first update.
        sym_actor_loss = torch.tensor(0.0, device=device)
        pbar = tqdm.tqdm(total=args.num_learning_iterations, initial=self.global_step)

        while self.global_step <= args.num_learning_iterations:
            # Synchronize curriculum metrics across GPUs before rollout
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            with self.logging_helper.record_collection_time():
                with torch.no_grad(), self._maybe_amp():
                    norm_obs = normalize_obs(obs, update=False)
                    actions = policy(obs=norm_obs, dones=dones)

                # Pass action mean to env for mean-action penalty terms
                action_mean = getattr(self.actor, "_cached_action_mean", None)
                next_obs, rewards, dones, infos = env.step(actions.float(), action_mean=action_mean)
                truncations = infos["time_outs"]

                # Update episode stats using logging helper
                self.logging_helper.update_episode_stats(rewards, dones, infos)

                next_critic_obs = infos["observations"]["critic"]

                # Store the true next observation: on a time-out the env has already
                # reset, so take the pre-reset final observation instead.
                true_next_obs = torch.where(
                    truncations[:, None] > 0, infos["observations"]["final"]["actor_obs"], next_obs
                )
                true_next_critic_obs = torch.where(
                    truncations[:, None] > 0,
                    infos["observations"]["final"]["critic_obs"],
                    next_critic_obs,
                )
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
                transition["critic_observations"] = critic_obs
                transition["next"]["critic_observations"] = true_next_critic_obs

                obs = next_obs
                critic_obs = next_critic_obs

                rb.extend(transition)

            # NOTE: args.batch_size is the global batch size
            batch_size = max(args.batch_size // env.num_envs // self.gpu_world_size, 1)
            # Guard: need both (a) enough total training steps and (b) enough
            # transitions in the current buffer. After a phase transition the
            # buffer is empty even though global_step is large from the checkpoint.
            buffer_count = min(rb.ptr, rb.buffer_size)
            if self.global_step > args.learning_starts and buffer_count > args.learning_starts:
                with self.logging_helper.record_learn_time():
                    # Batched sampling: sample, normalize and split into updates,
                    # all at once or in chunks (sample_chunk_updates).
                    prepared_batches = self._iter_prepared_batches(
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
                            sym_critic_loss,
                        ) = update_main(data)
                        if args.num_updates > 1:
                            if i % args.policy_frequency == 1:
                                actor_grad_norm, actor_loss, policy_entropy, action_std, sym_actor_loss = update_pol(data)
                        elif self.global_step % args.policy_frequency == 0:
                            actor_grad_norm, actor_loss, policy_entropy, action_std, sym_actor_loss = update_pol(data)

                        # Accumulate training metrics for smoother logging
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
                        # Symmetry-loss metrics: log raw unweighted terms plus
                        # their weighted contributions so the coefficients can be
                        # calibrated empirically. Added only when the flag is on,
                        # so the logged metric set is unchanged for existing runs.
                        # sym_actor_loss (like actor_loss/action_std above) is
                        # produced by the policy-frequency actor update and
                        # persists across iterations between updates.
                        if args.use_symmetry_loss:
                            current_metrics["symmetry_actor_loss"] = sym_actor_loss
                            current_metrics["symmetry_actor_loss_weighted"] = (
                                args.symmetry_actor_coef * sym_actor_loss
                            )
                            current_metrics["symmetry_critic_loss"] = sym_critic_loss
                            current_metrics["symmetry_critic_loss_weighted"] = (
                                args.symmetry_critic_coef * sym_critic_loss
                            )
                        self.training_metrics.add(current_metrics)

                        with torch.no_grad():
                            src_ps = [p.data for p in qnet.parameters()]
                            tgt_ps = [p.data for p in qnet_target.parameters()]
                            torch._foreach_mul_(tgt_ps, 1.0 - args.tau)
                            torch._foreach_add_(tgt_ps, src_ps, alpha=args.tau)

                        # Release this batch before the generator draws the next
                        # chunk: it is a view into its chunk's block, and would
                        # otherwise keep that whole block alive alongside the new one.
                        del data

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
                if args.save_interval > 0 and self.global_step > 0 and self.global_step % args.save_interval == 0:
                    if self.is_main_process:
                        logger.info(f"Saving model at global step {self.global_step}")
                        self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
                        self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

            # --- Sigma scheduler: update reward params based on progress ---
            if hasattr(self, "_sigma_scheduler") and self._sigma_scheduler is not None:
                self._sigma_scheduler.update(self.global_step)

            # --- Early stop: check graduation condition ---
            if hasattr(self, "_early_stop_checker") and self._early_stop_checker is not None:
                env_log_dict = getattr(env, "log_dict", {})
                if self.is_multi_gpu:
                    env_log_dict = self._mean_over_ranks(env_log_dict, self._early_stop_checker.cfg.metric)
                if self._early_stop_checker.check(self.global_step, env_log_dict):
                    # Save checkpoint before exiting
                    if self.is_main_process:
                        logger.info(f"[EarlyStop] Saving final checkpoint at step {self.global_step}")
                        self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
                        self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))
                    break

            # Avoid global_step being incremented beyond args.num_learning_iterations, so that the final checkpoint is
            # saved at exactly args.num_learning_iterations. In the `while` condition, we check for self.global_step <=
            # args.num_learning_iterations, so that we have complete logging data at the final step too (assuming
            # `args.num_learning_iterations` is a multiple of `args.logging_interval`).
            if self.global_step >= args.num_learning_iterations:
                break
            self.global_step += 1

            pbar.update(1)

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
        # Extract URDF text from the robot config
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

        default_dof_pos = self.unwrapped_env.default_dof_pos[0].tolist()

        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "default_dof_pos": default_dof_pos,
            "kp": kp_list,
            "kd": kd_list,
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
        metric_keys = (
            "ee_pos_error",
            "ee_rot_error",
            "ee_pos_error_left",
            "ee_pos_error_right",
            "ee_rot_error_left",
            "ee_rot_error_right",
        )
        metric_trajectory: list[list[float]] = []

        obs = self.env.reset()

        for _ in itertools.islice(itertools.count(), max_eval_steps):
            if self.obs_normalization:
                normalized_obs = self.obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs
            # Actions are already scaled by the actor
            actions = self.actor(normalized_obs)[0]
            obs, _, _, extras = self.env.step(actions)

            to_log = extras.get("to_log", {})
            step_values: list[float] = []
            for key in metric_keys:
                metric_value = float("nan")
                if key in to_log:
                    raw_value = to_log[key]
                    if isinstance(raw_value, torch.Tensor):
                        if raw_value.numel() > 0:
                            metric_value = float(raw_value.detach().float().mean().item())
                    else:
                        metric_value = float(raw_value)

                    if not math.isfinite(metric_value):
                        metric_value = float("nan")
                step_values.append(metric_value)
            metric_trajectory.append(step_values)

        if len(metric_trajectory) == 0:
            return torch.empty((0, len(metric_keys)), dtype=torch.float32)
        return torch.tensor(metric_trajectory, dtype=torch.float32)
