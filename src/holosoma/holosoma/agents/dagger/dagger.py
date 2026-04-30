"""Pure DAgger distillation: student rollout + teacher action targets, no critic.

The student actor is the only learnable network. The teacher is loaded from a
PPO checkpoint, frozen, and queried each rollout step on its own observation
slice. The actor loss is a behaviour-cloning loss on action mean/std.

This mirrors VideoMimic's stage-3 DAgger (with ``bc_loss_coef=1.0``) but drops
the critic entirely — there is no value loss, no GAE, no advantage normalisation.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.modules.data_utils import RolloutStorage
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.config_types.algo import DAggerConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_motion_and_policy_as_onnx,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)


class DAgger(BaseAlgo):
    """Pure DAgger trainer (student actor only, frozen teacher).

    The student observes a (typically smaller / VideoMimic-style) actor obs
    plus optional perception (heightmap). The teacher observes its original
    obs slice via the env's ``teacher_actor_obs`` group plus shared perception.
    """

    config: DAggerConfig

    def __init__(
        self,
        env: BaseTask,
        config: DAggerConfig,
        log_dir,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ):
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

        self.algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()
        groups = self.env.observation_manager.cfg.groups
        if "actor_obs" not in groups:
            raise RuntimeError("DAgger requires an 'actor_obs' group in the observation config.")
        if self.config.teacher_actor_obs_key not in groups:
            raise RuntimeError(
                f"DAgger requires the teacher obs group "
                f"'{self.config.teacher_actor_obs_key}' in the observation config."
            )

        self.num_act = self.env.robot_config.actions_dim
        self.actor_learning_rate = self.config.actor_learning_rate
        self.actor_obs_keys = self.config.module_dict.actor.input_dim
        self.current_learning_iteration = 0
        self.eval_callbacks: list = []

        _ = self.env.reset_all()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def setup(self):
        logger.info("Setting up DAgger student")
        self._setup_student()
        # Skip teacher loading when no checkpoint is supplied (e.g. running
        # eval_agent on a trained student — the teacher is irrelevant there).
        ckpt = self.config.policy_to_clone
        if ckpt and Path(ckpt).expanduser().exists():
            logger.info("Loading teacher")
            self._setup_teacher()
        else:
            logger.info(
                "Skipping teacher load (policy_to_clone is empty or missing) — "
                "student-only mode (eval / inference)."
            )
            self.teacher_actor = None
            self.teacher_use_motion_encoder = False
            self.teacher_use_height_map = False
        logger.info("Setting up rollout storage")
        self._setup_storage()

    def _setup_student(self):
        student_cfg = self.config.module_dict
        height_map_enc_cfg = self._height_map_encoder_config(student_cfg.height_map_encoder)

        history_length = {
            "actor_obs": self.env.observation_manager.cfg.groups["actor_obs"].history_length,
            # critic_obs not used by DAgger but module_utils may probe the dict;
            # fall back to actor_obs's value when there is no critic group.
            "critic_obs": self.env.observation_manager.cfg.groups.get(
                "critic_obs", self.env.observation_manager.cfg.groups["actor_obs"]
            ).history_length,
        }

        self.actor = setup_ppo_actor_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=student_cfg.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            device=self.device,
            history_length=history_length,
            motion_encoder_config=None,
            height_map_encoder_config=height_map_enc_cfg,
        )
        self.actor_optimizer = instantiate(
            self.config.actor_optimizer, params=self.actor.parameters(), lr=self.actor_learning_rate
        )

        # Student never sees future motion — distillation point is precisely
        # that the student must do without it. Only the heightmap branch is
        # gated by config.
        actor_type = student_cfg.actor.type
        self.student_use_height_map = (
            actor_type in ("MLPWithHeightMap", "MLPWithHeightMapVideoMimic")
            and student_cfg.height_map_encoder is not None
            and "height_map_obs" in self.algo_obs_dim_dict
        )
        # Alias mirroring PPO's API so eval_agent.py / inference helpers that
        # introspect `use_height_map` work transparently.
        self.use_height_map = self.student_use_height_map

        if self.is_multi_gpu:
            for param in self.actor.parameters():
                torch.distributed.broadcast(param.data, src=0)

    def _setup_teacher(self):
        ckpt_path = self.config.policy_to_clone
        if not ckpt_path:
            raise ValueError("DAggerConfig.policy_to_clone is required (path to teacher .pt).")

        teacher_exp_cfg, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=ckpt_path))
        teacher_algo_cfg = teacher_exp_cfg.algo.config
        teacher_module_dict = teacher_algo_cfg.module_dict

        # Remap the env's teacher obs group(s) into the keys the teacher's
        # PPO modules expect ("actor_obs", "future_motion_targets",
        # "height_map_obs"). The teacher checkpoint was trained with those
        # canonical names.
        teacher_obs_dim_dict = dict(self.algo_obs_dim_dict)
        teacher_actor_dim = self.algo_obs_dim_dict[self.config.teacher_actor_obs_key]
        teacher_obs_dim_dict["actor_obs"] = teacher_actor_dim

        teacher_history_key = self.config.teacher_actor_obs_history_key or self.config.teacher_actor_obs_key
        teacher_history_length = {
            "actor_obs": self.env.observation_manager.cfg.groups[teacher_history_key].history_length,
            "critic_obs": self.env.observation_manager.cfg.groups.get(
                "critic_obs", self.env.observation_manager.cfg.groups[teacher_history_key]
            ).history_length,
        }

        teacher_motion_enc_cfg = self._teacher_motion_encoder_config(teacher_module_dict)
        teacher_height_map_enc_cfg = self._height_map_encoder_config(teacher_module_dict.height_map_encoder)

        teacher_actor = setup_ppo_actor_module(
            obs_dim_dict=teacher_obs_dim_dict,
            module_config=teacher_module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=teacher_algo_cfg.init_noise_std,
            device=self.device,
            history_length=teacher_history_length,
            motion_encoder_config=teacher_motion_enc_cfg,
            height_map_encoder_config=teacher_height_map_enc_cfg,
        )

        loaded = torch.load(ckpt_path, map_location=self.device)
        teacher_actor.load_state_dict(loaded["actor_model_state_dict"])
        teacher_actor.eval()
        for p in teacher_actor.parameters():
            p.requires_grad = False

        self.teacher_actor = teacher_actor
        teacher_actor_type = teacher_module_dict.actor.type
        self.teacher_use_motion_encoder = (
            teacher_actor_type in ("MLPWithMotionEncoder", "MLPWithHeightMap", "MLPWithHeightMapVideoMimic")
            and teacher_module_dict.motion_encoder is not None
            and "future_motion_targets" in self.algo_obs_dim_dict
        )
        self.teacher_use_height_map = (
            teacher_actor_type in ("MLPWithHeightMap", "MLPWithHeightMapVideoMimic")
            and teacher_module_dict.height_map_encoder is not None
            and "height_map_obs" in self.algo_obs_dim_dict
        )
        logger.info(
            f"Teacher loaded from {ckpt_path} "
            f"(actor_type={teacher_actor_type}, motion_encoder={self.teacher_use_motion_encoder}, "
            f"height_map={self.teacher_use_height_map})"
        )

    def _teacher_motion_encoder_config(self, teacher_module_dict):
        me_cfg = teacher_module_dict.motion_encoder
        if me_cfg is None:
            return None
        input_dim = me_cfg.input_dim
        if input_dim == 0 and "future_motion_targets" in self.algo_obs_dim_dict:
            total_dim = self.algo_obs_dim_dict["future_motion_targets"]
            input_dim = total_dim // me_cfg.num_timesteps
        return {
            "input_dim": input_dim,
            "hidden_dim": me_cfg.hidden_dim,
            "output_dim": me_cfg.output_dim,
            "num_timesteps": me_cfg.num_timesteps,
            "activation": me_cfg.activation,
        }

    @staticmethod
    def _height_map_encoder_config(hm_cfg):
        if hm_cfg is None:
            return None
        return {
            "latent_dim": hm_cfg.latent_dim,
            "num_heads": hm_cfg.num_heads,
            "map_height": hm_cfg.map_height,
            "map_width": hm_cfg.map_width,
            "num_channels": hm_cfg.num_channels,
            "conv_hidden_channels": hm_cfg.conv_hidden_channels,
            "combine_mode": getattr(hm_cfg, "combine_mode", "concat"),
        }

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.config.num_steps_per_env, device=self.device)

        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        self.storage.register("actor_obs", shape=(actor_obs_dim,), dtype=torch.float)

        if self.student_use_height_map:
            self.storage.register(
                "height_map_obs",
                shape=(self.algo_obs_dim_dict["height_map_obs"],),
                dtype=torch.float,
            )

        self.storage.register("teacher_action_mean", shape=(self.num_act,), dtype=torch.float)
        self.storage.register("teacher_action_sigma", shape=(self.num_act,), dtype=torch.float)
        self.storage.register("dones", shape=(1,), dtype=torch.bool)

    def _get_obs_dim(self, obs_keys: list[str]) -> int:
        obs_dim = 0
        for k in obs_keys:
            obs_dim += self.algo_obs_dim_dict[k]
        return obs_dim

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def learn(self):
        self.actor.train()
        obs_dict = self.env.reset_all()

        if self.config.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        for k in obs_dict:
            obs_dict[k] = obs_dict[k].to(self.device)

        for it in range(
            self.current_learning_iteration,
            self.current_learning_iteration + self.config.num_learning_iterations,
        ):
            self.current_learning_iteration = it

            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            with self.logging_helper.record_collection_time():
                obs_dict = self._rollout_step(obs_dict)

            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()

            eval_metrics = {}
            if (
                self.config.eval_interval > 0
                and it % self.config.eval_interval == 0
                and it > 0
                and self.is_main_process
            ):
                eval_metrics, obs_dict = self._evaluate_during_training()

            if self.is_main_process:
                self._post_epoch_logging(it, loss_dict, eval_metrics=eval_metrics)

            if it % self.config.save_interval == 0 and self.is_main_process:
                self.save(os.path.join(self.log_dir, f"model_{it:05d}.pt"))
                self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{it:05d}.onnx"))

        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"))
            self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.onnx"))

    def _build_student_state(self, obs_dict):
        actor_obs = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
        state = {"actor_obs": actor_obs}
        if self.student_use_height_map:
            state["height_map_obs"] = obs_dict["height_map_obs"]
        return state, actor_obs

    def _build_teacher_state(self, obs_dict):
        state = {"actor_obs": obs_dict[self.config.teacher_actor_obs_key]}
        if self.teacher_use_height_map:
            state["height_map_obs"] = obs_dict["height_map_obs"]
        if self.teacher_use_motion_encoder:
            state["future_motion_targets"] = obs_dict["future_motion_targets"]
        return state

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                student_state, actor_obs = self._build_student_state(obs_dict)
                student_actions = self.actor.act(student_state)

                teacher_state = self._build_teacher_state(obs_dict)
                self.teacher_actor.act(teacher_state)
                teacher_mu = self.teacher_actor.action_mean.detach()
                teacher_sigma = self.teacher_actor.action_std.detach()

                actions_to_step = teacher_mu if self.config.take_teacher_actions else student_actions

                obs_dict, rewards, dones, infos = self.env.step({"actions": actions_to_step})
                for k in obs_dict:
                    obs_dict[k] = obs_dict[k].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                storage_data = {
                    "actor_obs": actor_obs,
                    "teacher_action_mean": teacher_mu,
                    "teacher_action_sigma": teacher_sigma,
                    "dones": dones.view(-1, 1),
                }
                if self.student_use_height_map:
                    storage_data["height_map_obs"] = student_state["height_map_obs"]
                self.storage.add(**storage_data)

                self.actor.reset(dones)

                if self.log_dir is not None:
                    self.logging_helper.update_episode_stats(rewards, dones, infos)
        return obs_dict

    def _training_step(self) -> dict[str, float]:
        loss_dict = {"BC": 0.0, "BC_mu": 0.0, "BC_sigma": 0.0}
        generator = self.storage.mini_batch_generator(self.config.num_mini_batches, self.config.num_learning_epochs)
        for minibatch in generator:
            losses = self._compute_bc_loss(minibatch)

            self.actor_optimizer.zero_grad()
            losses["bc_loss"].backward()

            if self.is_multi_gpu:
                self._reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()

            loss_dict["BC"] += losses["bc_loss"].item()
            loss_dict["BC_mu"] += losses["mu_loss"].item()
            loss_dict["BC_sigma"] += losses["sigma_loss"].item()

        num_updates = self.config.num_learning_epochs * self.config.num_mini_batches
        for k in loss_dict:
            loss_dict[k] /= num_updates
        self.storage.clear()
        return loss_dict

    def _compute_bc_loss(self, minibatch: dict[str, torch.Tensor]):
        teacher_mu = minibatch["teacher_action_mean"]
        teacher_sigma = minibatch["teacher_action_sigma"]

        if self.config.clip_teacher_actions:
            t = self.config.clip_actions_threshold
            teacher_mu = torch.clamp(teacher_mu, -t, t)

        state = {"actor_obs": minibatch["actor_obs"]}
        if self.student_use_height_map and "height_map_obs" in minibatch:
            state["height_map_obs"] = minibatch["height_map_obs"]

        # Run actor to populate distribution; sampled action itself is unused.
        self.actor.act(state)
        mu_student = self.actor.action_mean
        sigma_student = self.actor.action_std

        mu_loss = (teacher_mu - mu_student).pow(2).sum(dim=-1).mean()
        sigma_loss = (sigma_student - teacher_sigma).pow(2).sum(dim=-1).mean()

        bc_loss = mu_loss + self.config.bc_sigma_loss_coef * sigma_loss
        return {"bc_loss": bc_loss, "mu_loss": mu_loss, "sigma_loss": sigma_loss}

    def _reduce_parameters(self):
        grads = [p.grad.view(-1) for p in self.actor.parameters() if p.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in self.actor.parameters():
            if p.grad is not None:
                numel = p.numel()
                p.grad.data.copy_(all_grads[offset : offset + numel].view_as(p.grad))
                offset += numel

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _post_epoch_logging(self, it, loss_dict, eval_metrics=None):
        extra_log_dicts = {
            "Policy": {
                "mean_noise_std": self.actor.std.mean().item(),
                "actor_learning_rate": self.actor_learning_rate,
            },
        }
        if eval_metrics:
            extra_log_dicts["_flat_eval_metrics"] = eval_metrics
        loss_dict["actor_learning_rate"] = self.actor_learning_rate
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save(self, path, infos=None):
        ckpt = {
            "actor_model_state_dict": self.actor.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        ckpt.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        env_state = self._collect_env_state()
        if env_state:
            ckpt["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(ckpt, path)

    def load(self, ckpt_path: str | None, strict: bool = True):
        if ckpt_path is None:
            return None
        logger.info(f"Loading DAgger student checkpoint from {ckpt_path} (strict={strict})")
        loaded = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(loaded["actor_model_state_dict"], strict=strict)
        if self.config.load_optimizer and strict and "actor_optimizer_state_dict" in loaded:
            self.actor_optimizer.load_state_dict(loaded["actor_optimizer_state_dict"])
            self.actor_learning_rate = loaded["actor_optimizer_state_dict"]["param_groups"][0]["lr"]
        self.current_learning_iteration = loaded.get("iter", 0)
        self._restore_env_state(loaded.get("env_state"))
        return loaded.get("infos")

    # ------------------------------------------------------------------ #
    # ONNX export (mirrors PPO)
    # ------------------------------------------------------------------ #

    @property
    def actor_onnx_wrapper(self):
        from holosoma.agents.modules.ppo_modules import PPOActorWithHeightMapVideoMimic

        is_videomimic_hm = isinstance(self.actor, PPOActorWithHeightMapVideoMimic)
        videomimic_combine_mode = getattr(self.actor, "combine_mode", "concat") if is_videomimic_hm else "concat"

        if self.student_use_height_map:
            class W(nn.Module):
                def __init__(self, actor, is_vm, mode):
                    super().__init__()
                    self.actor = actor
                    self.height_map_encoder = actor.height_map_encoder
                    self.actor_module = actor.actor_module
                    self.is_vm = is_vm
                    self.mode = mode

                def forward(self, actor_obs, height_map_obs):
                    map_latent = self.height_map_encoder(height_map_obs) if self.is_vm else self.height_map_encoder(
                        height_map_obs, actor_obs
                    )
                    if self.mode == "add_to_hidden":
                        x = self.actor_module[0](actor_obs)
                        x = x + map_latent
                        for layer in self.actor_module[1:]:
                            x = layer(x)
                        return x
                    return self.actor_module(torch.cat([actor_obs, map_latent], dim=-1))

            return W(self.actor, is_videomimic_hm, videomimic_combine_mode)

        class W(nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, actor_obs):
                return self.actor.act_inference({"actor_obs": actor_obs})

        return W(self.actor)

    def get_example_obs(self):
        n = self.env.num_envs
        ex = {
            "actor_obs": torch.zeros(n, self._get_obs_dim(self.actor_obs_keys), device=self.device),
        }
        if self.student_use_height_map:
            ex["height_map_obs"] = torch.zeros(n, self.algo_obs_dim_dict["height_map_obs"], device=self.device)
        return ex

    def export(self, onnx_file_path: str):
        was_training = self.actor.training
        self.actor.eval()

        motion_command = self.env.command_manager.get_state("motion_command")
        if (
            motion_command is not None
            and not self.student_use_height_map
            and hasattr(motion_command, "motion")
        ):
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
                example_obs_dict=self.get_example_obs(),
            )

        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.env)
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)
        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
        }
        metadata.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        attach_onnx_metadata(onnx_path=onnx_file_path, metadata=metadata)
        self.logging_helper.save_to_wandb(onnx_file_path)

        if was_training:
            self.actor.train()

    # ------------------------------------------------------------------ #
    # Eval — port of PPO's callback-driven loop, dropped critic/motion-encoder
    # branches since pure-DAgger student has neither.
    # ------------------------------------------------------------------ #

    def _evaluate_during_training(self):
        """Run eval_callbacks (e.g. SuccessRateCallback) and return (metrics, obs_dict).

        Mirrors PPO's behaviour: switches the actor to eval mode, runs the
        callback-driven loop on the env, then restores training state and
        returns a fresh obs_dict for the training loop to resume.
        """
        logger.info("Starting periodic evaluation...")

        was_training = self.actor.training

        self._create_eval_callbacks()
        self.actor.eval()
        self.env.set_is_evaluating()

        with torch.inference_mode():
            obs_dict = self.env.reset_all()
            for c in self.eval_callbacks:
                c.on_pre_evaluate_policy()

            actor_state = self._create_actor_state()
            self.eval_policy = self.get_inference_policy()
            init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
            actor_state.update({"obs": obs_dict, "actions": init_actions})

            max_steps = 100000
            for step in range(max_steps):
                actor_state["step"] = step
                actor_state = self._pre_eval_env_step(actor_state)
                if actor_state.get("stop", False):
                    break
                actor_state = self.env_step(actor_state)
                actor_state = self._post_eval_env_step(actor_state)
                if actor_state.get("stop", False):
                    break

            self._post_evaluate_policy()

            all_metrics: dict[str, float] = {}
            for cb in self.eval_callbacks:
                if hasattr(cb, "metrics") and cb.metrics:
                    all_metrics.update(cb.metrics)

            self.env.is_evaluating = False
            if was_training:
                self.actor.train()

            obs_dict = self.env.reset_all()
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].to(self.device)

            logger.info(f"Evaluation complete: {all_metrics}")
            return all_metrics, obs_dict

    def get_inference_policy(self, device=None):
        self.actor.eval()
        if device is not None:
            self.actor.to(device)
        return self.actor.act_inference

    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}

    def _create_eval_callbacks(self):
        if self.eval_callbacks:
            return
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env: bool = True):
        self.actor.eval()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()
        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state):
        actor_obs = torch.cat([actor_state["obs"][k] for k in self.actor_obs_keys], dim=1)
        policy_input = {"actor_obs": actor_obs}
        if self.student_use_height_map and "height_map_obs" in actor_state["obs"]:
            policy_input["height_map_obs"] = actor_state["obs"]["height_map_obs"]
        actions = self.eval_policy(policy_input)
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update({"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras})
        return actor_state

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        import itertools

        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        self.eval_policy = self.get_inference_policy()

        obs_dict = self.env.reset_all()
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].to(self.device)
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})

        actor_state = self._pre_eval_env_step(actor_state)
        for step in itertools.islice(itertools.count(), max_eval_steps):
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            if actor_state.get("stop", False):
                break
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
            if actor_state.get("stop", False):
                break

        self._post_evaluate_policy()
