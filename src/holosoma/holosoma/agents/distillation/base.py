"""Training-loop plumbing shared by the distillation algorithms.

:class:`~holosoma.agents.distillation.distillation.Distillation` (pure DAgger) and
:class:`~holosoma.agents.distillation_ppo.distillation_ppo.DistillationPPO` differ only
in their policy, optimizer, rollout storage, and update rule. Everything else --
observation-group lookup, the iteration loop, teacher loading, checkpoints, ONNX
export, and evaluation -- lives here.

Observation groups:

- ``actor_obs``: proprioception, consumed by the student.
- ``teacher_obs``: privileged observation, consumed by the frozen teachers.
- ``critic_obs``: privileged observation, consumed by the critic (PPO only).
- the depth group named by ``module.depth_obs_group``: only read when
  ``module.depth_backbone`` is set, in which case the student also consumes the
  backbone's latent of that image.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any, Callable, ClassVar, cast

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.distillation.checkpoint import (
    convert_legacy_teacher_state,
    infer_num_teachers_from_checkpoint,
    legacy_mlp_teacher_configs,
)
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.modules.student_teacher_modules import StudentTeacher
from holosoma.config_types.algo import DistillationConfig, DistillationPPOConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)

ACTOR_GROUP = "actor_obs"
TEACHER_GROUP = "teacher_obs"
CRITIC_GROUP = "critic_obs"


class DistillationBase(BaseAlgo):
    """Base class for student/teacher distillation algorithms.

    Subclasses implement :meth:`_build_policy`, :meth:`_build_optimizer`,
    :meth:`_build_storage`, :meth:`_rollout_step`, :meth:`_training_step`, and
    :meth:`_post_epoch_logging`, and may hook :meth:`_on_iteration_start` and the
    checkpoint-state methods.
    """

    config: DistillationConfig | DistillationPPOConfig
    policy: StudentTeacher
    optimizer: torch.optim.Optimizer

    uses_critic: ClassVar[bool] = False
    """Whether the algorithm trains a critic on the ``critic_obs`` group."""

    def __init__(
        self,
        env: BaseTask,
        config: DistillationConfig | DistillationPPOConfig,
        log_dir: str | os.PathLike[str],
        device: str = "cpu",
        multi_gpu_cfg: dict[str, int] | None = None,
    ):
        super().__init__(env, config, device, multi_gpu_cfg)
        if self.config.empirical_normalization:
            # The observation manager is expected to apply any normalization.
            raise ValueError(f"empirical_normalization is not supported by {type(self).__name__}.")

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

        # Index of the next training iteration to run; iterations below it are complete.
        self.current_learning_iteration = 0
        self.eval_callbacks: list[RLEvalCallback] = []
        self.learning_rate = self.config.learning_rate

        # Number of frozen teacher MLPs to build in ``_build_policy``. Set before
        # ``setup()``, from the teacher checkpoints or a saved checkpoint (see
        # ``restore_structure_from_checkpoint``); defaults to 1. Applications using
        # several teachers must override the policy's ``teacher_act`` to supply a
        # per-sample teacher index.
        self.num_teachers: int = 1

        # Flipped per iteration inside ``learn()``. While active, rollouts step the
        # env with teacher actions instead of the student's sampled actions.
        self._warmup_active = False

        # Resolved eagerly so a bad config surfaces at construction, not in update().
        if self.config.distill_loss_type == "mse":
            self.distill_loss_fn: Callable[..., torch.Tensor] = F.mse_loss
        elif self.config.distill_loss_type == "huber":
            self.distill_loss_fn = F.huber_loss
        else:
            raise ValueError(f"Unknown distill_loss_type: {self.config.distill_loss_type!r}")

        _ = self.env.reset_all()

    # ------------------------------------------------------------------ setup

    @property
    def uses_depth(self) -> bool:
        """Whether the student consumes a depth image through a backbone."""
        return self.config.module.depth_backbone is not None

    def setup(self) -> None:  # type: ignore[override]
        logger.info(f"Setting up {type(self).__name__}")
        self._resolve_obs_dims()
        self._build_policy()
        # Broadcast rank 0's initial weights before the optimizer sees them.
        if self.is_multi_gpu:
            self._synchronize_model_weights()
        self._build_optimizer()
        self._build_storage()

    def _build_policy(self) -> None:
        raise NotImplementedError

    def _build_optimizer(self) -> None:
        raise NotImplementedError

    def _build_storage(self) -> None:
        raise NotImplementedError

    def _resolve_obs_dims(self) -> None:
        """Read the observation dimensions the policy and storage are built from."""
        assert self.env.observation_manager is not None
        obs_dims = self.env.observation_manager.get_obs_dims()

        groups = [ACTOR_GROUP, TEACHER_GROUP] + ([CRITIC_GROUP] if self.uses_critic else [])
        for group in groups:
            if group not in obs_dims:
                raise KeyError(f"{type(self).__name__} expects observation group {group!r} to be present.")
            if not isinstance(obs_dims[group], int):
                raise TypeError(
                    f"Observation group {group!r} must concatenate into a flat int dimension; "
                    f"got {obs_dims[group]!r} from observation_manager.get_obs_dims()."
                )

        self.num_act = self.env.robot_config.actions_dim
        self.actor_obs_dim = cast("int", obs_dims[ACTOR_GROUP])
        self.teacher_obs_dim = cast("int", obs_dims[TEACHER_GROUP])
        if self.uses_critic:
            self.critic_obs_dim = cast("int", obs_dims[CRITIC_GROUP])
        self.depth_shape = self._resolve_depth_shape(obs_dims) if self.uses_depth else None

        dims: dict[str, object] = {"actor": self.actor_obs_dim, "teacher": self.teacher_obs_dim}
        if self.uses_critic:
            dims["critic"] = self.critic_obs_dim
        if self.depth_shape is not None:
            dims["depth"] = self.depth_shape
        dims["actions"] = self.num_act
        logger.info(f"{type(self).__name__} obs dims -- " + ", ".join(f"{k}: {v}" for k, v in dims.items()))

    def _resolve_depth_shape(self, obs_dims: dict[str, Any]) -> tuple[int, ...]:
        group, term = self.config.module.depth_obs_group, self.config.module.depth_obs_term
        if group not in obs_dims:
            raise KeyError(
                f"{type(self).__name__} has a depth backbone, so it expects observation group {group!r} to be present."
            )
        entry = obs_dims[group]
        if isinstance(entry, int):
            # A concatenated group is already flat.
            return (entry,)
        if not isinstance(entry, dict):
            raise TypeError(f"Unexpected type for depth obs dims: {type(entry)}")
        if term not in entry:
            raise KeyError(f"depth obs group {group!r} missing term {term!r}; found {list(entry.keys())}")
        # get_obs_dims() reports image terms flattened; sample the term for its real shape.
        assert self.env.observation_manager is not None
        depth = self.env.observation_manager.compute_group(group, modify_history=False)
        if not isinstance(depth, dict):
            raise TypeError(f"expected compute_group({group!r}) to return a dict when concatenate=False.")
        return tuple(depth[term].shape[1:])

    # --------------------------------------------------------------- obs helpers

    def _extract_obs(self, obs_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Pull the training inputs out of a manager-produced observation dict.

        Returns the student inputs (:meth:`_extract_student_obs`) plus ``teacher_obs``,
        and ``critic_obs`` for algorithms with a critic.
        """
        obs = self._extract_student_obs(obs_dict)
        obs[TEACHER_GROUP] = obs_dict[TEACHER_GROUP]
        if self.uses_critic:
            obs[CRITIC_GROUP] = obs_dict[CRITIC_GROUP]
        return obs

    def _extract_student_obs(self, obs_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Pull only the student's inputs: ``actor_obs``, plus ``depth_obs`` with a depth backbone.

        Evaluation uses this, so a trained student runs without the teacher and critic groups.
        """
        obs = {ACTOR_GROUP: obs_dict[ACTOR_GROUP]}
        if self.uses_depth:
            entry = obs_dict[self.config.module.depth_obs_group]
            obs["depth_obs"] = entry[self.config.module.depth_obs_term] if isinstance(entry, dict) else entry
        return obs

    @staticmethod
    def _step_actions(
        actions: torch.Tensor, teacher_actions: torch.Tensor, teacher_valid: torch.Tensor, warmup: bool
    ) -> torch.Tensor:
        """Actions that step the env: during warmup, the teacher's where one applies."""
        if not warmup:
            return actions
        return torch.where(teacher_valid.unsqueeze(-1), teacher_actions, actions)

    @staticmethod
    def _dagger_target(
        student_mean: torch.Tensor, teacher_actions: torch.Tensor, teacher_valid: torch.Tensor
    ) -> torch.Tensor:
        """Behaviour-cloning target with rows lacking a teacher masked out.

        Masked rows take the student's own detached output, so they add exactly zero loss and
        zero gradient while the loss stays averaged over the full batch. With every row valid
        the target is ``teacher_actions`` unchanged.
        """
        return torch.where(teacher_valid.view(-1, 1), teacher_actions, student_mean.detach())

    def _dagger_loss(
        self, student_mean: torch.Tensor, teacher_actions: torch.Tensor, teacher_valid: torch.Tensor
    ) -> torch.Tensor:
        """Batch-mean behaviour-cloning loss; rows where ``teacher_valid`` is false add zero.

        Override to change how labels without a teacher are handled.
        """
        return self.distill_loss_fn(student_mean, self._dagger_target(student_mean, teacher_actions, teacher_valid))

    def _obs_to_device(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        for key, value in obs_dict.items():
            if isinstance(value, torch.Tensor):
                obs_dict[key] = value.to(self.device)
            elif isinstance(value, dict):
                obs_dict[key] = {k: v.to(self.device) for k, v in value.items()}
        return obs_dict

    # --------------------------------------------------------------- train mode

    def _train_mode(self) -> None:
        self.policy.train()  # teachers stay in eval mode; see StudentTeacher.train

    def _eval_mode(self) -> None:
        self.policy.eval()

    # -------------------------------------------------------------------- learn

    def learn(self, num_learning_iterations: int | None = None) -> None:  # type: ignore[override]
        self._train_mode()
        self.policy.require_loaded_teachers()

        obs_dict = self._obs_to_device(self.env.reset_all())

        total_iters = (
            num_learning_iterations if num_learning_iterations is not None else self.config.num_learning_iterations
        )
        start_iter = self.current_learning_iteration
        stop_iter = start_iter + total_iters

        for it in range(start_iter, stop_iter):
            # Warm-up is an absolute initial-training window. A resumed run must
            # not replay it from zero.
            self._warmup_active = it < self.config.distillation_warmup_steps
            self._on_iteration_start(it)

            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            with self.logging_helper.record_collection_time():
                obs_dict = self._rollout_step(obs_dict)

            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()

            if self.is_main_process:
                self._post_epoch_logging(it, loss_dict)

            # Iteration ``it`` is complete: the next learn() call or resume starts after it.
            self.current_learning_iteration = it + 1

            if it % self.config.save_interval == 0 and self.is_main_process:
                self.save(os.path.join(self.log_dir, f"model_{it:05d}.pt"))

        if self.is_main_process and self.current_learning_iteration > start_iter:
            self.save(os.path.join(self.log_dir, f"model_{self.last_completed_iteration:05d}.pt"))

    def _on_iteration_start(self, iteration: int) -> None:
        """Hook run at the start of every training iteration, before the rollout."""

    def _rollout_step(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _training_step(self) -> dict[str, float]:
        raise NotImplementedError

    def _post_epoch_logging(self, it: int, loss_dict: dict[str, float]) -> None:
        raise NotImplementedError

    # --------------------------------------------------------------- multi-gpu

    def _reduce_parameters(self) -> None:
        """Average trainable gradients across distributed workers.

        Frozen teacher parameters have no gradient and are skipped.
        """
        parameters = [parameter for parameter in self.policy.parameters() if parameter.grad is not None]
        if not parameters:
            return

        flat_grad = torch.cat([parameter.grad.view(-1) for parameter in parameters])
        torch.distributed.all_reduce(flat_grad, op=torch.distributed.ReduceOp.SUM)
        flat_grad /= self.gpu_world_size

        offset = 0
        for parameter in parameters:
            numel = parameter.numel()
            parameter.grad.copy_(flat_grad[offset : offset + numel].view_as(parameter.grad))
            offset += numel

    def _synchronize_model_weights(self) -> None:
        """Start every distributed worker from the rank-zero policy weights."""
        for parameter in self.policy.parameters():
            torch.distributed.broadcast(parameter.data, src=0)
        logger.info(f"Synchronized {type(self).__name__} model weights across {self.gpu_world_size} GPUs")

    # ---------------------------------------------------------- save/load/teacher

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        """Algorithm-specific entries to store in a full training checkpoint."""
        return {}

    def _restore_extra_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        """Restore the entries written by :meth:`_extra_checkpoint_state`."""

    def save(self, path: str | None = None, name: str = "last.ckpt") -> None:
        if path is None:
            path = os.path.join(self.log_dir, name)
        checkpoint_dict: dict[str, Any] = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # The last completed iteration, as PPO checkpoints record it.
            "iter": self.last_completed_iteration,
            "learning_rate": self.learning_rate,
            **self._extra_checkpoint_state(),
            "num_teachers": self.policy.num_teachers,
            "teacher_loaded": list(self.policy.loaded_teachers),
            "teacher_configs": self.policy.teacher_construction_configs(),
        }
        checkpoint_dict.update(self._checkpoint_metadata(iteration=self.last_completed_iteration))
        env_state = self._collect_env_state()
        if env_state:
            checkpoint_dict["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(checkpoint_dict, path)

    @property
    def last_completed_iteration(self) -> int:
        """Index of the last completed training iteration, ``-1`` before any."""
        return self.current_learning_iteration - 1

    def restore_structure_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Size the policy like the one ``checkpoint`` was saved from. Call before :meth:`setup`.

        Entrypoints that build the algorithm and then :meth:`load` a full checkpoint use
        this so a multi-teacher checkpoint loads into a policy with the same teacher slots.
        """
        teacher_count = infer_num_teachers_from_checkpoint(checkpoint)
        if teacher_count is not None:
            self.num_teachers = teacher_count

    def load(self, path: str | None) -> dict[str, Any] | None:  # type: ignore[override]
        if path is None:
            return None
        logger.info(f"Loading {type(self).__name__} checkpoint from {path}")
        loaded = torch.load(path, map_location=self.device)
        state_dict = loaded["model_state_dict"]
        teacher_configs = loaded.get("teacher_configs")
        if teacher_configs is None:
            # Saved before teachers recorded their construction: rebuild them as the MLPs it used.
            state_dict = convert_legacy_teacher_state(state_dict, self.policy.num_teachers)
            teacher_configs = legacy_mlp_teacher_configs(
                state_dict,
                self.policy.num_teachers,
                hidden_dims=list(self.config.module.teacher_hidden_dims),
                activation=self.config.module.activation,
                num_actions=self.policy.num_actions,
            )
        self.policy.build_teachers(teacher_configs)
        self.policy.load_state_dict(state_dict, strict=True)
        teacher_loaded = loaded.get("teacher_loaded")
        if teacher_loaded is None:
            logger.warning(
                "Checkpoint predates teacher-readiness metadata. Teacher weights remain "
                "unverified; provide training.teacher_checkpoint before continuing training."
            )
            teacher_loaded = [False] * self.policy.num_teachers
        self.policy.set_loaded_teachers(teacher_loaded)
        if "optimizer_state_dict" in loaded:
            try:
                self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
            except Exception as exc:  # optimizer state has no public signature to validate
                logger.warning(f"Could not load optimizer state: {exc}")
        # ``iter`` is the last completed iteration; training resumes after it.
        self.current_learning_iteration = int(loaded.get("iter", -1)) + 1
        if "learning_rate" in loaded:
            self.learning_rate = float(loaded["learning_rate"])
        self._restore_extra_checkpoint_state(loaded)
        self.policy.project_action_std()
        self._restore_env_state(loaded.get("env_state"))
        return cast("dict[str, Any] | None", loaded.get("infos"))

    def load_teacher(self, paths: str | list[str]) -> None:
        """Rebuild ``policy.teachers`` from teacher training checkpoints, one path per slot.

        Each teacher is reconstructed from its own checkpoint and loaded strictly; see
        :func:`holosoma.agents.modules.teacher_policies.teacher_from_checkpoint`.
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
            logger.info(f"Loading teacher[{i}] from {path}")
            self.policy.load_teacher_checkpoint(torch.load(path, map_location="cpu"), teacher_index=i)

    # --------------------------------------------------------- inference / eval

    @property
    def inference_model(self) -> StudentTeacher:
        return self.policy

    @property
    def actor_onnx_wrapper(self) -> nn.Module:
        if self.uses_depth:
            raise NotImplementedError(
                f"{type(self).__name__} exports a depth_backbone/student ONNX pair; use export()."
            )

        class StudentWrapper(nn.Module):
            def __init__(self, policy: StudentTeacher) -> None:
                super().__init__()
                self.policy = policy

            def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
                return self.policy.act_inference({"actor_obs": actor_obs})

        return StudentWrapper(self.policy)

    @torch.no_grad()
    def _check_export_matches_inference(self, backbone: nn.Module, student: nn.Module) -> None:
        """Fail if running the exported pair as deployed differs from ``act_inference``.

        Catches a policy whose input preparation the export modules do not reproduce.
        """
        assert self.depth_shape is not None
        generator = torch.Generator(device="cpu").manual_seed(0)
        actor_obs = torch.randn(4, self.actor_obs_dim, generator=generator).to(self.device)
        depth_obs = torch.rand(4, *self.depth_shape, generator=generator).to(self.device) - 0.5
        exported = student(torch.cat([actor_obs, backbone(depth_obs)], dim=-1))
        expected = self.policy.act_inference({"actor_obs": actor_obs, "depth_obs": depth_obs})
        if not torch.allclose(exported, expected, rtol=1e-5, atol=1e-6):
            raise RuntimeError(
                f"{type(self.policy).__name__}'s export_modules() do not reproduce act_inference "
                f"(max abs diff {(exported - expected).abs().max().item():.3g}). Prepare the student "
                "input in _combine_student_input, or override export_modules()."
            )

    def export(self, onnx_file_path: str) -> None:
        """Export the student for deployment.

        A proprioception-only student is written to ``onnx_file_path`` with the same
        input/output names as PPO's export. A student with a depth backbone is
        written as a ``depth_backbone.onnx`` + ``student.onnx`` pair in a directory
        named after ``onnx_file_path`` without ``.onnx`` (``model_00500.onnx`` ->
        ``model_00500/``): the backbone encodes a depth image into its latent, and
        the student maps ``[actor_obs | depth_latent]`` to actions. Teachers and
        critic are training-only and are not exported. For resuming training, see
        :meth:`save`.
        """
        was_training = self.policy.training
        self._eval_mode()

        if self.uses_depth:
            export_dir = str(Path(onnx_file_path).with_suffix(""))
            os.makedirs(export_dir, exist_ok=True)
            backbone_path = os.path.join(export_dir, "depth_backbone.onnx")
            student_path = os.path.join(export_dir, "student.onnx")
            assert self.depth_shape is not None
            backbone, student = self.policy.export_modules()
            self._check_export_matches_inference(backbone, student)
            torch.onnx.export(
                backbone,
                torch.zeros(1, *self.depth_shape, device=self.device),
                backbone_path,
                input_names=["depth_image"],
                output_names=["depth_latent"],
                opset_version=13,
                dynamo=False,
            )
            torch.onnx.export(
                student,
                torch.zeros(1, self.actor_obs_dim + self.policy.depth_output_dim, device=self.device),
                student_path,
                input_names=["obs"],
                output_names=["actions"],
                opset_version=13,
                dynamo=False,
            )
            exported_paths = [backbone_path, student_path]
        else:
            student_path = onnx_file_path
            export_policy_as_onnx(
                wrapper=self.actor_onnx_wrapper,
                onnx_file_path=student_path,
                example_obs_dict={"actor_obs": torch.zeros(1, self.actor_obs_dim, device=self.device)},
            )
            exported_paths = [student_path]

        # Same metadata as PPO.export, plus the joint order under the
        # ``joint_names`` key the depth policy reads.
        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        action_scales = getattr(self.env, "action_scales", None)
        if action_scales is None:
            action_scale_metadata: float | list[float] = float(self.env.robot_config.control.action_scale)
        else:
            action_scale_metadata = action_scales.detach().cpu().tolist()
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)
        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "joint_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "action_scale": action_scale_metadata,
            "command_ranges": get_command_ranges_from_env(self.env),
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
        }
        metadata.update(self._checkpoint_metadata(iteration=self.last_completed_iteration))
        attach_onnx_metadata(onnx_path=student_path, metadata=metadata)

        for path in exported_paths:
            self.logging_helper.save_to_wandb(path)
        logger.info(f"Exported {', '.join(Path(p).name for p in exported_paths)} to {Path(student_path).parent}")

        if was_training:
            self._train_mode()

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        self.policy.eval()
        if device is not None:
            self.policy.to(device)

        @torch.no_grad()
        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            return self.policy.act_inference({"actor_obs": obs["actor_obs"], "depth_obs": obs.get("depth_obs")})

        # torch.no_grad() erases the decorated function's type.
        return cast("Callable[[dict[str, torch.Tensor]], torch.Tensor]", policy_fn)

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None) -> None:
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state: dict[str, Any] = {"done_indices": [], "stop": False}
        eval_policy = self.get_inference_policy()

        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state["obs"].update(self._extract_student_obs(obs_dict))

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

    def _pre_eval_env_step(
        self, actor_state: dict[str, Any], eval_policy: Callable[[dict[str, torch.Tensor]], torch.Tensor]
    ) -> dict[str, Any]:
        actor_state["actions"] = eval_policy(self._extract_student_obs(actor_state["obs"]))
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    # Keep the returned actor_state the same shape as PPO's.
    def env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update({"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras})
        return actor_state
