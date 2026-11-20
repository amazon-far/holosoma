"""Task specialization that owns the policy managers and rollout buffers."""

from __future__ import annotations

import dataclasses
import datetime
import logging
import os
import sys
import traceback
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

from loguru import logger

from holosoma.config_types.env import EnvConfig
from holosoma.config_types.experiment import EvaluationExperimentConfig, TrainingExperimentConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.action import ActionManager
from holosoma.managers.command import CommandManager
from holosoma.managers.curriculum import CurriculumManager
from holosoma.managers.observation import ObservationManager
from holosoma.managers.reward import RewardManager
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import load_checkpoint
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import setup_keyboard_listener

if TYPE_CHECKING:
    from holosoma.managers.curriculum.terms.locomotion import AverageEpisodeLengthTracker

_LEGACY_CHECKPOINT_REMOVAL_DATE = datetime.date(2027, 3, 1)


class PolicyTask(BaseTask):
    """BaseTask specialization for policy training and evaluation."""

    BASE_NUM_ENVS = 4096
    """Env count the curriculum terms' per-env rates are normalized to."""

    @classmethod
    def train(cls, config: TrainingExperimentConfig) -> None:
        """Train ``config.algo`` on this task class, built from ``config``."""
        from holosoma.simulator.session import task_session

        config = split_envs_across_ranks(config)
        try:
            # Before the task is built, so construction logs reach the run's log files.
            setup = _setup_training(config)
            with task_session(config.env_config(), cls) as task:
                train_on_task(config, task, setup)
        except Exception as e:
            logger.error(f"Exception occurred during training: {e}\n{traceback.format_exc()}")
            sys.exit(1)  # manually set exit code, not possible via isaacsim app.close()

    @classmethod
    def evaluate(cls, config: EvaluationExperimentConfig) -> None:
        """Evaluate ``config.eval.checkpoint`` on this task class, built from ``config``."""
        from holosoma.simulator.session import task_session
        from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config

        saved, saved_wandb_path = load_saved_experiment_config(CheckpointConfig(checkpoint=config.eval.checkpoint))
        with task_session(config.env_config(), cls) as task:
            _run_evaluation(task, config, saved_config=saved, saved_wandb_path=saved_wandb_path)

    def __init__(self, tyro_config: EnvConfig) -> None:
        for name in ("observation", "action", "reward", "termination", "command", "curriculum"):
            if getattr(tyro_config, name) is None:
                raise ValueError(f"{name}_config must be provided for manager-based environments.")
        self.is_evaluating = False
        self.dim_obs = tyro_config.robot.policy_obs_dim
        super().__init__(tyro_config)
        if not self.headless:
            setup_keyboard_listener(self)

    def _initialize_task_managers(self) -> None:
        super()._initialize_task_managers()
        config = self.config
        assert config.observation is not None
        assert config.action is not None
        assert config.reward is not None
        assert config.command is not None
        assert config.curriculum is not None

        self.observation_manager = ObservationManager(config.observation, self, self.device)
        self.action_manager = ActionManager(config.action, self, self.device)
        self.reward_manager = RewardManager(config.reward, self, self.device)
        self.command_manager = CommandManager(config.command, self, self.device)
        self.curriculum_manager = CurriculumManager(config.curriculum, self, self.device)

    def _prepare_manager_fields(self) -> None:
        self.simulator.prepare_manager_fields(
            randomization_manager=self.randomization_manager,
            observation_manager=self.observation_manager,
            reward_manager=self.reward_manager,
        )

    def _setup_task_managers(self) -> None:
        super()._setup_task_managers()
        self.action_manager.setup()
        self.command_manager.setup()
        self.curriculum_manager.setup()

    def _init_buffers(self) -> None:
        super()._init_buffers()

        # Record history length from observation manager config
        self.history_length: dict[str, int] = {}
        for group_name, group_cfg in self.observation_manager.cfg.groups.items():
            self.history_length[group_name] = group_cfg.history_length

        self.obs_buf_dict: dict[str, Any] = {}
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.extras: dict[str, Any] = {}
        self.log_dict: dict[str, Any] = {}
        self._pending_episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._pending_episode_update_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._pending_torque_rfi: tuple[bool, float] = (False, 0.0)

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return environment-specific state to persist in checkpoints."""
        state: dict[str, Any] = {}
        curriculum_state = self.curriculum_manager.state_dict()
        if curriculum_state:
            state["curriculum"] = curriculum_state
        return state

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore environment-specific state from a checkpoint."""
        if not state:
            return
        if "curriculum" not in state:
            state = self._convert_legacy_checkpoint_state(state)
        curriculum_state = state.get("curriculum")
        if not curriculum_state:
            return
        self.curriculum_manager.load_state_dict(curriculum_state)
        for name in curriculum_state:
            term = self.curriculum_manager.get_term(name)
            if term is not None:
                term.after_checkpoint_load()

    def synchronize_curriculum_state(self, *, device: str, world_size: int) -> None:
        """Broadcast rank 0's curriculum state to every rank."""
        if world_size <= 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        # Every rank builds the same terms from the same config, so all ranks agree on skipping. A term
        # whose state is empty depending on runtime data would deadlock here.
        state = self.curriculum_manager.state_dict()
        if not state:
            return
        if not getattr(self, "_logged_curriculum_sync", False):
            logger.info(f"Multi-GPU curriculum synchronization enabled for terms: {sorted(state)}")
            self._logged_curriculum_sync = True
        payload = [state]
        torch.distributed.broadcast_object_list(payload, src=0, device=torch.device(device))
        if torch.distributed.get_rank() != 0:
            self.curriculum_manager.load_state_dict(payload[0])

    def _convert_legacy_checkpoint_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Map the pre-``curriculum`` checkpoint layout onto curriculum term state."""
        from holosoma.managers.curriculum.terms.locomotion import AverageEpisodeLengthTracker, PenaltyCurriculum

        def term_name(term_cls: type) -> str | None:
            return next(
                (name for name, term in self.curriculum_manager.iter_terms() if isinstance(term, term_cls)), None
            )

        curriculum: dict[str, dict[str, Any]] = {}
        tracker_state = state.get("average_episode_tracker")
        tracker_name = term_name(AverageEpisodeLengthTracker)
        if tracker_state is not None and tracker_name is not None:
            curriculum[tracker_name] = tracker_state
        penalty_scale = state.get("reward_penalty_scale")
        penalty_name = term_name(PenaltyCurriculum)
        if penalty_scale is not None and penalty_name is not None:
            curriculum[penalty_name] = {"current_scale": float(penalty_scale)}
        if curriculum:
            message = (
                "This checkpoint uses the old env-state layout (top-level 'average_episode_tracker' / "
                "'reward_penalty_scale'), which is deprecated. Re-save it to store curriculum state per term "
                f"under 'curriculum'. Loading the old layout becomes an error after {_LEGACY_CHECKPOINT_REMOVAL_DATE}."
            )
            if datetime.datetime.now(tz=datetime.timezone.utc).date() > _LEGACY_CHECKPOINT_REMOVAL_DATE:
                raise RuntimeError(message)
            # FutureWarning: shown by default (DeprecationWarning is filtered outside __main__).
            warnings.warn(message, FutureWarning, stacklevel=4)
        return {**state, "curriculum": curriculum}

    def set_is_evaluating(self, command: Any = None) -> None:
        """Switch to evaluation, called by the agent before evaluating a policy.

        Command terms enter their evaluation mode; ``command``, when given, is written to every
        env's locomotion command.
        """
        logger.info("Setting Env is evaluating")
        self.is_evaluating = True
        self.command_manager.on_evaluate()
        if command is not None:
            commands = self.command_manager.commands
            commands[:] = torch.as_tensor(command, device=self.device, dtype=commands.dtype).view(1, -1)

    @property
    def base_quat(self) -> torch.Tensor:
        return self.simulator.base_quat

    @property
    def average_episode_length(self) -> float:
        avg = self._get_average_episode_tracker().get_average()
        return float(avg.detach().cpu().item())

    def _setup_robot_body_indices(self) -> None:
        super()._setup_robot_body_indices()
        foot_body_names = [s for s in self.body_names if self.robot_config.foot_body_name in s]
        foot_height_names = [s for s in self.body_names if self.robot_config.foot_height_name in s]
        self.feet_indices = torch.tensor(
            [self.simulator.find_rigid_body_indice(name) for name in foot_body_names],
            dtype=torch.long,
            device=self.device,
        )
        self.feet_height_indices = torch.tensor(
            [self.simulator.find_rigid_body_indice(name) for name in foot_height_names],
            dtype=torch.long,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Reset hooks

    def _prepare_reset(self, env_ids: torch.Tensor) -> None:
        # Reset observation history BEFORE state changes (must happen first to clear history buffers)
        self.observation_manager.reset(env_ids)
        self._pending_episode_lengths[env_ids] = self.episode_length_buf[env_ids]
        self._pending_episode_update_mask[env_ids] = True

    def _reset_task_managers(self, env_ids: torch.Tensor) -> None:
        reward_extras = self.reward_manager.reset(env_ids)

        # Normalise extras dictionary to contain (possibly empty) sub-sections.
        self.extras["episode"] = reward_extras.get("episode", {})
        self.extras["episode_all"] = reward_extras.get("episode_all", {})
        self.extras["raw_episode"] = reward_extras.get("raw_episode", {})
        self.extras["raw_episode_all"] = reward_extras.get("raw_episode_all", {})

        self.extras["time_outs"] = self.time_out_buf

        super()._reset_task_managers(env_ids)
        self.action_manager.reset(env_ids)
        self.command_manager.reset(env_ids)
        self.curriculum_manager.reset(env_ids)

    def _reset_buffers_callback(
        self, env_ids: torch.Tensor, target_buf: Mapping[str, torch.Tensor] | None = None
    ) -> None:
        super()._reset_buffers_callback(env_ids)
        if target_buf is not None:
            self.simulator.dof_pos[env_ids] = target_buf["dof_pos"].to(self.simulator.dof_pos.dtype)
            self.simulator.dof_vel[env_ids] = target_buf["dof_vel"].to(self.simulator.dof_vel.dtype)
            self.episode_length_buf[env_ids] = target_buf["episode_length_buf"].to(self.episode_length_buf.dtype)
            self.reset_buf[env_ids] = target_buf["reset_buf"].to(self.reset_buf.dtype)
            self.time_out_buf[env_ids] = target_buf["time_out_buf"].to(self.time_out_buf.dtype)
            self._pending_episode_update_mask[env_ids] = False
            self._pending_episode_lengths[env_ids] = 0

    def _reset_robot_states_callback(
        self, env_ids: torch.Tensor, target_states: Mapping[str, torch.Tensor] | None = None
    ) -> None:
        """Reset the robot unless a command term resets it; target root positions are relative to env origins."""
        if self.command_manager.resets_robot:
            return
        if target_states is not None:
            root_states = target_states["root_states"].clone()
            root_states[:, :3] += self._get_env_origins()[env_ids]
            super()._reset_robot_states_callback(
                env_ids, {"root_states": root_states, "dof_states": target_states["dof_states"]}
            )
            return
        super()._reset_robot_states_callback(env_ids)

    def _after_reset_synchronization(self, env_ids: torch.Tensor) -> None:
        self.command_manager.after_reset(env_ids)
        self._pre_compute_observations_callback()

    def reset_all(self) -> dict[str, Any]:
        super().reset_all()
        return self.compute_reset_observations()

    def compute_reset_observations(self) -> dict[str, Any]:
        """Compute observations of the current state after a reset, without advancing physics."""
        self.observation_manager.new_frame()
        self.obs_buf_dict = self.observation_manager.step()
        return self.obs_buf_dict

    # ------------------------------------------------------------------
    # Control frame

    def step(self, actor_state: dict[str, Any] | None = None) -> Any:
        """Apply ``actor_state["actions"]`` and advance one control frame."""
        if actor_state is None:
            raise TypeError("PolicyTask.step() requires an actor_state with 'actions'.")
        self._pre_physics_step(actor_state)
        return super().step()

    def _pre_physics_step(self, actor_state: dict[str, Any]) -> None:
        self.action_manager.process_actions(actor_state["actions"])

    def _apply_force_in_physics_step(self) -> None:
        self.action_manager.apply_actions()

    def _evaluate_frame(self, env_ids: torch.Tensor) -> None:
        self.observation_manager.new_frame()
        self._pre_compute_observations_callback()
        self._compute_reward()
        self._update_log_dict()
        # Time-outs are a subset of resets; only they are bootstrapped from a terminal observation.
        timeout_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
        if timeout_ids.numel() > 0:
            self._store_final_observations(timeout_ids, self.observation_manager.step_terminal(timeout_ids))

    def _post_physics_step(self) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict[str, Any]]:
        super()._post_physics_step()
        self.obs_buf_dict = self.observation_manager.step()
        self.extras["to_log"] = self.log_dict
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _step_managers(self) -> None:
        # After resets, so WBT termination checks see the tracking error before the command
        # manager advances the motion clip.
        self.command_manager.step()
        self.curriculum_manager.step()
        super()._step_managers()

    def _store_final_observations(self, env_ids: torch.Tensor, final_obs_dict: dict[str, Any]) -> None:
        if not final_obs_dict:
            return
        final_store = self.extras.setdefault("final_observations", {})

        def _store_value(store_parent: dict[str, Any], key: str, value: Any, template: Any) -> None:
            # A concatenate=False group is a dict of per-term tensors; recurse one level so the
            # per-env final-obs copy works for image groups too (env-axis is dim 0 either way).
            if isinstance(value, dict):
                sub = store_parent.setdefault(key, {})
                for sub_key, sub_val in value.items():
                    _store_value(sub, sub_key, sub_val, template[sub_key])
                return
            if key not in store_parent:
                store_parent[key] = torch.zeros_like(template)
            store_parent[key][env_ids] = value

        for obs_key, values in final_obs_dict.items():
            _store_value(final_store, obs_key, values, self.obs_buf_dict[obs_key])

    def _compute_reward(self) -> None:
        self.rew_buf[:] = self.reward_manager.compute(self.dt)

    # ------------------------------------------------------------------
    # Hooks for subclasses

    def _pre_compute_observations_callback(self) -> None:
        """Hook invoked after physics but before observation terms compute: refresh terrain heights."""
        self.terrain_manager.update_heights()

    def _update_log_dict(self) -> None:
        """Append per-frame metrics to ``log_dict``: the average episode length when tracked, and command metrics."""
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is not None:
            avg = cast("AverageEpisodeLengthTracker", tracker).get_average()
            self.log_dict["average_episode_length"] = avg.detach().cpu()
        self.log_dict.update(self.command_manager.log_metrics())

    def _get_average_episode_tracker(self) -> AverageEpisodeLengthTracker:
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is None:
            raise RuntimeError("AverageEpisodeLengthTracker is not registered with the curriculum manager.")
        return cast("AverageEpisodeLengthTracker", tracker)


class _MultGPUConfig(TypedDict):
    global_rank: int
    local_rank: int
    world_size: int


def _configure_multi_gpu() -> _MultGPUConfig | None:
    """Configure multi-gpu training and return configuration dictionary, or `None` if single-GPU training."""
    import torch

    gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
    is_distributed = gpu_world_size > 1

    if not is_distributed:
        return None

    gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
    gpu_global_rank = int(os.getenv("RANK", "0"))

    if gpu_local_rank >= gpu_world_size:
        raise ValueError(f"Local rank '{gpu_local_rank}' is greater than or equal to world size '{gpu_world_size}'.")

    if gpu_global_rank >= gpu_world_size:
        raise ValueError(f"Global rank '{gpu_global_rank}' is greater than or equal to world size '{gpu_world_size}'.")

    dist_backend = os.getenv("TORCH_DIST_BACKEND", "nccl")
    dist_timeout_s = int(os.getenv("TORCH_DIST_INIT_TIMEOUT_S", "7200"))
    from datetime import timedelta

    torch.distributed.init_process_group(
        backend=dist_backend,
        rank=gpu_global_rank,
        world_size=gpu_world_size,
        timeout=timedelta(seconds=dist_timeout_s),
    )
    torch.cuda.set_device(gpu_local_rank)

    multi_gpu_config: _MultGPUConfig = {
        "global_rank": gpu_global_rank,
        "local_rank": gpu_local_rank,
        "world_size": gpu_world_size,
    }
    logger.info(f"Running with multi-GPU parameters: {multi_gpu_config}")

    return multi_gpu_config


def _configure_logging(distributed_conf: _MultGPUConfig | None = None, log_dir: Path | None = None) -> None:
    # Configure logging.
    from holosoma.utils.logging import LoguruLoggingBridge

    logger.remove()
    is_main_process = distributed_conf is None or distributed_conf["global_rank"] == 0

    # logging to file (from all ranks)
    if log_dir is not None:
        fname = f"train_rank_{distributed_conf['global_rank']:02d}.log" if distributed_conf is not None else "train.log"
        log_path = log_dir / fname
        logger.add(str(log_path), level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default in rank0
    if is_main_process:
        console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    else:
        console_log_level = "ERROR"
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG if is_main_process else logging.ERROR)
    logging.getLogger().addHandler(LoguruLoggingBridge())


def split_envs_across_ranks(tyro_config: TrainingExperimentConfig) -> TrainingExperimentConfig:
    """Return ``tyro_config`` with ``run.num_envs`` divided among the distributed ranks.

    Call it before ``create_task()``: each rank builds only its share of the environments.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size <= 1:
        return tyro_config
    original_num_envs = tyro_config.run.num_envs
    tyro_config = dataclasses.replace(
        tyro_config, run=dataclasses.replace(tyro_config.run, num_envs=original_num_envs // world_size)
    )
    logger.info(
        f"Distributed training: each of {world_size} GPUs will run {tyro_config.run.num_envs} "
        f"environments (total across all GPUs: {original_num_envs})"
    )
    return tyro_config


@dataclasses.dataclass(frozen=True)
class _TrainingSetup:
    """Process group, run timestamp and experiment directory of one training run."""

    distributed_conf: _MultGPUConfig | None
    timestamp: str
    experiment_dir: Path


def _setup_training(tyro_config: TrainingExperimentConfig) -> _TrainingSetup:
    """Start the process group and route logs to the run's experiment directory."""
    from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp

    distributed_conf = _configure_multi_gpu()
    timestamp = get_timestamp()
    run = tyro_config.run
    experiment_dir = get_experiment_dir(run.logger, run.project, run.name, timestamp, task_name="locomotion")
    _configure_logging(distributed_conf=distributed_conf, log_dir=experiment_dir)
    return _TrainingSetup(distributed_conf, timestamp, experiment_dir)


def train_on_task(tyro_config: TrainingExperimentConfig, env: PolicyTask, setup: _TrainingSetup | None = None) -> None:
    """Train an agent on ``env``, a task built from ``tyro_config`` that the caller owns.

    ``setup`` comes from :func:`_setup_training`; it is created here when the caller has none.
    """
    import torch
    import torch.distributed as dist
    import wandb

    from holosoma.agents.base_algo.base_algo import BaseAlgo
    from holosoma.agents.distillation.checkpoint import infer_num_teachers_from_checkpoint
    from holosoma.cli import write_run_record

    if setup is None:
        setup = _setup_training(tyro_config)
    distributed_conf = setup.distributed_conf
    timestamp = setup.timestamp
    experiment_dir = setup.experiment_dir
    device: str = env.device
    is_distributed = distributed_conf is not None
    is_main_process = distributed_conf is None or distributed_conf["global_rank"] == 0

    logger_cfg = tyro_config.run.logger
    wandb_enabled = logger_cfg.type == "wandb"

    wandb_run_path: str | None = None

    # Configure wandb in rank 0
    if wandb_enabled and is_main_process:
        from holosoma.config_types.logger import WandbLoggerConfig

        assert isinstance(logger_cfg, WandbLoggerConfig), "Logger config must be WandbLoggerConfig when type is wandb"
        wandb_cfg = logger_cfg
        robot_type = tyro_config.robot.asset.robot_type
        default_run_name = f"{timestamp}_{tyro_config.run.name}_{wandb_cfg.group or 'default'}_{robot_type}"
        wandb_dir = Path(wandb_cfg.dir or (experiment_dir / ".wandb"))
        wandb_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving wandb logs to {wandb_dir}")

        # Only pass optional parameters when specified so wandb can fall back to environment defaults.
        wandb_kwargs: dict[str, Any] = {
            "project": tyro_config.run.project,
            "name": wandb_cfg.display_name or default_run_name,
            "config": dataclasses.asdict(tyro_config),
            "dir": str(wandb_dir),
            "mode": wandb_cfg.mode,
        }
        wandb_entity = os.getenv("WANDB_ENTITY") or wandb_cfg.entity
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity
        if wandb_cfg.group:
            wandb_kwargs["group"] = wandb_cfg.group
        if wandb_cfg.id:
            wandb_kwargs["id"] = wandb_cfg.id
        if wandb_cfg.tags:
            wandb_kwargs["tags"] = list(wandb_cfg.tags)
        if wandb_cfg.resume is not None:
            wandb_kwargs["resume"] = wandb_cfg.resume

        wandb.init(**wandb_kwargs)
        if wandb.run is not None:
            wandb_run_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"

    experiment_save_dir = experiment_dir
    experiment_save_dir.mkdir(exist_ok=True, parents=True)

    if is_main_process:
        logger.info(f"Saving config file to {experiment_save_dir}")
        config_path = experiment_save_dir / CONFIG_NAME
        tyro_config.save_config(str(config_path))
        write_run_record(experiment_save_dir)
        if wandb_enabled:
            wandb.save(str(config_path), base_path=experiment_save_dir)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=experiment_save_dir,
        multi_gpu_cfg=distributed_conf,
    )

    # Resolve teacher count before setup, when the policy allocates its slots.
    # Full checkpoints are inspected so resume does not require teacher paths again.
    teacher_checkpoints: list[str] = []
    if tyro_config.algo.teacher_checkpoint is not None:
        teacher_checkpoints = [s.strip() for s in tyro_config.algo.teacher_checkpoint.split(",") if s.strip()]
        if not teacher_checkpoints:
            raise ValueError("algo.teacher_checkpoint must contain at least one checkpoint path.")

    resume_checkpoint: Path | None = None
    resume_teacher_count: int | None = None
    if tyro_config.algo.resume_from is not None:
        resume_checkpoint = load_checkpoint(tyro_config.algo.resume_from, str(experiment_save_dir))
        checkpoint_data = torch.load(resume_checkpoint, map_location="cpu")
        if not isinstance(checkpoint_data, dict):
            raise TypeError("Training checkpoint must contain a dictionary.")
        resume_teacher_count = infer_num_teachers_from_checkpoint(checkpoint_data)
        tyro_config = dataclasses.replace(
            tyro_config, algo=dataclasses.replace(tyro_config.algo, resume_from=str(resume_checkpoint))
        )

    if teacher_checkpoints and resume_teacher_count is not None:
        if len(teacher_checkpoints) != resume_teacher_count:
            raise ValueError(
                f"Resume checkpoint contains {resume_teacher_count} teacher slots, but "
                f"{len(teacher_checkpoints)} teacher checkpoints were provided."
            )

    teacher_count = len(teacher_checkpoints) if teacher_checkpoints else resume_teacher_count
    if teacher_count is not None:
        if hasattr(algo, "num_teachers"):
            algo.num_teachers = teacher_count
        elif teacher_count > 1:
            raise RuntimeError(
                f"Training requires {teacher_count} teacher slots, "
                f"but algorithm {type(algo).__name__} has no num_teachers attribute; "
                f"only 1 teacher is supported."
            )

    algo.setup()
    algo.attach_checkpoint_metadata(tyro_config, wandb_run_path)
    if resume_checkpoint is not None:
        algo.load(str(resume_checkpoint))

    # Load teacher-only checkpoint(s) into the student-teacher policy, if the
    # algo supports it. Runs after `algo.load()` so a full-resume checkpoint
    # wins for student/critic/optimizer state, and we only overwrite the
    # frozen teacher branch(es) here.
    if teacher_checkpoints:
        if not hasattr(algo, "load_teacher"):
            raise RuntimeError(
                f"`algo.teacher_checkpoint` was set but algorithm "
                f"{type(algo).__name__} does not implement load_teacher()."
            )
        teacher_ckpt_paths = [str(load_checkpoint(tc, str(experiment_save_dir))) for tc in teacher_checkpoints]
        algo.load_teacher(teacher_ckpt_paths)

    # handle saving config
    algo.learn()

    # teardown wandb before SimApp closes ungracefully (IsaacLab)
    if is_main_process and wandb_enabled:
        logger.info("Shutting down wandb...")
        wandb.teardown()

    # shutdown dist before SimApp closes ungracefully (IsaacLab)
    if is_distributed:
        logger.info("Shutting down distributed processes...")
        dist.destroy_process_group()
    logger.info("Training complete; closing simulation resources.")


def _run_evaluation(
    env: PolicyTask,
    tyro_config: EvaluationExperimentConfig,
    *,
    saved_config: TrainingExperimentConfig,
    saved_wandb_path: str | None,
) -> None:
    """Load ``tyro_config.eval.checkpoint`` into the saved algorithm on ``env``, export it and evaluate it."""
    import os

    from holosoma.agents.base_algo.base_algo import BaseAlgo
    from holosoma.cli import write_run_record
    from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp

    run = tyro_config.run
    eval_log_dir = get_experiment_dir(run.logger, run.project, run.name, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))
    write_run_record(eval_log_dir)

    # Inject eval callbacks into algo config
    cb_configs = tyro_config.eval.callbacks.collect_active_callbacks()
    if cb_configs:
        object.__setattr__(saved_config.algo.config, "eval_callbacks", cb_configs)

    assert tyro_config.eval.checkpoint is not None
    checkpoint = load_checkpoint(tyro_config.eval.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(saved_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=env.device,
        env=env,
        config=saved_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    # Size the policy like the saved one (e.g. its teacher slots) before setup() allocates it.
    restore_structure = getattr(algo, "restore_structure_from_checkpoint", None)
    if restore_structure is not None:
        import torch

        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint_data, dict):
            raise TypeError("Evaluation checkpoint must contain a dictionary.")
        restore_structure(checkpoint_data)
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    exported_policy_dir_path = os.path.join(checkpoint_dir, "exported")
    os.makedirs(exported_policy_dir_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split("/")[-1]  # example: model_5000.pt
    exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")  # example: model_5000.onnx

    if tyro_config.eval.export_onnx:
        exported_onnx_path = os.path.join(exported_policy_dir_path, exported_onnx_name)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )

        algo.export(onnx_file_path=exported_onnx_path)
        logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

    algo.evaluate_policy(
        max_eval_steps=tyro_config.eval.max_eval_steps,
    )
