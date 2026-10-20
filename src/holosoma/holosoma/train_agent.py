from __future__ import annotations

import dataclasses
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

from loguru import logger

from holosoma.agents.distillation.checkpoint import infer_num_teachers_from_checkpoint
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import load_checkpoint
from holosoma.utils.helpers import get_class

if TYPE_CHECKING:
    from holosoma.policy_task import PolicyTask


class MultGPUConfig(TypedDict):
    global_rank: int
    local_rank: int
    world_size: int


def configure_multi_gpu() -> MultGPUConfig | None:
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

    multi_gpu_config: MultGPUConfig = {
        "global_rank": gpu_global_rank,
        "local_rank": gpu_local_rank,
        "world_size": gpu_world_size,
    }
    logger.info(f"Running with multi-GPU parameters: {multi_gpu_config}")

    return multi_gpu_config


def configure_logging(distributed_conf: MultGPUConfig | None = None, log_dir: Path | None = None) -> None:
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


def train(tyro_config: ExperimentConfig) -> None:
    """Train an agent on the task built from ``tyro_config``."""
    # Distribute environments across GPUs for proper multi-GPU training
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size > 1:
        original_num_envs = tyro_config.training.num_envs
        tyro_config = dataclasses.replace(
            tyro_config, training=dataclasses.replace(tyro_config.training, num_envs=original_num_envs // world_size)
        )
        logger.info(
            f"Distributed training: each of {world_size} GPUs will run {tyro_config.training.num_envs} "
            f"environments (total across all GPUs: {original_num_envs})"
        )

    try:
        with tyro_config.create_task() as env:
            train_on_task(tyro_config, cast("PolicyTask", env))
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Exception occurred during training: {e}\n{tb_str}")
        sys.exit(1)  # manually set exit code, not possible via isaacsim app.close()


def train_on_task(tyro_config: ExperimentConfig, env: PolicyTask) -> None:
    """Train an agent on ``env``, a task built from ``tyro_config`` that the caller owns."""
    import torch
    import torch.distributed as dist
    import wandb

    from holosoma.agents.base_algo.base_algo import BaseAlgo

    # Initialize process group
    distributed_conf: MultGPUConfig | None = configure_multi_gpu()
    device: str = env.device
    is_distributed = distributed_conf is not None
    is_main_process = distributed_conf is None or distributed_conf["global_rank"] == 0

    # Configure logger
    logger_cfg = tyro_config.logger
    wandb_enabled = logger_cfg.type == "wandb"

    # Compute experiment directory from logger and training config
    from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp

    timestamp = get_timestamp()
    experiment_dir = get_experiment_dir(logger_cfg, tyro_config.training, timestamp, task_name="locomotion")

    # Configure logging with experiment directory
    configure_logging(distributed_conf=distributed_conf, log_dir=experiment_dir)

    wandb_run_path: str | None = None

    # Configure wandb in rank 0
    if wandb_enabled and is_main_process:
        from holosoma.config_types.logger import WandbLoggerConfig

        assert isinstance(logger_cfg, WandbLoggerConfig), "Logger config must be WandbLoggerConfig when type is wandb"
        wandb_cfg = logger_cfg
        # Use training config for project/name, fallback to logger config, then defaults
        default_project = tyro_config.training.project or wandb_cfg.project or "default_project"
        default_run_name = (
            f"{timestamp}_{tyro_config.training.name or 'run'}_"
            f"{wandb_cfg.group or 'default'}_{tyro_config.robot.asset.robot_type}"
        )
        wandb_dir = Path(wandb_cfg.dir or (experiment_dir / ".wandb"))
        wandb_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving wandb logs to {wandb_dir}")

        # Only pass optional parameters when specified so wandb can fall back to environment defaults.
        wandb_kwargs: dict[str, Any] = {
            "project": wandb_cfg.project or default_project,
            "name": wandb_cfg.name or default_run_name,
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

    # For manager system, pre-process config AFTER env creation
    # (need managers to compute dims)
    observation_manager = getattr(env, "observation_manager", None)
    if observation_manager is None:
        raise RuntimeError(
            f"Manager environment {tyro_config.env_class} is missing observation_manager attribute. "
            "This should not happen if the environment is properly configured."
        )

    experiment_save_dir = experiment_dir
    experiment_save_dir.mkdir(exist_ok=True, parents=True)

    if is_main_process:
        logger.info(f"Saving config file to {experiment_save_dir}")
        config_path = experiment_save_dir / CONFIG_NAME
        tyro_config.save_config(str(config_path))
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
    if tyro_config.training.teacher_checkpoint is not None:
        teacher_checkpoints = [s.strip() for s in tyro_config.training.teacher_checkpoint.split(",") if s.strip()]
        if not teacher_checkpoints:
            raise ValueError("training.teacher_checkpoint must contain at least one checkpoint path.")

    resume_checkpoint: Path | None = None
    resume_teacher_count: int | None = None
    if tyro_config.training.checkpoint is not None:
        resume_checkpoint = load_checkpoint(tyro_config.training.checkpoint, str(experiment_save_dir))
        checkpoint_data = torch.load(resume_checkpoint, map_location="cpu")
        if not isinstance(checkpoint_data, dict):
            raise TypeError("Training checkpoint must contain a dictionary.")
        resume_teacher_count = infer_num_teachers_from_checkpoint(checkpoint_data)
        tyro_config = dataclasses.replace(
            tyro_config,
            training=dataclasses.replace(
                tyro_config.training,
                checkpoint=str(resume_checkpoint),
            ),
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
                f"`training.teacher_checkpoint` was set but algorithm "
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


def main() -> None:
    from holosoma.config_values.experiment import get_annotated_experiment_config
    from holosoma.utils.config_registry import parse_config

    # Pass the factory uncalled so parse_config builds it after plugins load.
    tyro_cfg = parse_config(get_annotated_experiment_config)
    train(tyro_cfg)


if __name__ == "__main__":
    main()
