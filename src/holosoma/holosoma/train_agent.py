from __future__ import annotations

import dataclasses
import logging
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypedDict, cast

import tyro
from loguru import logger

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    init_sim_imports,
    load_checkpoint,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


class TrainingContext:
    """Context manager for training lifecycle and resource management."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.simulation_app: Any | None = None

    def __enter__(self):
        self.simulation_app = init_sim_imports(self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        close_simulation_app(self.simulation_app)

    def train(self) -> None:
        """Train using this context's sim app."""
        train(self.config, training_context=self)


@contextmanager
def training_context(config: ExperimentConfig):
    """Context manager function for training."""
    with TrainingContext(config) as ctx:
        yield ctx


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


def get_device(config, distributed_conf: MultGPUConfig | None) -> str:
    import torch

    is_config_device_specified = hasattr(config, "device") and config.device is not None
    is_multi_gpu = distributed_conf is not None

    if is_config_device_specified:
        if is_multi_gpu and config.device != cast("dict", distributed_conf)["local_rank"]:
            raise ValueError(
                f"Device specified in config ({config.device}) \
                              does not match expected local rank {cast('dict', distributed_conf)['local_rank']}"
            )
        device = config.device
    elif is_multi_gpu:
        device = f"cuda:{cast('dict', distributed_conf)['local_rank']}"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return device


def configure_logging(distributed_conf: MultGPUConfig | None = None, log_dir: Path | None = None):
    from holosoma.utils.logging import LoguruLoggingBridge

    logger.remove()
    is_main_process = distributed_conf is None or distributed_conf["global_rank"] == 0

    if log_dir is not None:
        fname = f"train_rank_{distributed_conf['global_rank']:02d}.log" if distributed_conf is not None else "train.log"
        log_path = log_dir / fname
        logger.add(str(log_path), level="DEBUG")

    if is_main_process:
        console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    else:
        console_log_level = "ERROR"
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG if is_main_process else logging.ERROR)
    logging.getLogger().addHandler(LoguruLoggingBridge())

    """
    初始化 simulator/GPU/logger
创建 env
创建 algo
加载 checkpoint
调用 algo.learn()
清理资源
    """
def train(tyro_config: ExperimentConfig, training_context: TrainingContext | None = None) -> None:
    
    """
    Train an agent with optional context for sim app management.

    Parameters
    ----------
    training_context : Optional[TrainingContext]
        Optional training context with pre-initialized sim app.
        If None, creates and manages sim app automatically.
    """
#初始化仿真app，如果传入了training_context则使用其中的sim app，否则自动创建一个新的sim app，并在训练结束后关闭它。
    if training_context is not None:
        simulation_app = training_context.simulation_app
        auto_close = False
    else:
        simulation_app = init_sim_imports(tyro_config)
        auto_close = True
#延迟到这里再导入torch和wandb等库，以避免在不需要训练时就加载这些库
# （例如在只使用环境交互的脚本中）。同时，这些库可能依赖于已经初始化的仿真app，因此必须在初始化app之后导入。
    try:
        import torch
        import torch.distributed as dist
        import wandb
#导入训练算法基类和随机种子工具。
        from holosoma.agents.base_algo.base_algo import BaseAlgo
        from holosoma.utils.common import seeding
#配置 GPU / 多 GPU训练设置，获取设备字符串，并确定是否为分布式训练以及当前进程是否为主进程。
        distributed_conf: MultGPUConfig | None = configure_multi_gpu()
        device: str = get_device(tyro_config, distributed_conf)
        is_distributed = distributed_conf is not None
        is_main_process = distributed_conf is None or distributed_conf["global_rank"] == 0
#取 logger 配置，并检查是否启用 wandb 记录训练日志。
# 如果启用 wandb，则在主进程上初始化 wandb，并设置日志目录、项目
        logger_cfg = tyro_config.logger
        wandb_enabled = logger_cfg.type == "wandb"
        #创建日志/模型保存目录，路径基于实验配置和当前时间戳，以确保每次运行都有一个唯一的目录。
        from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp

        timestamp = get_timestamp()
        experiment_dir = get_experiment_dir(logger_cfg, tyro_config.training, timestamp, task_name="locomotion")

        configure_logging(distributed_conf=distributed_conf, log_dir=experiment_dir)
#设置随机种子，基于配置中的种子值，并在分布式训练中为每个进程添加一个偏移量，以确保不同进程使用不同的随机数序列。
        seed = tyro_config.training.seed
        if distributed_conf is not None:
            seed += distributed_conf["global_rank"]
        seeding(seed, torch_deterministic=tyro_config.training.torch_deterministic)
#初始化 wandb 相关变量，如果启用了 wandb 并且当前进程是主进程，则根据配置设置 wandb 项目、运行名称、日志目录等参数，并初始化 wandb。
        wandb_run_path: str | None = None

        if wandb_enabled and is_main_process:
            from holosoma.config_types.logger import WandbLoggerConfig

            assert isinstance(logger_cfg, WandbLoggerConfig), (
                "Logger config must be WandbLoggerConfig when type is wandb"
            )
            wandb_cfg = logger_cfg
            default_project = tyro_config.training.project or wandb_cfg.project or "default_project"
            default_run_name = (
                f"{timestamp}_{tyro_config.training.name or 'run'}_"
                f"{wandb_cfg.group or 'default'}_{tyro_config.robot.asset.robot_type}"
            )
            wandb_dir = Path(wandb_cfg.dir or (experiment_dir / ".wandb"))
            wandb_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving wandb logs to {wandb_dir}")

            wandb_kwargs: dict[str, Any] = {
                "project": wandb_cfg.project or default_project,
                "name": wandb_cfg.name or default_run_name,
                "config": dataclasses.asdict(tyro_config),
                "dir": str(wandb_dir),
                "mode": wandb_cfg.mode,
            }
            if wandb_cfg.entity:
                wandb_kwargs["entity"] = wandb_cfg.entity
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
#多 GPU 时切分环境数量，确保每个 GPU 运行的环境数量是总环境数量除以 GPU 数量，并在日志中记录每个 GPU 运行的环境数量。
        if distributed_conf is not None:
            original_num_envs = tyro_config.training.num_envs
            num_envs = original_num_envs // distributed_conf["world_size"]
            tyro_config = dataclasses.replace(
                tyro_config, training=dataclasses.replace(tyro_config.training, num_envs=num_envs)
            )
            logger.info(
                f"Distributed training: GPU {distributed_conf['global_rank']} will run {tyro_config.training.num_envs} "
                f"environments (total across all GPUs: {original_num_envs})"
            )

        env_target = tyro_config.env_class

        tyro_env_config = get_tyro_env_config(tyro_config)
        #根据环境类路径字符串获取环境类，并使用从配置生成的环境配置实例化环境对象，同时传入设备参数。
        env = get_class(env_target)(tyro_env_config, device=device)
#检查环境对象是否具有 observation_manager 属性，如果没有则抛出运行时错误。这是为了确保环境正确配置并提供必要的功能。
        observation_manager = getattr(env, "observation_manager", None)
        if observation_manager is None:
            raise RuntimeError(
                f"Manager environment {env_target} is missing observation_manager attribute. "
                "This should not happen if the environment is properly configured."
            )
#保存实验配置文件到日志目录，并在启用 wandb 时将配置文件保存到 wandb 运行目录中，以便在 wandb 仪表板上查看和跟踪实验配置。
        experiment_save_dir = experiment_dir
        experiment_save_dir.mkdir(exist_ok=True, parents=True)

        if is_main_process:
            logger.info(f"Saving config file to {experiment_save_dir}")
            config_path = experiment_save_dir / CONFIG_NAME
            tyro_config.save_config(str(config_path))
            if wandb_enabled:
                wandb.save(str(config_path), base_path=experiment_save_dir)
#根据配置中的算法类路径字符串获取算法类，并实例化算法对象，传入设备、环境、算法配置、日志目录和多 GPU 配置等参数。然后调用算法的 setup 方法进行任何必要的初始化，并附加检查点元数据以便后续保存和加载检查点。
        algo_class = get_class(tyro_config.algo._target_)
        algo: BaseAlgo = algo_class(
            device=device,
            env=env,
            config=tyro_config.algo.config,
            log_dir=experiment_save_dir,
            multi_gpu_cfg=distributed_conf,
        )
        #算法的 setup 方法通常用于设置优化器、学习率调度器、模型保存路径等训练相关的资源和配置。附加检查点元数据可以包括当前训练步骤、环境配置、算法配置等信息，这些信息对于后续加载检查点并恢复训练状态非常有用。
        algo.setup()
        #挂载检查点元数据，并根据配置加载检查点（如果指定了检查点路径），然后调用算法的 learn 方法开始训练过程。
        algo.attach_checkpoint_metadata(tyro_config, wandb_run_path)
        #如果有 checkpoint，就加载 checkpoint，并更新配置中的 checkpoint 路径，以便算法知道从哪里加载模型权重和训练状态。
        if tyro_config.training.checkpoint is not None:
            loaded_checkpoint = load_checkpoint(tyro_config.training.checkpoint, str(experiment_save_dir))
            tyro_config = dataclasses.replace(
                tyro_config, training=dataclasses.replace(tyro_config.training, checkpoint=str(loaded_checkpoint))
            )
            algo.load(loaded_checkpoint)
#调用算法的 learn 方法开始训练过程。这个方法通常包含训练循环，在其中算法与环境交互、收集经验、更新模型参数，并记录训练进度和性能指标。
        algo.learn()
#训练完成后，如果启用了 wandb 并且当前进程是主进程，则调用 wandb.teardown() 来正确关闭 wandb 运行并确保所有日志和数据都被正确保存和上传。
        if is_main_process and wandb_enabled:
            logger.info("Shutting down wandb...")
            wandb.teardown()

        if is_distributed:
            logger.info("Shutting down distributed processes...")
            dist.destroy_process_group()
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Exception occurred during training: {e}\n{tb_str}")
        sys.exit(1)
        #在训练过程中捕获任何异常，记录错误消息和堆栈跟踪，并以非零状态码退出程序，以便在训练失败时能够正确地报告错误。
    finally:
        if auto_close:
            close_simulation_app(simulation_app)

    logger.info("Training shutdown complete.")


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    print(tyro_cfg.curriculum)
    train(tyro_cfg)


if __name__ == "__main__":
    main()
