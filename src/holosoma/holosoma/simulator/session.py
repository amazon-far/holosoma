"""Task construction and resource ownership for every workflow."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any, TypeVar

from loguru import logger

from holosoma.config_types.env import EnvConfig
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import (
    close_simulation_resources,
    exit_on_sigterm,
    setup_isaaclab_launcher,
    setup_keyboard_listener,
    setup_simulator_imports,
)
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type

if TYPE_CHECKING:
    from holosoma.envs.base_task.base_task import BaseTask

_TaskT = TypeVar("_TaskT", bound="BaseTask")


def get_local_rank() -> int | None:
    """Return this process's local distributed rank, or None when not distributed."""
    return int(os.environ.get("LOCAL_RANK", "0")) if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None


def get_process_seed(config: EnvConfig) -> int:
    """Return the configured seed offset by this process's global rank."""
    global_rank = int(os.environ.get("RANK", "0")) if int(os.environ.get("WORLD_SIZE", "1")) > 1 else 0
    return config.training.seed + global_rank


def resolve_simulation_device(config: EnvConfig, local_rank: int | None = None) -> str:
    """Resolve the task device from the request, the process rank, and the simulator backend."""
    from holosoma.config_types.simulator import MujocoBackend

    sim_config = config.simulator.config
    simulator_name = sim_config.name
    is_mujoco_warp = (
        simulator_name == "mujoco"
        and getattr(sim_config, "mujoco_backend", MujocoBackend.CLASSIC) == MujocoBackend.WARP
    )

    if config.device is not None:
        device = config.device
    elif local_rank is not None:
        device = f"cuda:{local_rank}"
    elif is_mujoco_warp:
        device = "cuda:0"
    elif simulator_name == "mujoco":
        device = "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if is_mujoco_warp and not device.startswith("cuda"):
        raise ValueError(f"MuJoCo Warp requires a CUDA device, got {device!r}.")
    if local_rank is not None and device != f"cuda:{local_rank}":
        raise ValueError(f"Distributed local rank {local_rank} requires device 'cuda:{local_rank}', got {device!r}.")
    return device


@contextmanager
def task_session(config: EnvConfig, required_class: type[_TaskT]) -> Iterator[_TaskT]:
    """Own signal handling, provider launch, task construction, and teardown.

    Parameters
    ----------
    config : EnvConfig
        Configuration of the task to build.
    required_class : type
        Class that ``config.env_class`` must name or subclass.

    Yields
    ------
    BaseTask
        The constructed task. It is closed before the provider application on exit.
    """
    with exit_on_sigterm():
        setup_simulator_imports(config)
        device = resolve_simulation_device(config, get_local_rank())
        logger.info(f"Device: {device}")
        if device.startswith("cuda"):
            torch.cuda.set_device(device)

        simulation_app: Any | None = None
        task: _TaskT | None = None
        try:
            if get_simulator_type() == SimulatorType.ISAACSIM:
                simulation_app = setup_isaaclab_launcher(config, device)

            task_class = get_class(config.env_class)
            if not issubclass(task_class, required_class):
                raise TypeError(f"env_class {config.env_class!r} is not a {required_class.__name__} subclass.")
            logger.info(f"Creating task: {config.env_class}")
            task = task_class(config)

            if not config.training.headless:
                setup_keyboard_listener(task)
            yield task
        except BaseException:
            try:
                close_simulation_resources(task, simulation_app)
            except Exception:
                with suppress(Exception):
                    logger.exception("Task cleanup failed while preserving the active exception")
            raise
        else:
            close_simulation_resources(task, simulation_app)
