#!/usr/bin/env python3
"""
Simulation Runner Script

This script provides a direct simulation runner for holosoma with bridge support without training
or evaluation environments.
"""

from __future__ import annotations

import os
import sys
import traceback

from loguru import logger

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.utils.eval_utils import init_eval_logging

_THREAD_LIMIT_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def limit_cpu_math_threads(config: RunSimConfig, device: str | None = None) -> None:
    """Serialize Torch's CPU intra-op pool for classic MuJoCo on cpu, unless the operator set a limit.

    Classic MuJoCo runs one env on the host, where every tensor is small enough that a machine-sized
    pool costs more per elementwise op in fork/join than the op itself. A no-op for Warp and
    IsaacSim, which keep this work on the GPU, and for training, which does not route through here.
    ``device`` is the resolved simulation device; it defaults to ``config.device``.
    """
    from holosoma.config_types.simulator import MujocoBackend
    from holosoma.utils.safe_torch_import import torch

    # Both halves are needed: ``mujoco_backend`` lives on the shared SimulatorInitConfig and
    # defaults to CLASSIC, so every backend's preset reports CLASSIC unless it overrides it. The
    # name is the same discriminator ``resolve_simulation_device`` uses.
    sim_config = config.simulator.config
    is_classic = sim_config.name == "mujoco" and getattr(sim_config, "mujoco_backend", None) == MujocoBackend.CLASSIC
    if device is None:
        device = config.device
    if not is_classic or device is None or not device.startswith("cpu"):
        return
    # Truthiness, not presence: OpenMP and MKL ignore an empty value, so an empty one is not an
    # operator's choice and must not suppress the cap.
    preset = [name for name in _THREAD_LIMIT_VARIABLES if os.environ.get(name)]
    if preset:
        logger.info(f"{', '.join(preset)} already set - leaving Torch at {torch.get_num_threads()} CPU thread(s)")
        return
    torch.set_num_threads(1)
    logger.info("Classic MuJoCo on cpu - limiting Torch to 1 CPU thread")


def run_simulation(config: RunSimConfig) -> None:
    """Build the configured simulation task and run it until shutdown.

    Parameters
    ----------
    config : RunSimConfig
        Configuration containing all simulation settings.
    """
    from holosoma.config_types.env import get_run_sim_env_config
    from holosoma.simulator.session import get_local_rank, resolve_simulation_device

    # Resolved as the task session will, so the CPU thread cap applies before the task is built.
    limit_cpu_math_threads(config, resolve_simulation_device(get_run_sim_env_config(config), get_local_rank()))

    try:
        with config.create_task() as task:
            task.run()
    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main function using tyro configuration with compositional subcommands."""
    # Initialize logging
    init_eval_logging()

    logger.info("Holosoma Direct Simulation Runner")
    logger.info("Compositional configuration via subcommands (like eval_agent.py)")

    from holosoma.utils.config_registry import parse_config

    config = parse_config(
        RunSimConfig,
        description="Run simulation with direct simulator control and bridge support.\n\n"
        "Usage: python -m holosoma.run_sim simulator:<sim> robot:<robot> terrain:<terrain>\n"
        "Examples:\n"
        "  python -m holosoma.run_sim # defaults \n"
        "  python -m holosoma.run_sim simulator:mujoco robot:t1_29dof_waist_wrist terrain:terrain_locomotion_plane\n"
        "  python -m holosoma.run_sim simulator:isaacsim robot:g1_29dof terrain:terrain_locomotion_mix",
    )

    # Run simulation directly with parsed config
    run_simulation(config)


if __name__ == "__main__":
    main()
