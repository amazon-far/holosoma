#!/usr/bin/env python3
"""
Simulation Runner Script

This script provides a direct simulation runner for holosoma with bridge support without training
or evaluation environments.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import traceback

from loguru import logger

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.utils.eval_utils import init_eval_logging
from holosoma.utils.sim_utils import DirectSimulation, graceful_simulation_signals, setup_simulation_environment

_THREAD_LIMIT_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def limit_cpu_math_threads(config: RunSimConfig) -> None:
    """Serialize Torch's CPU intra-op pool for classic MuJoCo on cpu, unless the operator set a limit.

    Classic MuJoCo runs one env on the host, where every tensor is small enough that a machine-sized
    pool costs more per elementwise op in fork/join than the op itself. A no-op for Warp and
    IsaacSim, which keep this work on the GPU, and for training, which does not route through here.
    """
    from holosoma.config_types.simulator import MujocoBackend
    from holosoma.utils.safe_torch_import import torch

    # Both halves are needed: ``mujoco_backend`` lives on the shared SimulatorInitConfig and
    # defaults to CLASSIC, so every backend's preset reports CLASSIC unless it overrides it. The
    # name is the same discriminator the device resolution above uses.
    sim_config = config.simulator.config
    is_classic = sim_config.name == "mujoco" and getattr(sim_config, "mujoco_backend", None) == MujocoBackend.CLASSIC
    if not is_classic or config.device is None or not config.device.startswith("cpu"):
        return
    # Truthiness, not presence: OpenMP and MKL ignore an empty value, so an empty one is not an
    # operator's choice and must not suppress the cap.
    preset = [name for name in _THREAD_LIMIT_VARIABLES if os.environ.get(name)]
    if preset:
        logger.info(f"{', '.join(preset)} already set - leaving Torch at {torch.get_num_threads()} CPU thread(s)")
        return
    torch.set_num_threads(1)
    logger.info("Classic MuJoCo on cpu - limiting Torch to 1 CPU thread")


@graceful_simulation_signals
def run_simulation(config: RunSimConfig) -> None:
    """Run simulation with direct simulator control.

    This function provides direct access to the simulator for continuous simulation
    with bridge support using the DirectSimulation class.

    Parameters
    ----------
    config : RunSimConfig
        Configuration containing all simulation settings.
    """
    # Resolve the simulation device from the backend when the user did not set one (device=None).
    # An explicit --device (including "cpu") is always honored.
    if config.device is None:
        from holosoma.config_types.simulator import MujocoBackend
        from holosoma.utils.safe_torch_import import torch

        sim_name = config.simulator.config.name
        is_warp = (
            hasattr(config.simulator.config, "mujoco_backend")
            and config.simulator.config.mujoco_backend == MujocoBackend.WARP
        )
        if is_warp:
            # MuJoCo Warp only runs on CUDA, so cpu is never a valid choice for it.
            resolved_device = "cuda:0"
            logger.info("Auto-detected MuJoCo Warp backend - setting device to cuda:0")
        elif sim_name == "isaacsim":
            # IsaacSim is GPU-first; prefer cuda:0 when it is actually available,
            # otherwise fall back to cpu (IsaacSim can run PhysX on CPU).
            if torch.cuda.is_available():
                resolved_device = "cuda:0"
                logger.info(f"Auto-detected {sim_name} backend with CUDA available - setting device to cuda:0")
            else:
                resolved_device = "cpu"
                logger.warning(f"{sim_name} backend selected but CUDA is unavailable - defaulting to cpu")
        else:
            # Classic MuJoCo (CPU physics engine).
            resolved_device = "cpu"
            logger.info(f"Auto-detected {sim_name} backend - setting device to cpu")

        config = dataclasses.replace(config, device=resolved_device)

    # After resolution, so an explicit --device cpu is covered too.
    limit_cpu_math_threads(config)

    logger.info("Starting Holosoma Direct Simulation...")
    logger.info(f"Robot: {config.robot.asset.robot_type}")
    logger.info(f"Simulator: {config.simulator._target_}")
    logger.info(f"Terrain: {config.terrain.terrain_term.mesh_type} ({config.terrain.terrain_term.func})")

    try:
        # Use shared utils for setup
        env, device, simulation_app = setup_simulation_environment(config, device=config.device)

        # Create and run direct simulation using context manager for automatic clean-up
        with DirectSimulation(config, env, device, simulation_app) as sim:
            sim.run()

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
