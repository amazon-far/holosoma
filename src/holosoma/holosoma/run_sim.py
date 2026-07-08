#!/usr/bin/env python3
"""
Simulation Runner Script

This script provides a direct simulation runner for holosoma with bridge support without training
or evaluation environments.
"""

import dataclasses
import sys
import traceback

import tyro
from loguru import logger

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.utils.eval_utils import init_eval_logging
from holosoma.utils.sim_utils import DirectSimulation, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


def run_simulation(config: RunSimConfig):
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
        elif sim_name in ("isaacsim", "isaacgym"):
            # Isaac backends are GPU-first; prefer cuda:0 when it is actually available,
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

    # Parse configuration with tyro - same pattern as ExperimentConfig
    config = tyro.cli(
        RunSimConfig,
        description="Run simulation with direct simulator control and bridge support.\n\n"
        "Usage: python -m holosoma.run_sim simulator:<sim> robot:<robot> terrain:<terrain>\n"
        "Examples:\n"
        "  python -m holosoma.run_sim # defaults \n"
        "  python -m holosoma.run_sim simulator:mujoco robot:t1_29dof_waist_wrist terrain:terrain_locomotion_plane\n"
        "  python -m holosoma.run_sim simulator:isaacgym robot:g1_29dof terrain:terrain_locomotion_mix",
        config=TYRO_CONIFG,
    )

    # Run simulation directly with parsed config
    run_simulation(config)


if __name__ == "__main__":
    main()
