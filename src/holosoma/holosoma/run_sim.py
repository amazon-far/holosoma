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


def _apply_robot_specific_run_sim_defaults(config: RunSimConfig) -> RunSimConfig:
    """Apply direct-simulation defaults that are safer for specific robot assets."""
    if config.robot.asset.robot_type != "h1":
        return config

    gantry_cfg = config.simulator.config.virtual_gantry
    h1_gantry = dataclasses.replace(
        gantry_cfg,
        enabled=gantry_cfg.enabled,
        attachment_body_names=["torso_link", "pelvis"],
        stiffness=250.0,
        damping=220.0,
        height=3.2,
    )
    simulator_config = dataclasses.replace(config.simulator.config, virtual_gantry=h1_gantry)
    simulator = dataclasses.replace(config.simulator, config=simulator_config)
    robot_asset = dataclasses.replace(config.robot.asset, enable_self_collisions=True)
    robot = dataclasses.replace(config.robot, asset=robot_asset)
    logger.info(
        "Applying H1 direct-simulation gantry defaults: "
        f"attachment_body_names={h1_gantry.attachment_body_names}, "
        f"stiffness={h1_gantry.stiffness}, damping={h1_gantry.damping}, "
        f"height={h1_gantry.height}, enable_self_collisions={robot_asset.enable_self_collisions}"
    )
    return dataclasses.replace(config, simulator=simulator, robot=robot)


def run_simulation(config: RunSimConfig):
    """Run simulation with direct simulator control.

    This function provides direct access to the simulator for continuous simulation
    with bridge support using the DirectSimulation class.

    Parameters
    ----------
    config : RunSimConfig
        Configuration containing all simulation settings.
    """
    config = _apply_robot_specific_run_sim_defaults(config)

    # Auto-set device for GPU-accelerated backends if still on default CPU
    if config.device == "cpu":
        # Check if using Warp backend (requires CUDA)
        if hasattr(config.simulator.config, "mujoco_backend"):
            from holosoma.config_types.simulator import MujocoBackend  # noqa: PLC0415 -- deferred

            if config.simulator.config.mujoco_backend == MujocoBackend.WARP:
                logger.info("Auto-detected MuJoCo Warp backend - setting device to cuda:0")
                config = dataclasses.replace(config, device="cuda:0")

    config = dataclasses.replace(config, device=config.device)

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
