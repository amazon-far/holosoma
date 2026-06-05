"""Locomotion experiment presets for the H1 robot."""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

h1_19dof = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-h1-manager", name="h1_19dof_manager"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.h1_19dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.h1_19dof_loco_single_wolinvel,
    action=action.h1_19dof_joint_pos,
    termination=termination.h1_19dof_termination,
    randomization=randomization.h1_19dof_randomization,
    command=command.h1_19dof_command,
    curriculum=curriculum.h1_19dof_curriculum,
    reward=reward.h1_19dof_loco,
    nightly=NightlyConfig(
        iterations=10000,
        metrics={"Episode/rew_tracking_ang_vel": [0.5, "inf"], "Episode/rew_tracking_lin_vel": [0.5, "inf"]},
    ),
)

h1_19dof_fast_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-h1-manager", name="h1_19dof_fast_sac_manager"),
    algo=replace(algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=50000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.h1_19dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.h1_19dof_loco_single_wolinvel,
    action=action.h1_19dof_joint_pos,
    termination=termination.h1_19dof_termination,
    randomization=randomization.h1_19dof_randomization,
    command=command.h1_19dof_command,
    curriculum=curriculum.h1_19dof_curriculum_fast_sac,
    reward=reward.h1_19dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.6, "inf"], "Episode/rew_tracking_lin_vel": [0.7, "inf"]},
    ),
)

__all__ = ["h1_19dof", "h1_19dof_fast_sac"]

