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

g1_23dof = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_23dof_manager"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.g1_23dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum,
    reward=reward.g1_23dof_loco,
    nightly=NightlyConfig(
        iterations=5000,
        metrics={"Episode/rew_tracking_ang_vel": [0.7, "inf"], "Episode/rew_tracking_lin_vel": [0.55, "inf"]},
    ),
)

g1_23dof_fast_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_23dof_fast_sac_manager"),
    algo=replace(algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=50000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.g1_23dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac,
    reward=reward.g1_23dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)

g1_29dof = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_manager"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum,
    reward=reward.g1_29dof_loco,
    nightly=NightlyConfig(
        iterations=5000,
        metrics={"Episode/rew_tracking_ang_vel": [0.7, "inf"], "Episode/rew_tracking_lin_vel": [0.55, "inf"]},
    ),
)

g1_29dof_fast_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fast_sac_manager"),
    algo=replace(algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=50000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac,
    reward=reward.g1_29dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)

__all__ = ["g1_23dof", "g1_23dof_fast_sac", "g1_29dof", "g1_29dof_fast_sac"]
