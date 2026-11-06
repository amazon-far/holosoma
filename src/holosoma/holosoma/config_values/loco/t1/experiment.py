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

t1_29dof = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-t1-manager", name="t1_29dof_manager"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True)),
    simulator=simulator.isaacsim,
    robot=robot.t1_29dof_waist_wrist,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.t1_29dof_loco_single_wolinvel,
    action=action.t1_29dof_joint_pos,
    termination=termination.t1_29dof_termination,
    randomization=randomization.t1_29dof_randomization,
    command=command.t1_29dof_command,
    curriculum=curriculum.t1_29dof_curriculum,
    reward=reward.t1_29dof_loco,
    nightly=NightlyConfig(
        iterations=10000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.75, "inf"]},
    ),
)

t1_29dof_fast_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-t1-manager", name="t1_29dof_fast_sac_manager"),
    algo=replace(
        algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=100000, use_symmetry=True)
    ),
    simulator=simulator.isaacsim,
    robot=robot.t1_29dof_waist_wrist,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.t1_29dof_loco_single_wolinvel,
    action=action.t1_29dof_joint_pos,
    termination=termination.t1_29dof_termination,
    randomization=randomization.t1_29dof_randomization,
    command=command.t1_29dof_command,
    curriculum=curriculum.t1_29dof_curriculum_fast_sac,
    reward=reward.t1_29dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.65, "inf"], "Episode/rew_tracking_lin_vel": [0.9, "inf"]},
    ),
)

__all__ = ["t1_29dof", "t1_29dof_fast_sac"]
