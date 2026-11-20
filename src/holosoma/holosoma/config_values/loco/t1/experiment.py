from dataclasses import replace

from holosoma.config_types.algo import NightlyConfig
from holosoma.config_types.experiment import TrainingExperimentConfig
from holosoma.config_types.run import RunConfig
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

t1_29dof = TrainingExperimentConfig(
    task="holosoma.envs.policy_task:PolicyTask.train",
    simulator=simulator.isaacsim,
    robot=robot.t1_29dof_waist_wrist,
    terrain=terrain.terrain_locomotion_mix,
    termination=termination.t1_29dof_termination,
    randomization=randomization.t1_29dof_randomization,
    observation=observation.t1_29dof_loco_single_wolinvel,
    action=action.t1_29dof_joint_pos,
    command=command.t1_29dof_command,
    curriculum=curriculum.t1_29dof_curriculum,
    reward=reward.t1_29dof_loco,
    algo=replace(
        algo.ppo,
        config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True),
        nightly=NightlyConfig(
            iterations=10000,
            metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.75, "inf"]},
        ),
    ),
    run=RunConfig(project="hv-t1-manager", name="t1_29dof_manager"),
)

t1_29dof_fast_sac = TrainingExperimentConfig(
    task="holosoma.envs.policy_task:PolicyTask.train",
    simulator=simulator.isaacsim,
    robot=robot.t1_29dof_waist_wrist,
    terrain=terrain.terrain_locomotion_mix,
    termination=termination.t1_29dof_termination,
    randomization=randomization.t1_29dof_randomization,
    observation=observation.t1_29dof_loco_single_wolinvel,
    action=action.t1_29dof_joint_pos,
    command=command.t1_29dof_command,
    curriculum=curriculum.t1_29dof_curriculum_fast_sac,
    reward=reward.t1_29dof_loco_fast_sac,
    algo=replace(
        algo.fast_sac,
        config=replace(algo.fast_sac.config, num_learning_iterations=100000, use_symmetry=True),
        nightly=NightlyConfig(
            iterations=50000,
            metrics={"Episode/rew_tracking_ang_vel": [0.65, "inf"], "Episode/rew_tracking_lin_vel": [0.9, "inf"]},
        ),
    ),
    run=RunConfig(project="hv-t1-manager", name="t1_29dof_fast_sac_manager"),
)

__all__ = ["t1_29dof", "t1_29dof_fast_sac"]
