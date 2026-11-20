"""Fixed-base G1 arm-tracking experiment (GAC controller).

Used by tools/generate_arm_trajectories.py, which runs this environment to
turn sampled joint targets into end-effector trajectories by forward kinematics.
"""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_values import algo, robot, simulator, terrain
from holosoma.config_values.arm_tracking.g1 import (
    action,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    termination,
)


g1_arm_tracking_gac = ExperimentConfig(
    env_class="holosoma.envs.arm_tracking.arm_tracking_manager.ArmTrackingManager",
    training=TrainingConfig(project="g1-arm-tracking", name="g1_arm_tracking"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=10000, use_symmetry=False)),
    simulator=simulator.isaacsim,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_arm_tracking_observation_with_controller_cmd,
    action=action.g1_arm_tracking_action_gac,
    termination=termination.g1_arm_tracking_termination,
    randomization=randomization.g1_arm_tracking_randomization,
    command=command.g1_arm_tracking_command,
    curriculum=curriculum.g1_arm_tracking_curriculum,
    reward=reward.g1_arm_tracking_reward_controller,
)


# Backward-compatible alias expected by global experiment registry.
g1_arm_tracking = g1_arm_tracking_gac

