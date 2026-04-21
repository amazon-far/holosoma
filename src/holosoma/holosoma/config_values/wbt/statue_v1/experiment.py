from dataclasses import replace

from holosoma.config_types.algo import MotionEncoderConfig
from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    terrain,
)
from holosoma.config_values.wbt.statue_v1 import command as statue_v1_command
from holosoma.config_values.wbt.statue_v1 import termination as statue_v1_termination

# Statue v1 RMD WBT with future motion (motion encoder, stiff control from MJCF kp/kd)
statue_v1_rmd_wbt_future_motion = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="statue_v1_rmd_wbt_future_motion",
        num_envs=8192,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=40000,
            save_interval=1000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            init_at_random_ep_len=False,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
            module_dict=replace(
                algo.ppo.config.module_dict,
                actor=replace(
                    algo.ppo.config.module_dict.actor,
                    type="MLPWithMotionEncoder",
                    input_dim=["actor_obs"],
                    layer_config=replace(
                        algo.ppo.config.module_dict.actor.layer_config,
                        hidden_dims=[768, 512, 256],
                        activation="SiLU",
                    ),
                ),
                critic=replace(
                    algo.ppo.config.module_dict.critic,
                    type="MLPWithMotionEncoder",
                    input_dim=["critic_obs"],
                    layer_config=replace(
                        algo.ppo.config.module_dict.critic.layer_config,
                        hidden_dims=[768, 512, 256],
                        activation="SiLU",
                    ),
                ),
                motion_encoder=MotionEncoderConfig(
                    input_dim=0,  # Will be auto-calculated
                    hidden_dim=60,
                    output_dim=128,
                    num_timesteps=20,
                    activation="SiLU",
                ),
            ),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.statue_v1_rmd,
        asset=replace(robot.statue_v1_rmd.asset, enable_self_collisions=True),
        init_state=replace(robot.statue_v1_rmd.init_state, pos=[0.0, 0.0, 0.89]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation_future_motion,
    action=action.g1_29dof_joint_pos,
    termination=statue_v1_termination.statue_v1_rmd_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=statue_v1_command.statue_v1_rmd_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_reward,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.16, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.20, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.45, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.30, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.30, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.02, "inf"],
        },
    ),
)

__all__ = [
    "statue_v1_rmd_wbt_future_motion",
]
