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
from holosoma.config_values.wbt.statue import command as statue_command
from holosoma.config_values.wbt.statue import termination as statue_termination

# Statue WBT with stiff control (high kp/kd, action_scale=1.0)
statue_v0_1_wbt_stiff = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="statue_v0_1_wbt_stiff",
        num_envs=8192,  # Reduced from 8192 due to complex mesh collisions causing slow initialization
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
        robot.statue_v0_1_stiff,
        asset=replace(robot.statue_v0_1_stiff.asset, enable_self_collisions=True),
        init_state=replace(robot.statue_v0_1_stiff.init_state, pos=[0.0, 0.0, 0.885]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,  # Reuse g1 observation for now
    action=action.g1_29dof_joint_pos,  # Reuse g1 action for now
    termination=statue_termination.statue_v0_1_wbt_termination,  # Statue-specific termination
    randomization=randomization.g1_29dof_wbt_randomization,  # Reuse g1 randomization for now
    command=statue_command.statue_v0_1_wbt_command,  # Statue-specific command with waist_pitch_link
    curriculum=curriculum.g1_29dof_wbt_curriculum,  # Reuse g1 curriculum for now
    reward=reward.g1_29dof_wbt_reward,  # Reuse g1 reward for now
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

# Statue WBT with compliant control (similar to g1, action_scale=0.25)
statue_v0_1_wbt_compliant = replace(
    statue_v0_1_wbt_stiff,
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="statue_v0_1_wbt_compliant",
        num_envs=8192,  # Reduced from 8192 due to complex mesh collisions causing slow initialization
    ),
    robot=replace(
        robot.statue_v0_1,  # Use compliant version
        asset=replace(robot.statue_v0_1.asset, enable_self_collisions=True),
        init_state=replace(robot.statue_v0_1.init_state, pos=[0.0, 0.0, 0.885]),
    ),
)

# Statue WBT with future motion (with motion encoder, based on stiff config)
statue_v0_1_wbt_future_motion = replace(
    statue_v0_1_wbt_stiff,
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="statue_v0_1_wbt_future_motion",
        num_envs=8192,  # Reduced from 8192 due to complex mesh collisions causing slow initialization
    ),
    observation=observation.g1_29dof_wbt_observation_future_motion,
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
                    input_dim=["actor_obs"],  # future_motion_targets handled by motion encoder
                    layer_config=replace(
                        algo.ppo.config.module_dict.actor.layer_config,
                        hidden_dims=[768, 512, 256],
                        activation="SiLU",
                    ),
                ),
                critic=replace(
                    algo.ppo.config.module_dict.critic,
                    type="MLPWithMotionEncoder",
                    input_dim=["critic_obs"],  # future_motion_targets handled by motion encoder
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
)

__all__ = [
    "statue_v0_1_wbt_stiff",
    "statue_v0_1_wbt_compliant",
    "statue_v0_1_wbt_future_motion",
]

"""
Example usage:
python src/holosoma/holosoma/train_agent.py \
    exp:statue-v0-1-wbt-stiff \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/statue_v0_1/whole_body_tracking/walk_chunk_0000_mj.npz"

python src/holosoma/holosoma/train_agent.py \
    exp:statue-v0-1-wbt-compliant \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/statue_v0_1/whole_body_tracking/walk_chunk_0000_mj.npz"
"""
