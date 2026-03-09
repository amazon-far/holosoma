"""PhUMA multi-motion experiment config for G1 robot."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MultiMotionConfig, NoiseToInitialPoseConfig
from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

# ── PhUMA hyperparameters (easy to tweak) ──
NUM_ENVS = 8192
POOL_SIZE = NUM_ENVS
RESAMPLE_INTERVAL = 5000           # resample pool from disk every N env steps (~208 iterations)
MIN_MOTION_LENGTH = 30             # skip motions shorter than this (frames)
NUM_LEARNING_ITERATIONS = 500000
SAVE_INTERVAL = 1000
EVAL_INTERVAL = SAVE_INTERVAL

# PhUMA-specific init pose noise config (same as existing WBT)
phuma_init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.1, 0.1, 0.05],
    root_ang_vel=[0.1, 0.1, 0.1],
    object_pos=[0.0, 0.0, 0.0],
)

phuma_motion_config = MultiMotionConfig(
    motion_dir="/path/to/g1_npz/",  # Override via CLI
    split_file="",  # Override via CLI, e.g. "/path/to/split/phuma_train.txt"
    pool_size=POOL_SIZE,
    resample_interval=RESAMPLE_INTERVAL,
    min_motion_length=MIN_MOTION_LENGTH,
    body_names_to_track=[
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ],
    body_name_ref=["torso_link"],
    start_at_timestep_zero_prob=0.2,
    freeze_at_timestep_zero_prob=0.95,
    enable_default_pose_prepend=False,
    enable_default_pose_append=False,
    noise_to_initial_pose=phuma_init_pose_config,
)

g1_29dof_phuma_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MultiMotionCommand",
            params={
                "motion_config": phuma_motion_config,
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MultiMotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MultiMotionCommand",
        )
    },
)

# Eval callback config for success rate measurement
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass(frozen=True)
class _SuccessRateCbConfig:
    _target_: str = "holosoma.agents.callbacks.success_rate_callback.SuccessRateCallback"


_phuma_eval_callbacks = {"success_rate": _SuccessRateCbConfig()}

g1_29dof_phuma = ExperimentConfig(
    training=TrainingConfig(
        project="PhUMA",
        name="g1_29dof_phuma",
        num_envs=NUM_ENVS,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=NUM_LEARNING_ITERATIONS,
            save_interval=SAVE_INTERVAL,
            entropy_coef=0.005,
            init_noise_std=1.0,
            init_at_random_ep_len=False,
            use_symmetry=False,
            eval_interval=EVAL_INTERVAL,
            eval_callbacks=_phuma_eval_callbacks,
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
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=g1_29dof_phuma_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_reward,
)

# PhUMA with future motion encoder (for train_future_diverse_action_scale.sh)
from holosoma.config_types.algo import MotionEncoderConfig

g1_29dof_phuma_future_motion = replace(
    g1_29dof_phuma,
    training=replace(g1_29dof_phuma.training, name="g1_29dof_phuma_future_motion"),
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=NUM_LEARNING_ITERATIONS,
            save_interval=SAVE_INTERVAL,
            entropy_coef=0.005,
            init_noise_std=1.0,
            init_at_random_ep_len=False,
            use_symmetry=False,
            eval_interval=EVAL_INTERVAL,
            eval_callbacks=_phuma_eval_callbacks,
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
                    input_dim=0,  # Auto-calculated
                    hidden_dim=60,
                    output_dim=128,
                    num_timesteps=20,
                    activation="SiLU",
                ),
            ),
        ),
    ),
    observation=observation.g1_29dof_wbt_observation_future_motion,
)

__all__ = ["g1_29dof_phuma", "g1_29dof_phuma_future_motion"]
