import tyro
from typing_extensions import Annotated
from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    curriculum,
    robot,
    simulator,
    terrain,
)

from holosoma.config_values.wbt.g1.observation import g1_29dof_wbt_observation
from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization

from holosoma.config_values.wbt.t1.command import t1_23dof_wbt_command
from holosoma.config_values.wbt.t1.termination import t1_23dof_wbt_termination
from holosoma.config_values.wbt.t1.reward import t1_23dof_wbt_reward, t1_23dof_wbt_fast_sac_reward

from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt,
    g1_29dof_wbt_fast_sac,
    g1_29dof_wbt_fast_sac_w_object,
    g1_29dof_wbt_w_object,
)

from holosoma.config_values.loco.t1.experiment import t1_29dof, t1_29dof_fast_sac

# Valid body names for T1 23-DOF robot configuration
VALID_T1_23DOF_BODY_NAMES = [
    'Trunk', 
    'H1', 'H2',
    'AL1', 'AL2', 'AL3', 'left_hand_link',
    'AR1', 'AR2', 'AR3', 'right_hand_link',
    'Waist', 
    'Hip_Pitch_Left', 'Hip_Roll_Left', 'Hip_Yaw_Left', 
    'Shank_Left', 'Ankle_Cross_Left', 'left_foot_link',
    'Hip_Pitch_Right', 'Hip_Roll_Right', 'Hip_Yaw_Right', 
    'Shank_Right', 'Ankle_Cross_Right', 'right_foot_link'
]

# Joint names for T1 23-DOF robot control
T1_23DOF_JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch", 
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw", 
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw", 
    "Waist", 
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll", 
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"
]

# ====================================================================================
# Experiment Configurations
# ====================================================================================

# 1. PPO-based Whole Body Tracking for T1 23-DOF robot
t1_23dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="t1_23dof_wbt_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=40000,
            save_interval=4000,
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
    # Robot configuration for T1 23-DOF
    robot=replace(
        robot.t1_23dof,
        control=replace(robot.t1_23dof.control, action_scale=1.0),
        asset=replace(robot.t1_23dof.asset, enable_self_collisions=True),
        init_state=replace(robot.t1_23dof.init_state, pos=[0.0, 0.0, 0.68]),
        body_names=VALID_T1_23DOF_BODY_NAMES 
    ),
    terrain=terrain.terrain_locomotion_plane,

    # Using G1 observation configuration adapted for T1
    observation=g1_29dof_wbt_observation, 
    
    action=replace(
        action.g1_29dof_joint_pos,
        params={"asset_name": "robot", "joint_names": T1_23DOF_JOINT_NAMES}
    ),
    
    termination=t1_23dof_wbt_termination,
    command=t1_23dof_wbt_command,
    reward=t1_23dof_wbt_reward, 
    
    # Using G1 randomization configuration
    randomization=g1_29dof_wbt_randomization,
    
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.16, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.45, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.30, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.30, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.02, "inf"],
        },
    ),
)

# 2. FastSAC-based Whole Body Tracking for T1 23-DOF robot
t1_23dof_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="t1_23dof_wbt_fast_sac_manager",
        num_envs=1024,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=400000,
            v_max=20.0,
            v_min=-20.0,
            gamma=0.99,  # High gamma is better for motion tracking with long episodes
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
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
    # Robot configuration for T1 23-DOF
    robot=replace(
        robot.t1_23dof,
        control=replace(robot.t1_23dof.control, action_scale=1.0),
        asset=replace(robot.t1_23dof.asset, enable_self_collisions=True),
        init_state=replace(robot.t1_23dof.init_state, pos=[0.0, 0.0, 0.68]),
        body_names=VALID_T1_23DOF_BODY_NAMES
    ),
    terrain=terrain.terrain_locomotion_plane,

    # Using G1 observation configuration adapted for T1
    observation=g1_29dof_wbt_observation,
    
    action=replace(
        action.g1_29dof_joint_pos,
        params={"asset_name": "robot", "joint_names": T1_23DOF_JOINT_NAMES}
    ),

    termination=t1_23dof_wbt_termination,
    command=t1_23dof_wbt_command,
    reward=t1_23dof_wbt_fast_sac_reward, 
   
    # Using G1 randomization configuration
    randomization=g1_29dof_wbt_randomization,
    
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

# Default configurations available for CLI selection
DEFAULTS = {
    # G1 robot configurations
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_w_object,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac,
    "g1_29dof_wbt_fast_sac_w_object": g1_29dof_wbt_fast_sac_w_object,
    
    # T1 robot configurations
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
    "t1_23dof_wbt": t1_23dof_wbt,
    "t1_23dof_wbt_fast_sac": t1_23dof_wbt_fast_sac,
}

# Tyro configuration for CLI argument parsing
AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{k.replace('_', '-')}": v for k, v in DEFAULTS.items()}
        )
    ),
]

__all__ = ["t1_23dof_wbt", "t1_23dof_wbt_fast_sac"]

"""
Example usage:
1. Train T1 23-DOF with PPO:
python src/holosoma/holosoma/train_agent.py \
    exp:t1-23dof-wbt

2. Train T1 23-DOF with FastSAC:
python src/holosoma/holosoma/train_agent.py \
    exp:t1-23dof-wbt-fast-sac

3. Train G1 29-DOF with object (as reference):
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-w-object

Note: For detailed parameter overrides, refer to individual configuration blocks.
"""
