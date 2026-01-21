"""Whole Body Tracking reward presets for the T1-23dof robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

# ===================================================================
# PPO Reward Configuration for T1-23DOF Whole Body Tracking
# ===================================================================

t1_23dof_wbt_reward = RewardManagerCfg(
    terms={
        # Motion tracking rewards - global reference frame
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        
        # Motion tracking rewards - body velocities
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        
        # Regularization and penalty terms
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,  # Penalize large action changes for smooth control
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},  # 90% of joint limit for soft penalty
            weight=-100.0,  # Strong penalty to prevent joint limit violations
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                # Exclude foot links from undesired contacts (allow foot contact)
                "undesired_contacts_body_names": (
                    "^(?!left_foot_link$)(?!right_foot_link$).+$"
                ),
            },
            weight=-0.5,
        ),
    }
)

# ===================================================================
# FastSAC Reward Configuration for T1-23DOF Whole Body Tracking
# ===================================================================

t1_23dof_wbt_fast_sac_reward = RewardManagerCfg(
    terms={
        **t1_23dof_wbt_reward.terms,  # Inherit all terms from PPO configuration
        
        # Modified regularization term for FastSAC
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,  # Stronger action rate penalty for FastSAC
        ),
        
        # Adjusted motion tracking weights for FastSAC
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,  # Increased weight for position tracking
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,  # Maintained orientation tracking weight
        ),
        
        # Motion tracking rewards - relative body frame with adjusted weights
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=2.0,  # Double weight for relative body position tracking
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,  # Maintained relative orientation weight
        ),
    }
)

__all__ = ["t1_23dof_wbt_reward", "t1_23dof_wbt_fast_sac_reward"]
