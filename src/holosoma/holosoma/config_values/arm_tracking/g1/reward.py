"""Reward presets for the G1 arm-tracking environment."""

from dataclasses import replace

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg


_TERMS = "holosoma.managers.reward.terms.arm_tracking"


g1_arm_tracking_reward = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_ee_pos": RewardTermCfg(
            func=f"{_TERMS}:tracking_ee_pos",
            weight=4.0,
            params={"tracking_sigma": 0.01},
        ),
        "tracking_ee_rot": RewardTermCfg(
            func=f"{_TERMS}:tracking_ee_rot",
            weight=2.5,
            params={"tracking_sigma": 0.20},
        ),
        "alive": RewardTermCfg(
            func=f"{_TERMS}:alive",
            weight=0.5,
            params={},
        ),
        "penalty_termination": RewardTermCfg(
            func=f"{_TERMS}:penalty_termination",
            weight=-200.0,
            params={},
        ),
        "tracking_orientation": RewardTermCfg(
            func=f"{_TERMS}:tracking_orientation",
            weight=1.0,
            params={"tracking_sigma": 0.10},
        ),
        "tracking_torso_orientation": RewardTermCfg(
            func=f"{_TERMS}:tracking_torso_orientation",
            weight=1.0,
            params={"tracking_sigma": 0.10},
        ),
        "penalty_base_ang_vel": RewardTermCfg(
            func=f"{_TERMS}:penalty_base_ang_vel",
            weight=-1.0,
            params={},
        ),
        "penalty_base_lin_vel_xy": RewardTermCfg(
            func=f"{_TERMS}:penalty_base_lin_vel_xy",
            weight=-0.5,
            params={},
        ),
        "tracking_base_height": RewardTermCfg(
            func=f"{_TERMS}:tracking_base_height",
            weight=1.0,
            params={"base_height_target": 0.73, "tracking_sigma": 3.0},
        ),
        "penalty_leg_action_rate": RewardTermCfg(
            func=f"{_TERMS}:penalty_leg_action_rate",
            weight=-0.5,
            params={},
        ),
        "penalty_arm_action_rate": RewardTermCfg(
            func=f"{_TERMS}:penalty_arm_action_rate",
            weight=-0.5,
            params={},
        ),
        "penalty_arm_action_accel": RewardTermCfg(
            func=f"{_TERMS}:penalty_arm_action_accel",
            weight=-0.04,
            params={},
        ),
        "penalty_leg_action_accel": RewardTermCfg(
            func=f"{_TERMS}:penalty_leg_action_accel",
            weight=-0.04,
            params={},
        ),
        "penalty_arm_residual_l2": RewardTermCfg(
            func=f"{_TERMS}:penalty_arm_residual_l2",
            weight=-0.025,
            params={},
        ),
        "penalty_dof_pos_limits": RewardTermCfg(
            func=f"{_TERMS}:penalty_dof_pos_limits",
            weight=-1.0,
            params={
                "soft_dof_pos_limit": 0.95,
                "use_arm_dofs_only": False,
            },
        ),
        "penalty_arm_dof_vel": RewardTermCfg(
            func=f"{_TERMS}:penalty_arm_dof_vel",
            weight=-0.0125,
            params={},
        ),
        "penalty_leg_dof_vel": RewardTermCfg(
            func=f"{_TERMS}:penalty_leg_dof_vel",
            weight=-0.1,
            params={},
        ),
        "penalty_pose_deviation": RewardTermCfg(
            func=f"{_TERMS}:penalty_pose_deviation",
            weight=-0.5,
            params={
                "include_arm_dofs": False,
                # 0-5: left leg, 6-11: right leg, 12-14: torso,
                # 15-21: left arm, 22-28: right arm
                "pose_weights": [
                    0.01, 1.0, 5.0, 0.01, 5.0, 5.0,   # left leg
                    0.01, 1.0, 5.0, 0.01, 5.0, 5.0,   # right leg
                    50.0, 50.0, 50.0,                   # torso
                    50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,  # left arm
                    50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,  # right arm
                ],
            },
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func=f"{_TERMS}:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
        ),
        "penalty_feet_ori": RewardTermCfg(
            func=f"{_TERMS}:penalty_feet_ori",
            weight=-5.0,
            params={},
        ),
    },
)


g1_arm_tracking_reward_controller = replace(
    g1_arm_tracking_reward,
    terms={
        **g1_arm_tracking_reward.terms,
        "penalty_arm_residual_l2": replace(
            g1_arm_tracking_reward.terms["penalty_arm_residual_l2"],
            weight=-0.025,
        ),
        "penalty_pose_deviation": replace(
            g1_arm_tracking_reward.terms["penalty_pose_deviation"],
            weight=-0.5,
        ),
    },
)

