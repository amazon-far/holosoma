"""Reward configs for the loco-arm-tracking environment (G1 robot)."""

from dataclasses import replace

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg


_TERMS_MODULE = "holosoma.managers.reward.terms.loco_arm_tracking"


_ARM_TRACKING_TERMS_MODULE = "holosoma.managers.reward.terms.arm_tracking"


g1_loco_arm_tracking_reward = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func=f"{_TERMS_MODULE}:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func=f"{_TERMS_MODULE}:tracking_ang_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ee_pos": RewardTermCfg(
            func=f"{_TERMS_MODULE}:tracking_ee_pos_phase_gated",
            weight=4.0,
            params={
                "tracking_sigma": 0.01,
                "touchdown_gate_sigma_scale": 1.0,
                "touchdown_gate_phase_width": 0.20,
            },
        ),
        "tracking_ee_rot": RewardTermCfg(
            func=f"{_TERMS_MODULE}:tracking_ee_rot",
            weight=2.5,
            params={"tracking_sigma": 0.20},
        ),
        "feet_phase": RewardTermCfg(
            func=f"{_TERMS_MODULE}:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "tracking_orientation": RewardTermCfg(
            func=f"{_TERMS_MODULE}:tracking_orientation",
            weight=1.0,
            params={"tracking_sigma": 0.10},
        ),
        "tracking_torso_orientation": RewardTermCfg(
            func=f"{_TERMS_MODULE}:tracking_torso_orientation",
            weight=1.0,
            params={"tracking_sigma": 0.10},
        ),
        "tracking_base_height": RewardTermCfg(
            func=f"{_ARM_TRACKING_TERMS_MODULE}:tracking_base_height",
            weight=1.0,
            params={"base_height_target": 0.73, "tracking_sigma": 3.0},
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_arm_action_rate": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_arm_action_rate",
            weight=-0.5,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_leg_action_rate": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_leg_action_rate",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_dof_pos_limits": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_dof_pos_limits",
            weight=-1.0,
            params={"soft_dof_pos_limit": 0.95},
            tags=["penalty_curriculum"],
        ),
        "penalty_pose_deviation": RewardTermCfg(
            func=f"{_ARM_TRACKING_TERMS_MODULE}:penalty_pose_deviation",
            weight=-0.5,
            params={
                "include_arm_dofs": False,
                # 0-5: left leg, 6-11: right leg, 12-14: torso,
                # 15-21: left arm, 22-28: right arm
                "pose_weights": [
                    0.01, 1.0, 5.0, 0.01, 5.0, 5.0,   # left leg
                    0.01, 1.0, 5.0, 0.01, 5.0, 5.0,   # right leg
                    50.0, 50.0, 50.0,                   # torso
                    50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,  # left arm (zeroed by include_arm_dofs=False)
                    50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,  # right arm (zeroed by include_arm_dofs=False)
                ],
            },
            tags=["penalty_curriculum"],
        ),
        "penalty_arm_dof_vel": RewardTermCfg(
            func=f"{_ARM_TRACKING_TERMS_MODULE}:penalty_arm_dof_vel",
            weight=-0.0125,
            params={},
        ),
        "penalty_arm_residual_l2": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_arm_residual_l2",
            weight=-0.05,
            params={},
        ),
        "penalty_ee_body_twist": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_ee_body_twist",
            weight=0.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_lateral_velocity": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_lateral_velocity_quadratic",
            weight=-2.0,
            params={"min_cmd_speed": 0.03},
            tags=["penalty_curriculum"],
        ),
        "penalty_turn_in_place_leakage": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_turn_in_place_leakage",
            weight=-2.5,
            params={"lin_cmd_threshold": 0.03, "yaw_cmd_threshold": 0.1},
            tags=["penalty_curriculum"],
        ),
        "penalty_straight_walk_yaw_leakage": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_straight_walk_yaw_leakage",
            weight=-2.5,
            params={"lin_cmd_threshold": 0.05, "yaw_cmd_threshold": 0.03},
            tags=["penalty_curriculum"],
        ),
        # Gait-averaged leakage penalties (disabled by default; enabled via the
        # --gait-avg-leakage pipeline flag, which also zeroes the instantaneous
        # straight-walk yaw + lateral penalties above).
        "penalty_straight_walk_yaw_leakage_gait_avg": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_straight_walk_yaw_leakage_gait_avg",
            weight=0.0,
            params={
                "lin_cmd_threshold": 0.05,
                "yaw_cmd_threshold": 0.03,
                "instant_weight": 0.15,
            },
            tags=["penalty_curriculum"],
        ),
        "penalty_lateral_velocity_gait_avg": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_lateral_velocity_gait_avg",
            weight=0.0,
            params={"min_cmd_speed": 0.03, "instant_weight": 0.15},
            tags=["penalty_curriculum"],
        ),
        # Heading-hold penalty: penalizes accumulated yaw drift from latch
        # heading (disabled by default; enabled via CLI --reward.terms override).
        "penalty_heading_hold": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_heading_hold",
            weight=0.0,
            params={"lin_cmd_threshold": 0.05, "yaw_cmd_threshold": 0.03},
            tags=["penalty_curriculum"],
        ),
        "penalty_standing_ang_vel": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_standing_ang_vel",
            weight=-2.5,
            params={"lin_cmd_threshold": 0.03, "yaw_cmd_threshold": 0.03},
            tags=["penalty_curriculum"],
        ),
        "penalty_foot_contact_force": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_foot_contact_force",
            weight=-0.5,
            params={"fz_threshold": 400.0},
            tags=["penalty_curriculum"],
        ),
        "penalty_foot_touchdown_shock": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_foot_touchdown_shock",
            weight=0.0,
            params={"contact_threshold": 5.0, "vz_threshold": 0.05},
            tags=["penalty_curriculum"],
        ),
        # Pre-touchdown velocity shaping (design §5). Disabled by default;
        # enable as a low-weight fine-tune on a stable checkpoint via CLI
        # --reward.terms.penalty_foot_pretouchdown_velocity.weight.
        "penalty_foot_pretouchdown_velocity": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_foot_pretouchdown_velocity",
            weight=0.0,
            params={
                "contact_threshold": 5.0,
                "clearance_threshold": 0.05,
                "safe_downward_speed": 0.15,
                "min_cmd_speed": 0.03,
            },
            tags=["penalty_curriculum"],
        ),
        "penalty_termination": RewardTermCfg(
            func=f"{_TERMS_MODULE}:penalty_termination",
            weight=-200.0,
            params={},
        ),
        "alive": RewardTermCfg(
            func=f"{_TERMS_MODULE}:alive",
            weight=2.5,
            params={},
        ),
    },
)


# GAC-enriched fast-SAC: residual penalty at -0.025 (regularizes corrections
# on top of the controller).
g1_loco_arm_tracking_reward_fast_sac = replace(
    g1_loco_arm_tracking_reward,
    terms={
        **g1_loco_arm_tracking_reward.terms,
        "alive": replace(
            g1_loco_arm_tracking_reward.terms["alive"], weight=10.0
        ),
    },
)

