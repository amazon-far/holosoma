"""Locomotion reward presets for the G1 robot."""

from __future__ import annotations

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

# Shared reward terms for G1 locomotion (used by both PPO and FastSAC).
_g1_29dof_loco_common_terms: dict[str, RewardTermCfg] = {
    "tracking_lin_vel": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
        weight=2.0,
        params={"tracking_sigma": 0.25},
    ),
    "tracking_ang_vel": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
        weight=1.5,
        params={"tracking_sigma": 0.25},
    ),
    "penalty_ang_vel_xy": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
        weight=-1.0,
        params={},
        tags=["penalty_curriculum"],
    ),
    "penalty_orientation": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
        weight=-10.0,
        params={},
        tags=["penalty_curriculum"],
    ),
    "penalty_action_rate": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
        weight=-2.0,
        params={},
        tags=["penalty_curriculum"],
    ),
    "feet_phase": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:feet_phase",
        weight=5.0,
        params={"swing_height": 0.09, "tracking_sigma": 0.008},
    ),
    "pose": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:pose",
        weight=-0.5,
        params={
            "pose_weights": [
                0.01,
                1.0,
                5.0,  # left hip: pitch, roll, yaw
                0.01,
                5.0,
                5.0,  # left: knee, ankle_pitch, ankle_roll
                0.01,
                1.0,
                5.0,  # right hip: pitch, roll, yaw
                0.01,
                5.0,
                5.0,  # right: knee, ankle_pitch, ankle_roll
                50.0,
                50.0,
                50.0,  # waist: yaw, roll, pitch
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,  # left arm
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,  # right arm
            ],
        },
        tags=["penalty_curriculum"],
    ),
    "penalty_close_feet_xy": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
        weight=-10.0,
        params={"close_feet_threshold": 0.15},
        tags=["penalty_curriculum"],
    ),
    "penalty_feet_ori": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
        weight=-5.0,
        params={},
        tags=["penalty_curriculum"],
    ),
}

g1_29dof_loco = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        **_g1_29dof_loco_common_terms,
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=1.0,
            params={},
        ),
    },
)

g1_29dof_loco_fast_sac = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        **_g1_29dof_loco_common_terms,
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=10.0,
            params={},
        ),
    },
)

__all__ = ["g1_29dof_loco", "g1_29dof_loco_fast_sac"]
