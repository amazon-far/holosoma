"""Locomotion reward presets for the H1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

_h1_pose_weights = [
    0.5,
    8.0,
    6.0,
    1.0,
    8.0,
    0.5,
    8.0,
    6.0,
    1.0,
    8.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
]

h1_19dof_loco = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=0.8,
            params={"tracking_sigma": 0.25},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-20.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-4.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=3.0,
            params={"swing_height": 0.06, "tracking_sigma": 0.02},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.8,
            params={"pose_weights": _h1_pose_weights},
            tags=["penalty_curriculum"],
        ),
        "base_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:base_height",
            weight=-3.0,
            params={"desired_base_height": 0.98, "zero_vel_penalty_scale": 2.0},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-25.0,
            params={"close_feet_threshold": 0.28},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-8.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=1.0,
            params={},
        ),
    },
)

h1_19dof_loco_fast_sac = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        **h1_19dof_loco.terms,
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=5.0,
            params={},
        ),
    },
)

__all__ = ["h1_19dof_loco", "h1_19dof_loco_fast_sac"]
