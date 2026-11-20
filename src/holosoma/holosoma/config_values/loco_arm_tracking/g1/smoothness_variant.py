"""Smoothness variant configs for loco_arm_tracking (G1).

Consolidates acc and lpf action/observation/reward/experiment configs
into a single module. Each smoothness variant is orthogonal to the base
controller (GAC, baseline) and to other features (height tracking, etc.).

Smoothness variants control how action smoothness is enforced:
- ACC_MEAN: Mean-action acceleration penalty (use_true_action_mean=True)
- LPF: Low-pass filter (EMA) on residual actions
"""

from dataclasses import replace
from enum import Enum

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
from holosoma.config_types.observation import ObsTermCfg, ObservationManagerCfg
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_values.loco_arm_tracking.g1.action import GAC_CONTROLLER_CLASS
from holosoma.config_values.loco_arm_tracking.g1.observation import (
    g1_loco_arm_tracking_observation_gac,
    with_height_tracking,
    with_ee_twist_full_critic,
)
from holosoma.config_values.loco_arm_tracking.g1.reward import (
    g1_loco_arm_tracking_reward_fast_sac,
)


class SmoothnessVariant(Enum):
    ACC_MEAN = "acc_mean"
    LPF = "lpf"


_LOCO_ARM_MODULE = "holosoma.managers.action.terms.loco_arm_tracking"


_LPF_LOCO_MODULE = "holosoma.managers.action.terms.lpf_loco_arm_tracking"


_ACTION_CONFIGS: dict[SmoothnessVariant, ActionManagerCfg] = {
    SmoothnessVariant.ACC_MEAN: ActionManagerCfg(
        terms={
            "loco_arm_residual": ActionTermCfg(
                func=f"{_LOCO_ARM_MODULE}:LocoArmMeanActionResidualAction",
                params={
                    "arm_residual_scale": 3.0,
                    "arm_clip_actions": 1.0,
                    "controller_class": GAC_CONTROLLER_CLASS,
                    "use_true_action_mean": True,
                    "controller_kwargs": {},
                },
            )
        }
    ),
    SmoothnessVariant.LPF: ActionManagerCfg(
        terms={
            "loco_arm_residual": ActionTermCfg(
                func=f"{_LPF_LOCO_MODULE}:LPFLocoArmResidualAction",
                params={
                    "arm_residual_scale": 3.0,
                    "arm_clip_actions": 1.0,
                    "controller_class": GAC_CONTROLLER_CLASS,
                    "lpf_cutoff_freqs": {"arms": 5.0, "torso": 3.0, "legs": 5.0},
                },
            )
        }
    ),
}


def get_smoothness_action(variant: SmoothnessVariant) -> ActionManagerCfg:
    """Return the action config for the given smoothness variant."""
    return _ACTION_CONFIGS[variant]


_LOCO_OBS_MODULE = "holosoma.managers.observation.terms.loco_arm_tracking"


_LPF_OBS_MODULE = "holosoma.managers.observation.terms.lpf_arm_tracking"


_prev_prev_action_term = ObsTermCfg(
    func=f"{_LOCO_OBS_MODULE}:prev_prev_action",
    params={},
    scale=1.0,
    noise=0.0,
)


_last_filtered_action_term = ObsTermCfg(
    func=f"{_LPF_OBS_MODULE}:last_filtered_action",
    params={},
    scale=1.0,
    noise=0.0,
)


_OBS_EXTRA_TERMS: dict[SmoothnessVariant, dict[str, ObsTermCfg]] = {
    SmoothnessVariant.ACC_MEAN: {"prev_prev_action": _prev_prev_action_term},
    SmoothnessVariant.LPF: {"last_filtered_action": _last_filtered_action_term},
}


def get_smoothness_observation(
    variant: SmoothnessVariant,
    base: ObservationManagerCfg | None = None,
) -> ObservationManagerCfg:
    """Extend observation with variant-specific terms."""
    if base is None:
        base = g1_loco_arm_tracking_observation_gac
    extra = _OBS_EXTRA_TERMS[variant]
    new_groups = {}
    for name, group in base.groups.items():
        new_groups[name] = replace(
            group,
            terms={**group.terms, **extra},
        )
    return replace(base, groups=new_groups)


_ARM_TRACKING_TERMS = "holosoma.managers.reward.terms.arm_tracking"


_REWARD_EXTRA_TERMS: dict[SmoothnessVariant, dict[str, RewardTermCfg]] = {
    SmoothnessVariant.ACC_MEAN: {
        "penalty_mean_action_accel_arm": RewardTermCfg(
            func=f"{_ARM_TRACKING_TERMS}:penalty_mean_action_accel_arm",
            weight=-0.10,
            params={},
        ),
        "penalty_mean_action_accel_leg": RewardTermCfg(
            func=f"{_ARM_TRACKING_TERMS}:penalty_mean_action_accel_leg",
            weight=-0.10,
            params={},
        ),
    },
    SmoothnessVariant.LPF: {},  # LPF uses base reward unchanged
}


def get_smoothness_reward(
    variant: SmoothnessVariant,
    base: RewardManagerCfg | None = None,
) -> RewardManagerCfg:
    """Build reward config with variant-specific smoothness penalties."""
    if base is None:
        base = g1_loco_arm_tracking_reward_fast_sac
    extra = _REWARD_EXTRA_TERMS[variant]
    return replace(base, terms={**base.terms, **extra})


from holosoma.config_values import algo, robot, simulator, terrain


from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig


from holosoma.config_values.loco_arm_tracking.g1.command import (
    g1_loco_arm_tracking_command,
)


from holosoma.config_values.loco_arm_tracking.g1.curriculum_manager import (
    g1_loco_arm_tracking_curriculum,
)


from holosoma.config_values.loco_arm_tracking.g1.randomization import (
    g1_loco_arm_tracking_randomization,
)


from holosoma.config_values.loco_arm_tracking.g1.termination import (
    g1_loco_arm_tracking_termination,
    g1_loco_arm_tracking_height_termination,
)


_ENV_CLASS = (
    "holosoma.envs.loco_arm_tracking"
    ".loco_arm_tracking_manager.LocoArmTrackingManager"
)


def build_smoothness_experiment(
    variant: SmoothnessVariant,
    *,
    height_tracking: bool = False,
    name_override: str | None = None,
) -> ExperimentConfig:
    """Build a complete loco_arm_tracking experiment config for the given smoothness variant.

    Args:
        variant: Which smoothness variant.
        height_tracking: If True, adds height obs terms and uses lowered termination.
        name_override: Override the auto-generated training name.
    """
    variant_str = variant.value
    height_str = "_height" if height_tracking else ""
    auto_name = f"g1_loco_arm_tracking_gac_{variant_str}{height_str}_fast_sac"
    training_name = name_override or auto_name

    obs = get_smoothness_observation(variant)
    if height_tracking:
        obs = with_height_tracking(obs)

    term_cfg = (
        g1_loco_arm_tracking_height_termination
        if height_tracking
        else g1_loco_arm_tracking_termination
    )

    return ExperimentConfig(
        env_class=_ENV_CLASS,
        training=TrainingConfig(
            project="g1-arm-tracking",
            name=training_name,
        ),
        algo=replace(
            algo.fast_sac,
            config=replace(
                algo.fast_sac.config,
                num_learning_iterations=50000,
                use_symmetry=False,
            ),
        ),
        simulator=simulator.isaacsim,
        robot=robot.g1_29dof,
        terrain=terrain.terrain_locomotion_plane,
        observation=obs,
        action=get_smoothness_action(variant),
        termination=term_cfg,
        randomization=g1_loco_arm_tracking_randomization,
        command=g1_loco_arm_tracking_command,
        curriculum=g1_loco_arm_tracking_curriculum,
        reward=get_smoothness_reward(variant),
        height_tracking=height_tracking,
    )


from holosoma.config_values.loco_arm_tracking.g1.command import (
    g1_loco_arm_tracking_command_ground_heading,
)


_ACC_MEAN_ANGJ_ACTION = ActionManagerCfg(
    terms={
        "loco_arm_residual": ActionTermCfg(
            func=f"{_LOCO_ARM_MODULE}:LocoArmMeanActionResidualAction",
            params={
                "arm_residual_scale": 3.0,
                "arm_clip_actions": 1.0,
                "controller_class": GAC_CONTROLLER_CLASS,
                "use_true_action_mean": True,
                "controller_kwargs": {
                    "use_angular_base_jacobian": True,
                },
            },
        )
    }
)


g1_loco_arm_tracking_gac_acc_height_ground_heading_angj_fast_sac = replace(
    build_smoothness_experiment(
        SmoothnessVariant.ACC_MEAN,
        height_tracking=True,
    ),
    training=TrainingConfig(
        project="g1-arm-tracking",
        name="g1_loco_arm_tracking_gac_acc_height_ground_heading_angj_fast_sac",
    ),
    action=_ACC_MEAN_ANGJ_ACTION,
    command=g1_loco_arm_tracking_command_ground_heading,
    observation=with_ee_twist_full_critic(
        with_height_tracking(get_smoothness_observation(SmoothnessVariant.ACC_MEAN))
    ),
)

