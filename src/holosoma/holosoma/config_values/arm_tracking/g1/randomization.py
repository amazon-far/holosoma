"""Arm tracking randomization presets for the G1 robot.

Follows the same structure as ``config_values/loco/g1/randomization.py``.
"""

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg


# Default: no randomization (existing behavior)
g1_arm_tracking_randomization = RandomizationManagerCfg()


# Dynamics randomization for ablation studies.
# Reuses locomotion randomization terms with tuned ranges for arm tracking.
g1_arm_tracking_dynamics_randomization = RandomizationManagerCfg(
    setup_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
            params={
                "push_interval_s": [5, 10],
                "max_push_vel": [0.5, 0.5],  # Lower than locomotion (arm tracking is more sensitive)
                "enabled": True,
            },
        ),
        "actuator_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
            params={
                "kp_range": [0.9, 1.1],
                "kd_range": [0.9, 1.1],
                "rfi_lim_range": [0.5, 1.5],
                "enable_pd_gain": True,
                "enable_rfi_lim": False,
            },
        ),
        "mass_randomizer": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_mass_startup",
            params={
                "enable_link_mass": True,
                "link_mass_range": [0.9, 1.1],
                "enable_base_mass": True,
                "added_mass_range": [-0.5, 1.5],
            },
        ),
        "randomize_friction_startup": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_friction_startup",
            params={
                "friction_range": [0.5, 1.25],
                "enabled": True,
            },
        ),
    },
    reset_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
        ),
        "actuator_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
        ),
        "randomize_push_schedule": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_push_schedule",
        ),
        "randomize_dof_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_dof_state",
            params={
                "joint_pos_scale_range": [0.9, 1.1],
                "joint_pos_bias_range": [0.0, 0.0],
                "joint_vel_range": [0.0, 0.0],
                "randomize_dof_pos_bias": False,
            },
        ),
    },
    step_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
        ),
        "apply_pushes": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:apply_pushes",
        ),
    },
)

