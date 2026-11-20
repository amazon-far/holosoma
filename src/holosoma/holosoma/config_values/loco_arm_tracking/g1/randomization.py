"""Randomization config for the loco-arm-tracking environment (G1 robot).

Uses the same dynamics randomization terms as the arm_tracking env
(``g1_arm_tracking_dynamics_randomization``): push, PD gain, mass, friction,
and base CoM randomization. Explicitly excludes ``setup_action_delay_buffers``
and ``setup_dof_pos_bias`` — those terms from the locomotion randomization
produced NaN in this environment during early testing.

Torque RFI (residual force injection) is wired in but **disabled by default**
(``setup_torque_rfi.params.enabled = False``), so the config is behaviorally
identical to the previous version unless explicitly toggled on — when disabled,
the injection block is skipped regardless of the actuator scale buffer. RFI
adds bounded per-step uniform torque noise and is a candidate mitigation for
the arm command-chatter limit cycle, since it injects an in-loop high-frequency
disturbance during training that the smoothness penalties can act against.

The per-joint peak is ``rfi_lim · rfi_lim_scale · torque_limit``. With
``enable_rfi_lim=True`` on the actuator term, ``rfi_lim_scale`` is drawn per
env/joint from ``rfi_lim_range=[0.5, 1.5]``, so the effective peak fraction
spans ``rfi_lim·[0.5, 1.5]`` (e.g. [0.04, 0.12] for rfi_lim=0.08). Enable from
a launch script with::

    --randomization.setup_terms.setup_torque_rfi.params.enabled True
    --randomization.setup_terms.setup_torque_rfi.params.rfi_lim 0.08

Keep rfi_lim ≤ ~0.1; at 1.0 the noise would equal the full torque limit every
step and swamp the PD command. RFI was originally grouped with the NaN-prone
terms above, so watch early-training stability when turning it on.
"""

from holosoma.config_types.randomization import (
    RandomizationManagerCfg,
    RandomizationTermCfg,
)


g1_loco_arm_tracking_randomization = RandomizationManagerCfg(
    setup_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
            params={
                "push_interval_s": [5, 10],
                "max_push_vel": [0.5, 0.5],
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
                "enable_rfi_lim": True,
            },
        ),
        # Torque RFI (residual force injection): bounded per-step uniform
        # torque noise, ±rfi_lim·rfi_lim_scale·torque_limit per joint. Disabled
        # by default; toggle via
        # --randomization.setup_terms.setup_torque_rfi.params.enabled.
        # With enable_rfi_lim=True above, the per-env/per-joint scale is drawn
        # from rfi_lim_range=[0.5, 1.5], so the effective peak fraction spans
        # rfi_lim·[0.5, 1.5] = [0.04, 0.12] for rfi_lim=0.08.
        "setup_torque_rfi": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_torque_rfi",
            params={
                "enabled": False,
                "rfi_lim": 0.08,
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
        "randomize_base_com_startup": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
            params={
                "base_com_range": {
                    "x": [-0.05, 0.05],
                    "y": [-0.05, 0.05],
                    "z": [-0.05, 0.05],
                },
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

