"""Observation configs for the loco-arm-tracking environment (G1 robot)."""

from dataclasses import replace

from holosoma.config_types.observation import (
    ObservationManagerCfg,
    ObsGroupCfg,
    ObsTermCfg,
)


_TERMS_MODULE = "holosoma.managers.observation.terms.loco_arm_tracking"


_baseline_actor_terms = {
    "arm_command": ObsTermCfg(
        func=f"{_TERMS_MODULE}:arm_command",
        params={},
        scale=1.0,
        noise=0.0,
    ),
    "ee_error_pose": ObsTermCfg(
        func=f"{_TERMS_MODULE}:ee_error_pose",
        params={},
        scale=1.0,
        noise=0.0,
    ),
    "ee_error_pose_tanh": ObsTermCfg(
        func=f"{_TERMS_MODULE}:ee_error_pose_tanh",
        params={},
        scale=1.0,
        noise=0.0,
    ),
    "command_lin_vel": ObsTermCfg(
        func=f"{_TERMS_MODULE}:command_lin_vel",
        params={},
        scale=1.0,
        noise=0.0,
    ),
    "command_ang_vel": ObsTermCfg(
        func=f"{_TERMS_MODULE}:command_ang_vel",
        params={},
        scale=1.0,
        noise=0.0,
    ),
    "projected_gravity": ObsTermCfg(
        func=f"{_TERMS_MODULE}:projected_gravity",
        params={},
        scale=1.0,
        noise=0.001,
    ),
    "base_ang_vel": ObsTermCfg(
        func=f"{_TERMS_MODULE}:base_ang_vel",
        params={},
        scale=0.25,
        noise=0.01,
    ),
    "dof_pos_relative": ObsTermCfg(
        func=f"{_TERMS_MODULE}:dof_pos_relative",
        params={},
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func=f"{_TERMS_MODULE}:dof_vel",
        params={},
        scale=0.05,
        noise=0.02,
    ),
    "last_action": ObsTermCfg(
        func=f"{_TERMS_MODULE}:last_action",
        params={},
        scale=1.0,
        noise=0.0,
    ),
    "sin_phase": ObsTermCfg(
        func=f"{_TERMS_MODULE}:sin_phase",
        params={},
        scale=1.0,
        noise=0.0,
    ),
    "cos_phase": ObsTermCfg(
        func=f"{_TERMS_MODULE}:cos_phase",
        params={},
        scale=1.0,
        noise=0.0,
    ),
}


# Critic adds base_lin_vel
_base_lin_vel_term = ObsTermCfg(
    func=f"{_TERMS_MODULE}:base_lin_vel",
    params={},
    scale=1.0,
    noise=0.0,
)


_controller_arm_joint_command_term = ObsTermCfg(
    func=f"{_TERMS_MODULE}:controller_arm_joint_command",
    params={},
    scale=1.0,
    noise=0.0,
)


_ee_body_twist_term = ObsTermCfg(
    func=f"{_TERMS_MODULE}:ee_body_twist",
    params={},
    scale=0.05,
    noise=0.02,
)


_last_residual_policy_action_term = ObsTermCfg(
    func=f"{_TERMS_MODULE}:last_residual_policy_action",
    params={},
    scale=1.0,
    noise=0.0,
)


# GAC actor: baseline terms + controller_arm_joint_command + ee_body_twist,
# with last_action replaced by last_residual_policy_action
_gac_actor_terms = {
    **_baseline_actor_terms,
    "last_action": _last_residual_policy_action_term,
    "controller_arm_joint_command": _controller_arm_joint_command_term,
    "ee_body_twist": _ee_body_twist_term,
}


g1_loco_arm_tracking_observation_gac = ObservationManagerCfg(
    clip_observations=100.0,
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms=_gac_actor_terms,
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                **_gac_actor_terms,
                "base_lin_vel": _base_lin_vel_term,
            },
        ),
    },
)


_torso_projected_gravity_term = ObsTermCfg(
    func=f"{_TERMS_MODULE}:torso_projected_gravity",
    params={},
    scale=1.0,
    noise=0.0,
)


_base_height_command_term = ObsTermCfg(
    func=f"{_TERMS_MODULE}:base_height_command",
    params={},
    scale=1.0,
    noise=0.0,
)


_base_height_error_term = ObsTermCfg(
    func=f"{_TERMS_MODULE}:base_height_error",
    params={},
    scale=1.0,
    noise=0.0,
)


def with_height_tracking(
    base_cfg: ObservationManagerCfg,
) -> ObservationManagerCfg:
    """Add height tracking obs: command to actor+critic, error to critic only.

    Actor gets ``base_height_command`` (the target — deployable).
    Critic gets ``base_height_command`` + ``base_height_error`` (privileged
    ground-truth height for value estimation).
    """
    new_groups = {}
    for name, group in base_cfg.groups.items():
        if name == "critic_obs":
            new_groups[name] = replace(
                group,
                terms={
                    **group.terms,
                    "base_height_command": _base_height_command_term,
                    "base_height_error": _base_height_error_term,
                },
            )
        else:
            new_groups[name] = replace(
                group,
                terms={
                    **group.terms,
                    "base_height_command": _base_height_command_term,
                },
            )
    return replace(base_cfg, groups=new_groups)


_ee_body_twist_full_term = ObsTermCfg(
    func=f"{_TERMS_MODULE}:ee_body_twist_full",
    params={},
    scale=0.05,
    noise=0.0,
)


def with_ee_twist_full_critic(
    base_cfg: ObservationManagerCfg,
) -> ObservationManagerCfg:
    """Add ``ee_body_twist_full`` to critic, replacing the zeroed variant.

    For ``ground_heading`` reference frame, the actor's ``ee_body_twist``
    excludes the base z-velocity contribution (not estimable at deploy).
    This adds the corrected full twist to the critic for better value
    estimation, without changing the actor observation space.
    """
    critic = base_cfg.groups["critic_obs"]
    new_groups = dict(base_cfg.groups)
    new_groups["critic_obs"] = replace(
        critic,
        terms={
            **critic.terms,
            "ee_body_twist_full": _ee_body_twist_full_term,
        },
    )
    return replace(base_cfg, groups=new_groups)


_pelvis_gravity_command_term = ObsTermCfg(
    func=f"{_TERMS_MODULE}:pelvis_gravity_command", params={}, scale=1.0, noise=0.0,
)


_torso_gravity_command_term = ObsTermCfg(
    func=f"{_TERMS_MODULE}:torso_gravity_command", params={}, scale=1.0, noise=0.0,
)


_pelvis_gravity_error_term = ObsTermCfg(
    func=f"{_TERMS_MODULE}:pelvis_gravity_error", params={}, scale=1.0, noise=0.0,
)


_torso_gravity_error_term = ObsTermCfg(
    func=f"{_TERMS_MODULE}:torso_gravity_error", params={}, scale=1.0, noise=0.0,
)


def with_posture_command(
    base_cfg: ObservationManagerCfg,
) -> ObservationManagerCfg:
    """Add trunk-orientation command observations to actor and critic.

    Actor gains the two commanded gravity directions (3 dims each). Critic
    gains the same two commands plus their tracking errors and the measured
    torso gravity, so the value function can see how far posture is from the
    command.
    """
    new_groups = {}
    for name, group in base_cfg.groups.items():
        if name == "critic_obs":
            new_groups[name] = replace(
                group,
                terms={
                    **group.terms,
                    "pelvis_gravity_command": _pelvis_gravity_command_term,
                    "torso_gravity_command": _torso_gravity_command_term,
                    "pelvis_gravity_error": _pelvis_gravity_error_term,
                    "torso_gravity_error": _torso_gravity_error_term,
                    "torso_projected_gravity": _torso_projected_gravity_term,
                },
            )
        elif name.startswith("actor_obs"):
            new_groups[name] = replace(
                group,
                terms={
                    **group.terms,
                    "pelvis_gravity_command": _pelvis_gravity_command_term,
                    "torso_gravity_command": _torso_gravity_command_term,
                },
            )
        else:
            new_groups[name] = group
    return replace(base_cfg, groups=new_groups)

