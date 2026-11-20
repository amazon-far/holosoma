from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg


g1_arm_tracking_observation = ObservationManagerCfg(
    clip_observations=100.0,
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms={
                "arm_command": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:arm_command",
                    params={},
                    scale=1.0,
                    noise=0.0,
                ),
                "ee_error_pose": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:ee_error_pose",
                    params={},
                    scale=1.0,
                    noise=0.0,
                ),
                "ee_error_pose_tanh": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:ee_error_pose_tanh",
                    params={"temperature": 0.01},
                    scale=1.0,
                    noise=0.0,
                ),
                "base_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:base_pos",
                    params={},
                    scale=1.0,
                    noise=0.0,
                    enabled=False,
                ),
                "base_rot6d": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:base_rot6d",
                    params={},
                    scale=1.0,
                    noise=0.0,
                    enabled=False,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:projected_gravity",
                    params={},
                    scale=1.0,
                    noise=0.001,
                ),
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:base_ang_vel",
                    params={},
                    scale=0.25,
                    noise=0.01,
                ),
                "dof_pos_relative": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:dof_pos_relative",
                    params={},
                    # Match locomotion actor observation scaling/noise.
                    scale=1.0,
                    noise=0.01,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:dof_vel",
                    params={},
                    # Match locomotion actor observation scaling/noise.
                    scale=0.05,
                    noise=0.10,
                ),
                "last_action": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:last_action",
                    params={},
                    scale=1.0,
                    noise=0.0,
                )
            },
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "arm_command": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:arm_command",
                    params={},
                    scale=1.0,
                    noise=0.0,
                ),
                "ee_error_pose": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:ee_error_pose",
                    params={},
                    scale=1.0,
                    noise=0.0,
                ),
                "ee_error_pose_tanh": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:ee_error_pose_tanh",
                    params={"temperature": 0.01},
                    scale=1.0,
                    noise=0.0,
                ),
                "base_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:base_pos",
                    params={},
                    scale=1.0,
                    noise=0.0,
                    enabled=False,
                ),
                "base_rot6d": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:base_rot6d",
                    params={},
                    scale=1.0,
                    noise=0.0,
                    enabled=False,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:projected_gravity",
                    params={},
                    scale=1.0,
                    noise=0.0,
                ),
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:base_ang_vel",
                    params={},
                    scale=0.25,
                    noise=0.0,
                ),
                "dof_pos_relative": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:dof_pos_relative",
                    params={},
                    # Match locomotion critic observation scaling/noise.
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:dof_vel",
                    params={},
                    # Match locomotion critic observation scaling/noise.
                    scale=0.05,
                    noise=0.00,
                ),
                "last_action": ObsTermCfg(
                    func="holosoma.managers.observation.terms.arm_tracking:last_action",
                    params={},
                    scale=1.0,
                    noise=0.0,
                )
            },
        ),
    },
)


_controller_arm_joint_command_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.arm_tracking:controller_arm_joint_command",
    params={},
    scale=1.0,
    noise=0.0,
)


_last_residual_policy_action_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.arm_tracking:last_residual_policy_action",
    params={},
    scale=1.0,
    noise=0.0,
)


g1_arm_tracking_observation_with_controller_cmd = replace(
    g1_arm_tracking_observation,
    groups={
        "actor_obs": replace(
            g1_arm_tracking_observation.groups["actor_obs"],
            terms={
                **g1_arm_tracking_observation.groups["actor_obs"].terms,
                "last_action": _last_residual_policy_action_term,
                "controller_arm_joint_command": _controller_arm_joint_command_term,
            },
        ),
        "critic_obs": replace(
            g1_arm_tracking_observation.groups["critic_obs"],
            terms={
                **g1_arm_tracking_observation.groups["critic_obs"].terms,
                "last_action": _last_residual_policy_action_term,
                "controller_arm_joint_command": _controller_arm_joint_command_term,
            },
        ),
    },
)

