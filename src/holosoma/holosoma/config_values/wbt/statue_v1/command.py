"""Whole Body Tracking command presets for the Statue v1 RMD robot."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.1, 0.1, 0.05],
    root_ang_vel=[0.1, 0.1, 0.1],
    object_pos=[0.05, 0.05, 0.0],
)

motion_config = MotionConfig(
    motion_file="holosoma/data/motions/statue_v1_rmd/whole_body_tracking/unreal_engine/walk_statue_v1_rmd_50fps.npz",
    body_names_to_track=[
        "pelvis_link",
        "left_hip_roll_link",
        "left_knee_pitch_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_pitch_link",
        "right_ankle_roll_link",
        "waist_pitch_link",
        "left_shoulder_roll_link",
        "left_elbow_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_pitch_link",
        "right_wrist_yaw_link",
    ],
    body_name_ref=["waist_pitch_link"],
    use_adaptive_timesteps_sampler=False,
    noise_to_initial_pose=init_pose_config,
)

statue_v1_rmd_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config,
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
)

__all__ = [
    "statue_v1_rmd_wbt_command",
]
