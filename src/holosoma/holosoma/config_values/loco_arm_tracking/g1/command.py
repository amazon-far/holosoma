"""Command configs for the loco-arm-tracking environment (G1 robot).

Runs three command terms in parallel: locomotion gait, locomotion command,
and arm tracking command.
"""


from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg


def _make_loco_arm_command(arm_reference_frame: str) -> CommandManagerCfg:
    """Build loco-arm-tracking command config with given arm reference frame."""
    arm_term = CommandTermCfg(
        func=(
            "holosoma.managers.command.terms"
            ".arm_tracking:ArmTrackingCommand"
        ),
        params={
            "reference_frame": arm_reference_frame,
            "trajectory_mode": "precomputed",
            "trajectory_file": "data/trajectories/arm_smooth_phaseB.pt",
        },
    )
    return CommandManagerCfg(
        params={
            "locomotion_command_resampling_time": 10.0,
        },
        setup_terms={
            "locomotion_gait": CommandTermCfg(
                func="holosoma.managers.command.terms.locomotion:LocomotionGait",
                params={
                    "gait_period": 1.0,
                    "gait_period_range": [0.6, 1.5],
                },
            ),
            "locomotion_command": CommandTermCfg(
                func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
                params={
                    "command_ranges": {
                        "lin_vel_x": [-1.0, 1.0],
                        "lin_vel_y": [-1.0, 1.0],
                        "ang_vel_yaw": [-1.0, 1.0],
                        "heading": [-3.14, 3.14],
                    },
                    "stand_prob": 0.2,
                    "turn_in_place_prob": 0.0,
                },
            ),
            "gait_avg_velocity": CommandTermCfg(
                func="holosoma.managers.command.terms.locomotion:GaitAveragedVelocity",
                params={
                    "gait_term_name": "locomotion_gait",
                    "warmup_samples": 5,
                    "min_cmd_speed": 0.03,
                },
            ),
            "arm_command": arm_term,
        },
        reset_terms={
            "locomotion_gait": CommandTermCfg(
                func="holosoma.managers.command.terms.locomotion:LocomotionGait",
            ),
            "locomotion_command": CommandTermCfg(
                func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
            ),
            "gait_avg_velocity": CommandTermCfg(
                func="holosoma.managers.command.terms.locomotion:GaitAveragedVelocity",
            ),
            "arm_command": arm_term,
        },
        step_terms={
            "locomotion_gait": CommandTermCfg(
                func="holosoma.managers.command.terms.locomotion:LocomotionGait",
            ),
            "locomotion_command": CommandTermCfg(
                func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
            ),
            "gait_avg_velocity": CommandTermCfg(
                func="holosoma.managers.command.terms.locomotion:GaitAveragedVelocity",
            ),
            "arm_command": arm_term,
        },
    )


g1_loco_arm_tracking_command = _make_loco_arm_command("base")


g1_loco_arm_tracking_command_ground_heading = _make_loco_arm_command("ground_heading")

