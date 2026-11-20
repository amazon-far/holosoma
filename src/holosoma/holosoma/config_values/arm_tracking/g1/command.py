from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg


_ARM_TRACKING_COMMAND_FUNC = "holosoma.managers.command.terms.arm_tracking:ArmTrackingCommand"


g1_arm_tracking_command = CommandManagerCfg(
    setup_terms={
        "arm_command": CommandTermCfg(
            func=_ARM_TRACKING_COMMAND_FUNC,
            params={
                "reference_frame": "base",  # "base", "torso", "world", "heading", or "ground_heading"
                "use_fixed_sampling": False,  # If True, skip reset-time sampling and keep fixed EE targets.
            },
        )
    },
    reset_terms={
        "arm_command": CommandTermCfg(
            func=_ARM_TRACKING_COMMAND_FUNC,
            params={
                "reference_frame": "base",  # "base", "torso", "world", "heading", or "ground_heading"
                "use_fixed_sampling": False,  # If True, skip reset-time sampling and keep fixed EE targets.
            },
        )
    },
    step_terms={
        "arm_command": CommandTermCfg(
            func=_ARM_TRACKING_COMMAND_FUNC,
            params={
                "reference_frame": "base",  # "base", "torso", "world", "heading", or "ground_heading"
                "use_fixed_sampling": False,  # If True, skip reset-time sampling and keep fixed EE targets.
            },
        )
    },
)

