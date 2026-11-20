from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg


GAC_CONTROLLER_CLASS = "holosoma.envs.arm_tracking.gac_controller:GAC_controller"


CTRL_ARM_RESIDUAL_SCALE = 3.0


g1_arm_tracking_action_gac = ActionManagerCfg(
    terms={
        "residual_arm_tracking": ActionTermCfg(
            func="holosoma.managers.action.terms.arm_tracking:ResidualArmTrackingAction",
            params={
                "arm_residual_scale": CTRL_ARM_RESIDUAL_SCALE,
                "arm_clip_actions": 1.0,
                # Switch to H-inf PID by setting:
                # "controller_class": HINF_PID_CONTROLLER_CLASS
                # Disable base controller by setting:
                # "controller_class": NO_CONTROLLER_CLASS
                "controller_class": GAC_CONTROLLER_CLASS,
            },
        )
    }
)

