"""Termination configs for the loco-arm-tracking environment (G1 robot)."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg


g1_loco_arm_tracking_termination = TerminationManagerCfg(
    terms={
        "base_height": TerminationTermCfg(
            func="holosoma.managers.termination.terms.arm_tracking:base_height_termination",
            params={"min_height": 0.3},
        ),
        "base_orientation": TerminationTermCfg(
            func="holosoma.managers.termination.terms.arm_tracking:base_orientation_termination",
            params={
                "threshold_x": 0.85,
                "threshold_y": 0.85,
            },
        ),
        "dof_pos_limits": TerminationTermCfg(
            func="holosoma.managers.termination.terms.arm_tracking:dof_position_limit_termination",
            params={
                "soft_dof_pos_limit": 1.0,
                "use_arm_dofs_only": False,
            },
        ),
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
    }
)


g1_loco_arm_tracking_height_termination = TerminationManagerCfg(
    terms={
        "base_height": TerminationTermCfg(
            func="holosoma.managers.termination.terms.arm_tracking:base_height_termination",
            params={"min_height": 0.35},
        ),
        "base_orientation": TerminationTermCfg(
            func="holosoma.managers.termination.terms.arm_tracking:base_orientation_termination",
            params={
                "threshold_x": 0.85,
                "threshold_y": 0.85,
            },
        ),
        "dof_pos_limits": TerminationTermCfg(
            func="holosoma.managers.termination.terms.arm_tracking:dof_position_limit_termination",
            params={
                "soft_dof_pos_limit": 1.0,
                "use_arm_dofs_only": False,
            },
        ),
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
    }
)

