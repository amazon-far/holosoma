
from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg


g1_arm_tracking_termination = TerminationManagerCfg(
    terms={
        "base_height": TerminationTermCfg(
            func="holosoma.managers.termination.terms.arm_tracking:base_height_termination",
            params={},
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

