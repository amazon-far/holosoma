"""Locomotion termination presets for the T1 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

t1_29dof_termination = TerminationManagerCfg(
    terms={
        "contact": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:contact_forces_exceeded",
            params={
                "force_threshold": 1.0,
                "bodies": ["#root", "Waist", "#hip_links"],
            },
        ),
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
    }
)

__all__ = ["t1_29dof_termination"]
