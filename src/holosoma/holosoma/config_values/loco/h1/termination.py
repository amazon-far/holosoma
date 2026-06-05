"""Locomotion termination presets for the H1 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

h1_19dof_termination = TerminationManagerCfg(
    terms={
        "contact": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:contact_forces_exceeded",
            params={
                "force_threshold": 1.0,
                "contact_indices_attr": "termination_contact_indices",
            },
        ),
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
    }
)

__all__ = ["h1_19dof_termination"]

