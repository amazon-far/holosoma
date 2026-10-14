"""Configuration types for action manager."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import field, fields, replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from holosoma.utils.name_tags import expand_joint_value_map

if TYPE_CHECKING:
    from holosoma.config_types.robot import RobotConfig


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class JointControllerCfg:
    """Software controller applied by a joint action term or simulator plugin."""

    control_type: Literal["P", "V", "T"] = "P"
    """``"P"`` (position PD), ``"V"`` (velocity PD), or ``"T"`` (direct torque)."""

    stiffness: dict[str, float] = field(default_factory=dict)
    """PD stiffness keyed by exact DOF name or ``#joint_tag``."""

    damping: dict[str, float] = field(default_factory=dict)
    """PD damping keyed independently by exact DOF name or ``#joint_tag``."""

    clip_torques: bool = True
    """Clamp computed torques to the simulator's resolved physical effort limits."""

    action_scales_by_effort_limit_over_p_gain: bool = False
    """Multiply the action scale by ``effort_limit / stiffness`` per controlled DOF."""

    def resolve_gains(
        self,
        robot: RobotConfig,
        *,
        required_dof_names: Sequence[str] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Return gains in ``robot.dof_names`` order, requiring coverage of controlled DOFs."""
        required = self.control_type in ("P", "V")
        stiffness = expand_joint_value_map(
            self.stiffness,
            robot.joint_tags,
            robot.dof_names,
            what="joint controller stiffness",
            required=False,
        )
        damping = expand_joint_value_map(
            self.damping,
            robot.joint_tags,
            robot.dof_names,
            what="joint controller damping",
            required=False,
        )
        if required:
            controlled = list(robot.dof_names if required_dof_names is None else required_dof_names)
            unknown = [name for name in controlled if name not in robot.dof_names]
            if unknown:
                raise ValueError(f"joint controller required_dof_names contains unknown DOF(s) {unknown}.")
            missing_stiffness = [name for name in controlled if name not in stiffness]
            if missing_stiffness:
                raise ValueError(f"joint controller stiffness has no key covering DOF(s) {missing_stiffness}.")
            missing_damping = [name for name in controlled if name not in damping]
            if missing_damping:
                raise ValueError(f"joint controller damping has no key covering DOF(s) {missing_damping}.")
        return (
            [stiffness.get(name, 0.0) for name in robot.dof_names],
            [damping.get(name, 0.0) for name in robot.dof_names],
        )


@dataclass(frozen=True)
class ActionTermCfg:
    """Configuration for a single action-processing term."""

    func: str
    """Import path to the action term class (e.g. ``holosoma.managers.action.terms:JointPositionActionTerm``)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to initialize the action term."""

    scale: float | tuple[float, ...] = 1.0
    """Scaling factor(s) applied to the raw action values before processing."""

    clip: tuple[float, float] | None = None
    """Optional min/max clamp applied to the raw action values."""

    controller: JointControllerCfg | None = None
    """Software joint controller used by terms that compute joint torques."""


@dataclass(frozen=True)
class ActionManagerCfg:
    """Configuration for the action manager."""

    terms: dict[str, ActionTermCfg] = field(default_factory=dict)
    """Mapping of action term name to configuration."""


def replace_action_term(action: ActionManagerCfg, name: str, **changes: Any) -> ActionManagerCfg:
    """Return ``action`` with fields replaced on one existing term.

    The original action and its term mapping are not modified. Other terms retain their
    values and insertion order.
    """
    if name not in action.terms:
        raise KeyError(f"Action term {name!r} does not exist; available terms: {list(action.terms)}")

    valid_fields = {field.name for field in fields(ActionTermCfg)}
    unknown = sorted(set(changes) - valid_fields)
    if unknown:
        raise TypeError(f"Unknown ActionTermCfg field(s) {unknown}; valid fields: {sorted(valid_fields)}")

    terms = dict(action.terms)
    terms[name] = replace(terms[name], **changes)
    return replace(action, terms=terms)
