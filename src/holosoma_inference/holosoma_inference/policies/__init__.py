from typing import TYPE_CHECKING, Any

from .base import BasePolicy
from .dual_mode import DualModePolicy
from .locomotion import LocomotionPolicy

if TYPE_CHECKING:
    from .wbt import WholeBodyTrackingPolicy

__all__ = ["BasePolicy", "DualModePolicy", "LocomotionPolicy", "WholeBodyTrackingPolicy"]


# PEP 562 module-level hook.
def __getattr__(name: str) -> Any:
    """Resolve ``WholeBodyTrackingPolicy`` on first access.

    ``policies.wbt_utils`` imports pinocchio at module scope, so re-exporting the class eagerly makes
    every consumer of any policy pay for a rigid-body dynamics library and the numpy major version it
    pins. Locomotion and squat never call it. ``_select_policy_class`` and ``run_policy`` already
    import the class by module path, so this only defers the package-level alias.
    """
    if name == "WholeBodyTrackingPolicy":
        from .wbt import WholeBodyTrackingPolicy

        return WholeBodyTrackingPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
