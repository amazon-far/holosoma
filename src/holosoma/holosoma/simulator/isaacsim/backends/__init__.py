"""Version-specific Isaac Lab interfaces.

Keep selection import-safe: the launcher uses this module before Kit or Torch
may be imported. Native simulator imports stay inside backend methods.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holosoma.config_types.simulator import IsaacLab3Config

    from .base import IsaacLabBackend


def _installed_major() -> int | None:
    # Isaac Lab's extension metadata versions (e.g. 0.54.2 and 28.0.0)
    # are unrelated to the v2/v3 release tags. The separate PhysX package
    # is the defining Lab 3 API boundary and can be detected without import.
    try:
        if importlib.util.find_spec("isaaclab_physx") is not None:
            return 3
    except ValueError:
        # External-boundary tests can provide modules without import specs.
        pass
    try:
        importlib.metadata.version("isaaclab")
    except importlib.metadata.PackageNotFoundError:
        return None
    return 2


def get_backend(backend: str, *, native: IsaacLab3Config | None = None) -> IsaacLabBackend:
    """Construct the explicit API family and validate the installed runtime.

    Selection comes from the simulator config. Installed package capabilities
    validate that choice; they never choose a backend. Each owner receives its
    own adapter and native options, without process-shared mutable state.
    """
    major = _installed_major()
    name = backend
    if name == "isaaclab2":
        from .isaaclab2 import IsaacLab2Backend

        selected: IsaacLabBackend = IsaacLab2Backend()
    elif name == "isaaclab3":
        from .isaaclab3 import IsaacLab3Backend

        selected = IsaacLab3Backend(native)
    else:
        raise ValueError(f"Unknown Isaac Sim backend {name!r}; expected 'isaaclab2' or 'isaaclab3'.")
    if major is not None and major != selected.major:
        raise RuntimeError(
            f"The {name} backend requires Isaac Lab {selected.major}.x; Isaac Lab {major}.x is installed. "
            "Activate the matching setup environment or select its simulator preset."
        )
    return selected
