"""Version-specific Isaac Lab interfaces.

Keep selection import-safe: the launcher uses this module before Kit or Torch
may be imported. Native simulator imports stay inside backend methods.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


@lru_cache(maxsize=2)
def _backend_for(name: str) -> IsaacLabBackend:
    if name == "isaaclab2":
        from .isaaclab2 import IsaacLab2Backend

        return IsaacLab2Backend()
    if name == "isaaclab3":
        from .isaaclab3 import IsaacLab3Backend

        return IsaacLab3Backend()
    raise ValueError(f"Unknown Isaac Sim backend {name!r}; expected 'isaaclab2' or 'isaaclab3'.")


def get_backend(backend: str | None = None) -> IsaacLabBackend:
    """Resolve and validate a backend against the installed Isaac Lab version.

    An explicit simulator config wins over the environment. Helpers without a
    simulator use the setup script's environment selection, then the installed
    major version. Lab 2 remains the fallback when the optional runtime is absent.
    """
    major = _installed_major()
    name = backend or os.environ.get("HOLOSOMA_ISAACSIM_BACKEND")
    if name is None:
        name = f"isaaclab{major}" if major is not None else "isaaclab2"
    selected = _backend_for(name)
    if major is not None and major != selected.major:
        raise RuntimeError(
            f"The {name} backend requires Isaac Lab {selected.major}.x; Isaac Lab {major}.x is installed. "
            "Activate the matching setup environment or select its simulator preset."
        )
    return selected
