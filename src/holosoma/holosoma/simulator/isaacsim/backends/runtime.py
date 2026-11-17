"""Activate Kit's normal Python paths only when the selected runtime needs it.

Shell activation prepares native library lookup before Python starts. Standalone
runs retain ordinary namespace discovery; Kit runs expose the SDK packages before
argument registration imports ``isaacsim``. Neither path replaces SDK modules.
"""

from __future__ import annotations

import os
import sys


def activate_kit() -> None:
    """Expose the Kit packages prepared by ``source_isaaclab3_setup.sh``.

    Selecting Kit after the standalone namespace was imported cannot safely
    change the SDK's cached launcher class. Reject that order rather than
    replacing an imported package or restarting the caller's process.
    """
    paths = os.environ.get("ISAACLAB3_KIT_PYTHONPATH")
    if not paths:
        if os.environ.get("ISAACSIM_ROOT_PATH"):
            raise RuntimeError("Source scripts/source_isaaclab3_setup.sh before starting Python with Kit.")
        return
    existing = sys.modules.get("isaacsim")
    if existing is not None and vars(existing).get("__file__") is None:
        raise RuntimeError("Select the Kit runtime before importing the standalone isaacsim namespace.")
    sys.path.extend(path for path in paths.split(os.pathsep) if path and path not in sys.path)
