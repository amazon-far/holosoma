"""Resolve runtime utilities after Isaac Lab has launched."""

import isaaclab.sim.utils as _utils

if hasattr(_utils, "get_current_stage"):
    import isaaclab.sim.utils as prim_utils
    import isaaclab.sim.utils.stage as stage_utils
else:
    import isaacsim.core.utils.prims as prim_utils
    import isaacsim.core.utils.stage as stage_utils
