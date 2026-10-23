"""Lab 3 namespace adapter for the shared USD helpers."""

from functools import partial
from holosoma.simulator.isaacsim import prim_utils as _shared
from holosoma.simulator.isaacsim.api import ISAACLAB3
from holosoma.simulator.isaacsim.prim_utils import *  # noqa: F403
from holosoma.simulator.isaacsim.prim_utils import _find_robot_link_prim, _disambiguate_link_prims

get_current_stage = partial(_shared.get_current_stage, api=ISAACLAB3)
resolve_robot_link_prim_expression = partial(_shared.resolve_robot_link_prim_expression, api=ISAACLAB3)
resolve_robot_link_rigid_body_prim_expression = partial(
    _shared.resolve_robot_link_rigid_body_prim_expression, api=ISAACLAB3
)
