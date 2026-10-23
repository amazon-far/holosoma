"""Lab 3 configuration profile for shared robot and contact builders."""

from functools import partial
from holosoma.simulator.isaacsim.api import ISAACLAB3
from holosoma.simulator.isaacsim import isaacsim_articulation_cfg as _shared
from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import _per_joint_map

build_robot_articulation_cfg = partial(_shared.build_robot_articulation_cfg, api=ISAACLAB3)
build_contact_sensor_cfg = partial(_shared.build_contact_sensor_cfg, api=ISAACLAB3)
