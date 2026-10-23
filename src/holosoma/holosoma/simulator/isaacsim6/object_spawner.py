"""Lab 3 profile for the shared object and scene-file spawners."""

from functools import partial
from holosoma.simulator.isaacsim.api import ISAACLAB3
from holosoma.simulator.isaacsim import object_spawner as _shared
from holosoma.simulator.isaacsim.object_spawner import *  # noqa: F403

build_standalone_rigid_object = partial(_shared.build_standalone_rigid_object, api=ISAACLAB3)
expand_scene_file = partial(_shared.expand_scene_file, api=ISAACLAB3)
