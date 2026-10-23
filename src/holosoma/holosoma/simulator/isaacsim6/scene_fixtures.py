"""Lab 3 profile for shared terrain fixtures."""

from functools import partial
from holosoma.simulator.isaacsim.api import ISAACLAB3
from holosoma.simulator.isaacsim.scene_fixtures import build_terrain_fixtures as _build_terrain_fixtures

build_terrain_fixtures = partial(_build_terrain_fixtures, api=ISAACLAB3)
