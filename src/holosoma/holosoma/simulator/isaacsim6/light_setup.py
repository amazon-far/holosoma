"""Lab 3 light configuration adapter."""

from functools import partial
from holosoma.simulator.isaacsim.api import ISAACLAB3
from holosoma.simulator.isaacsim.light_setup import build_lights as _build_lights

build_lights = partial(_build_lights, api=ISAACLAB3)
