"""Compatibility import; implementation lives in the Isaac Sim 5.1 package."""

import sys
from holosoma.simulator.isaacsim import physics_properties as _shared

sys.modules[__name__] = _shared
