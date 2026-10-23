"""Compatibility import for the shared Isaac Sim sensor pipeline."""

import sys
from holosoma.simulator.isaacsim import sensor_setup as _shared

sys.modules[__name__] = _shared
