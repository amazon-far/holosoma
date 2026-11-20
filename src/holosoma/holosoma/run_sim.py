#!/usr/bin/env python3
"""Run a simulation with no policy: ``holosoma`` with ``SimulationTask.simulate`` as the default task."""

import sys

from holosoma.cli import run
from holosoma.config_types.experiment import SIMULATE

# This script's simulator names meant the interactive presets (bridge and virtual gantry on).
_INTERACTIVE = {f"simulator:{name}": f"env.simulator:sim-{name}" for name in ("isaacsim", "mujoco", "mjwarp")}


def main() -> None:
    run([_INTERACTIVE.get(token, token) for token in sys.argv[1:]], default_task=SIMULATE)


if __name__ == "__main__":
    main()
