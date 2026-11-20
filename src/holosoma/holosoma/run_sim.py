#!/usr/bin/env python3
"""Run a simulation with no policy: ``holosoma`` with ``SimulationTask.simulate`` as the task."""

import sys

from holosoma.cli import explicit_task, run
from holosoma.config_types.experiment import SIMULATE

# This script's simulator names meant the sim-to-sim presets (bridge and virtual gantry on).
_SIM2SIM = {f"simulator:{name}": f"simulator:{name}-sim2sim" for name in ("isaacsim", "mujoco", "mjwarp")}


def main() -> None:
    argv = [_SIM2SIM.get(token, token) for token in sys.argv[1:]]
    run(argv if explicit_task(argv) else [f"--task={SIMULATE}", *argv])


if __name__ == "__main__":
    main()
