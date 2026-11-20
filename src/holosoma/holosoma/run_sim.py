#!/usr/bin/env python3
"""Run a simulation with no policy: ``holosoma`` with ``SimulationTask.simulate`` as the task."""

import sys

from holosoma.cli import explicit_task, run
from holosoma.config_types.experiment import SIMULATE

# This script's simulator names meant the interactive presets (bridge and virtual gantry on).
_INTERACTIVE = {f"simulator:{name}": f"simulator:sim-{name}" for name in ("isaacsim", "mujoco", "mjwarp")}


def main() -> None:
    argv = [_INTERACTIVE.get(token, token) for token in sys.argv[1:]]
    run(argv if explicit_task(argv) else [f"--task={SIMULATE}", *argv])


if __name__ == "__main__":
    main()
