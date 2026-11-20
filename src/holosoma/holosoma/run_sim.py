#!/usr/bin/env python3
"""Run a simulation with no policy: ``holosoma`` with ``SimulationTask.simulate`` as the default task."""

from holosoma.cli import run
from holosoma.config_types.experiment import SIMULATE


def main() -> None:
    run(default_task=SIMULATE)


if __name__ == "__main__":
    main()
