#!/usr/bin/env python3
"""Replay a whole-body-tracking motion: ``holosoma`` with ``SimulationTask.replay`` as the task."""

import sys

from holosoma.cli import explicit_task, run
from holosoma.config_types.experiment import REPLAY


def main() -> None:
    argv = sys.argv[1:]
    run(argv if explicit_task(argv) else [f"--task={REPLAY}", *argv])


if __name__ == "__main__":
    main()
