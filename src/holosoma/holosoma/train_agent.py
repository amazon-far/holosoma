#!/usr/bin/env python3
"""Train a policy: ``holosoma`` with the ``exp:`` preset's training task (or the locomotion one) as the default."""

from holosoma.cli import run
from holosoma.config_types.experiment import TRAIN


def main() -> None:
    run(default_task=TRAIN)


if __name__ == "__main__":
    main()
