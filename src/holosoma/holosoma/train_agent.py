#!/usr/bin/env python3
"""Train a policy: ``holosoma`` with the ``exp:`` preset's training task (or the locomotion one) as the default."""

from holosoma.cli import run
from holosoma.config_types.experiment import LOCOMOTION_TRAIN


def main() -> None:
    run(default_task=LOCOMOTION_TRAIN)


if __name__ == "__main__":
    main()
