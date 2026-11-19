"""Replay a whole-body-tracking motion: ``python -m holosoma.run_sim task:replay ...``."""

from holosoma.run_sim import run_task


def main() -> None:
    run_task("replay")


if __name__ == "__main__":
    main()
