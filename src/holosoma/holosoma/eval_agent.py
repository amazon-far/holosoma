"""Evaluate a checkpoint: ``python -m holosoma.run_sim task:eval ...``."""

from holosoma.run_sim import run_task


def main() -> None:
    run_task("eval")


if __name__ == "__main__":
    main()
