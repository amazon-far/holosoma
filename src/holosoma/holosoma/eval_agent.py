"""Alias for ``holosoma task:eval``."""

from holosoma.cli import run_task


def main() -> None:
    run_task("eval")


if __name__ == "__main__":
    main()
