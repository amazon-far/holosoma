"""Alias for ``holosoma task:replay``."""

from holosoma.cli import run_task


def main() -> None:
    run_task("replay")


if __name__ == "__main__":
    main()
