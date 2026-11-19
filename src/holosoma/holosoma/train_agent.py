"""Alias for ``holosoma task:train``."""

from holosoma.cli import run_task


def main() -> None:
    run_task("train")


if __name__ == "__main__":
    main()
