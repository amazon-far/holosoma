"""Render a co-simulation source's state with a :class:`~holosoma.envs.render_task.RenderTask`.

usage: python -m holosoma.cosim.render --endpoint ipc://... <run_sim arguments>

The run_sim arguments describe the same scene as the source's, with the source's remote cameras configured
here as ordinary cameras. Start this before the source; it exits when the source closes.
"""

from __future__ import annotations

import sys

from loguru import logger

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.cosim.bridge import Receiver
from holosoma.utils.config_registry import parse_config


def split_endpoint(argv: list[str]) -> tuple[str, list[str]]:
    """Return the ``--endpoint`` value and the remaining arguments."""
    for index, arg in enumerate(argv):
        if arg.startswith("--endpoint="):
            return arg.partition("=")[2], argv[:index] + argv[index + 1 :]
        if arg == "--endpoint" and index + 1 < len(argv):
            return argv[index + 1], argv[:index] + argv[index + 2 :]
    raise SystemExit("usage: python -m holosoma.cosim.render --endpoint ipc://... <run_sim arguments>")


def main(argv: list[str] | None = None) -> None:
    endpoint, rest = split_endpoint(list(sys.argv[1:] if argv is None else argv))
    config = parse_config(RunSimConfig, args=rest)
    with config.create_render_task() as task:
        Receiver(task, endpoint=endpoint).run()
        logger.info(f"Co-simulation source closed after {task.frames_rendered} frames")


if __name__ == "__main__":
    main()
