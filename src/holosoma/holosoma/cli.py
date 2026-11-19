"""The ``holosoma`` command: run the task selected by ``task:<name>`` or the ``--config`` document.

``sim`` (the default) runs a simulation with no policy; ``train``, ``eval``, ``replay`` and any
extension task come from ``TASK_REGISTRY``. ``python -m holosoma`` is the same command.
"""

from __future__ import annotations

import dataclasses
import sys


def main() -> None:
    """Parse the selected task's config from ``sys.argv`` and run it."""
    from holosoma.config_types.experiment import ExperimentConfig
    from holosoma.config_values.task import TASK_REGISTRY
    from holosoma.utils.config_document import ConfigDocumentError, select_task
    from holosoma.utils.config_registry import load_plugins
    from holosoma.utils.helpers import get_class

    try:
        name, argv = select_task(sys.argv[1:])
    except ConfigDocumentError as exc:
        raise SystemExit(str(exc)) from exc
    sys.argv = [sys.argv[0], *argv]

    load_plugins()
    tasks = {key.replace("_", "-"): value for key, value in TASK_REGISTRY.items()}
    # A task's arguments are spelled `<task>:` in documents and `--<task>.*` on the CLI.
    fields = {field.name.replace("_", "-") for field in dataclasses.fields(ExperimentConfig)}
    shadowing = sorted(set(tasks) & fields)
    if shadowing:
        raise SystemExit(f"Task names {shadowing} shadow ExperimentConfig fields; register them under other names.")
    name = (name or "sim").replace("_", "-")
    if name not in tasks:
        raise SystemExit(f"Unknown task {name!r}; choose from {sorted(tasks)}.")
    get_class(tasks[name]._target_).from_args(argv).run()


def run_task(name: str) -> None:
    """Run ``main`` with ``task:<name>`` selected, for the per-task alias scripts."""
    sys.argv.insert(1, f"task:{name}")
    main()
