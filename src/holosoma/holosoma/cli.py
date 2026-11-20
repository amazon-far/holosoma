"""The ``holosoma`` command: build the config its ``task`` runs, then call that task function."""

from __future__ import annotations

import importlib
import inspect
import sys
import typing
from collections.abc import Callable
from typing import Any

from holosoma.config_types.experiment import SIMULATE, ExperimentConfig, TrainingExperimentConfig, task_class_path


def resolve_task(task: str) -> tuple[type, Callable[[Any], None], type[ExperimentConfig]]:
    """Return the class, the classmethod and its config type that a ``module:Class.function`` task names.

    The function must be a classmethod whose single parameter is annotated with an ``ExperimentConfig``
    subclass; that annotation is the config type the command line builds.
    """
    module_name, _, attribute = task.partition(":")
    class_name, _, function_name = attribute.rpartition(".")
    try:
        owner = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise SystemExit(f"task {task!r}: cannot import {module_name}:{class_name} ({exc}).") from exc
    raw = inspect.getattr_static(owner, function_name, None)
    config_type = None
    if isinstance(raw, classmethod):
        function = getattr(owner, function_name)
        parameters = list(inspect.signature(function).parameters.values())
        if len(parameters) == 1:
            config_type = typing.get_type_hints(raw.__func__).get(parameters[0].name)
    if not (isinstance(config_type, type) and issubclass(config_type, ExperimentConfig)):
        raise SystemExit(
            f"{task!r} is not a runnable task function: it must be a classmethod whose single parameter is "
            "annotated with an ExperimentConfig subclass."
        )
    return owner, getattr(owner, function_name), config_type


def _pop_experiment(argv: list[str]) -> tuple[TrainingExperimentConfig | None, list[str]]:
    """The ``exp:<preset>`` the arguments select, and the arguments without it."""
    from holosoma.config_values.experiment import EXPERIMENT_REGISTRY

    tokens = [token for token in argv if token.startswith("exp:")]
    if not tokens:
        return None, argv
    if len(tokens) > 1:
        raise SystemExit(f"Only one exp: preset can be selected, got {tokens}.")
    name = tokens[0].removeprefix("exp:").replace("-", "_")
    if name not in EXPERIMENT_REGISTRY:
        choices = ", ".join(f"exp:{key.replace('_', '-')}" for key in EXPERIMENT_REGISTRY)
        raise SystemExit(f"Unknown experiment {tokens[0]!r}; choose from {choices}.")
    return EXPERIMENT_REGISTRY[name], [token for token in argv if token != tokens[0]]


def explicit_task(argv: list[str]) -> str | None:
    """The last ``--task`` value in the arguments, if any."""
    value = None
    for index, arg in enumerate(argv):
        if arg.startswith("--task="):
            value = arg.partition("=")[2]
        elif arg == "--task" and index + 1 < len(argv):
            value = argv[index + 1]
    return value


def run(args: list[str] | None = None, default_task: str = SIMULATE) -> None:
    """Parse ``args`` (default ``sys.argv``) into the config ``task`` names and run its task function.

    ``task`` comes from ``--task``, else the ``exp:`` preset, else ``default_task``. The config is
    parsed twice: once against the preset or the type's defaults to read the values the type's
    ``base`` depends on, then on top of that base.
    """
    from holosoma.envs.policy_task import PolicyTask
    from holosoma.utils.config_registry import load_plugins, parse_config
    from holosoma.utils.eval_utils import init_eval_logging
    from holosoma.utils.tyro_utils import split_import_file_args

    init_eval_logging()
    import_paths, argv = split_import_file_args(list(sys.argv[1:] if args is None else args))
    load_plugins(import_paths)
    exp, argv = _pop_experiment(argv)
    explicit = explicit_task(argv)
    task = explicit or (exp.task if exp is not None else default_task)
    owner, function, config_type = resolve_task(task)
    if explicit is not None and exp is not None and issubclass(owner, PolicyTask):
        if task_class_path(task) != task_class_path(exp.task):
            raise SystemExit(f"task {task!r} does not match the experiment's class {task_class_path(exp.task)!r}.")
    if explicit is None:
        argv = [f"--task={task}", *argv]
    preset = exp if isinstance(exp, config_type) else None
    first, _ = parse_config(config_type, default=preset, args=argv, return_unknown_args=True)
    config = parse_config(config_type, default=config_type.base(first, exp), args=argv)
    function(config)


def main() -> None:
    """Run the ``holosoma`` command."""
    run()


if __name__ == "__main__":
    main()
