#!/usr/bin/env python3
"""Evaluate a checkpoint: ``holosoma`` with the evaluate function of the class the checkpoint was trained with."""

from __future__ import annotations

import sys

from holosoma.cli import explicit_task, run

_CALLBACK_PREFIXES = ("--recording.", "--push.", "--payload.")


def _rewrite(argv: list[str]) -> list[str]:
    """Spell this script's older flags by their config paths: ``--checkpoint`` and the callback flags."""
    rewritten = []
    for token in argv:
        name, eq, value = token.partition("=")
        if name == "--checkpoint":
            rewritten.append(f"--eval.checkpoint{eq}{value}")
        elif token.startswith(_CALLBACK_PREFIXES):
            rewritten.append(f"--eval.callbacks.{token[2:]}")
        else:
            rewritten.append(token)
    return rewritten


def _checkpoint(argv: list[str]) -> str | None:
    value = None
    for index, arg in enumerate(argv):
        if arg.startswith("--eval.checkpoint="):
            value = arg.partition("=")[2]
        elif arg == "--eval.checkpoint" and index + 1 < len(argv):
            value = argv[index + 1]
    return value


def main() -> None:
    from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config

    argv = _rewrite(sys.argv[1:])
    checkpoint = _checkpoint(argv)
    if checkpoint is None:
        raise SystemExit("eval_agent needs --eval.checkpoint=<path or wandb:// URI> (or --checkpoint=...).")
    if explicit_task(argv) is None:
        saved, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=checkpoint))
        argv = [f"--task={saved.task.rpartition('.')[0]}.evaluate", *argv]
    run(argv)


if __name__ == "__main__":
    main()
