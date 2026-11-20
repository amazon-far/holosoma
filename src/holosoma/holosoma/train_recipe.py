"""Run a training recipe by name.

    python -m holosoma.train_recipe v19_fixed --seed 87
    python -m holosoma.train_recipe v19_fixed --describe            # what every phase gets
    python -m holosoma.train_recipe v19_fixed --describe DE --docs  # with the definitions
    python -m holosoma.train_recipe v19_fixed --set DE:command.loco.stand_prob=0.3
    python -m holosoma.train_recipe v19_fixed --logger disabled     # no Weights & Biases

Recipes live in :mod:`holosoma.recipes.catalog`, and the quantities they set
are defined in :mod:`holosoma.recipes.keys`. ``--set`` wins over everything in
the recipe, so a one-off probe never needs the recipe edited.
"""

from __future__ import annotations

import argparse
import ast
import sys

from loguru import logger

from holosoma.recipes import catalog
from holosoma.recipes.recipe import Recipe
from holosoma.training_pipeline_renew import run_renew_pipeline

__all__ = ["base_args_for", "apply_cli_sets", "main"]

_SEED_TAGS = {55: "s1", 87: "s2", 199: "s3"}


def base_args_for(
    recipe: Recipe, *, seed: int, run_name: str, entity: str | None = None,
    simulator: str = "isaacsim", logger_backend: str = "wandb", headless: bool = True,
) -> list[str]:
    """The base CLI the pipeline parses: experiment, backends and run identity."""
    args = [
        f"exp:{recipe.exp}",
        f"simulator:{simulator}",
        f"logger:{logger_backend}",
    ]
    if entity:
        args += ["--logger.entity", entity]
    return args + [
        "--training.headless", str(headless),
        "--training.seed", str(seed),
        "--logger.name", run_name,
    ]


def apply_cli_sets(recipe: Recipe, assignments: list[str]) -> None:
    """Apply ``--set PHASE:key=value`` (or ``--set key=value`` for every phase).

    Values are parsed as Python literals, falling back to the raw string.
    """
    for raw in assignments:
        target, _, value_str = raw.partition("=")
        if not value_str:
            raise ValueError(f"--set needs KEY=VALUE, got {raw!r}")
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str
        phase_key, sep, key_name = target.partition(":")
        if sep:
            recipe.set(phase_key, key_name, value, by="--set")
            logger.info(f"--set {phase_key}.{key_name} = {value!r}")
        else:
            recipe.set_all(target, value, by="--set")
            logger.info(f"--set (all phases) {target} = {value!r}")


def main() -> int | None:
    parser = argparse.ArgumentParser(
        description="Run a named training recipe.", allow_abbrev=False
    )
    parser.add_argument("recipe", choices=sorted(catalog.RECIPES), help="recipe name")
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument("--run-name", default=None,
                        help="W&B run name (default: lm_gac_final_<recipe>_<seed tag>)")
    parser.add_argument("--entity", default=None, help="W&B entity (default: your W&B default)")
    parser.add_argument("--logger", default="wandb", choices=("wandb", "wandb_offline", "disabled"),
                        help="logging backend")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--start-checkpoint", default=None)
    parser.add_argument("--start-phase", type=int, default=None,
                        help="0-based phase index to resume at, with --start-checkpoint")
    parser.add_argument("--set", action="append", default=[], metavar="PHASE:KEY=VALUE",
                        help="override one value; repeatable. Wins over the recipe.")
    parser.add_argument("--phases", default=None, metavar="A,B",
                        help="keep only these phases (comma-separated keys), e.g. for a short test")
    parser.add_argument("--iterations", action="append", default=[], metavar="PHASE=N",
                        help="override one phase's iteration count; repeatable")
    parser.add_argument("--describe", nargs="?", const="ALL", default=None, metavar="PHASE",
                        help="print the resolved recipe and exit; optionally one phase")
    parser.add_argument("--docs", action="store_true",
                        help="with --describe, also print each key's definition")
    args = parser.parse_args()

    recipe = catalog.get(args.recipe)
    apply_cli_sets(recipe, args.set)
    if args.phases:
        recipe.keep_phases(*args.phases.split(","))
    for item in args.iterations:
        phase_key, _, n = item.partition("=")
        recipe.set_iterations({phase_key: int(n)})

    if args.describe:
        phase = None if args.describe == "ALL" else args.describe
        print(recipe.describe(phase, docs=args.docs))
        return

    seed_tag = _SEED_TAGS.get(args.seed, f"seed{args.seed}")
    run_name = args.run_name or f"lm_gac_final_{args.recipe}_{seed_tag}"
    logger.info(f"=== recipe {args.recipe} | run {run_name} | seed {args.seed} ===")

    results = run_renew_pipeline(
        base_args=base_args_for(
            recipe, seed=args.seed, run_name=run_name, entity=args.entity, logger_backend=args.logger
        ),
        phases=recipe.resolve(),
        results_dir="logs/training_pipeline_lm_renew",
        shutdown_grace_seconds=30.0,
        start_checkpoint=args.start_checkpoint,
        start_phase=args.start_phase,
        num_gpus=args.num_gpus,
    )
    # Exit non-zero when a phase failed, so a cluster job (or any caller)
    # reports the failure instead of success.
    return 1 if any(r["returncode"] != 0 for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
