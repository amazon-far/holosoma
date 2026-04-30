"""Run SuccessRateCallback for every model_*.pt in a log directory.

Builds the env + algo once (using the latest checkpoint's saved config),
then iterates over each checkpoint:

    1. ``algo.load(ckpt_path)``                     — swap student weights
    2. inject a fresh ``SuccessRateCallback``       — clears per-motion state
    3. ``algo.evaluate_policy(max_eval_steps=None)``— runs all motion batches
    4. record ``Eval/success_rate_0.5m`` (and 0.25m)

After the sweep, saves a JSON of {iteration: success_rate_0.5m} and a PNG
plot. Designed for the DAgger student log dir, but works for any algo with
``evaluate_policy`` + ``eval_callbacks`` support.

Usage:
    python src/holosoma/holosoma/eval_success_rate_sweep.py \
        --log_dir logs/MotionTracking/20260428_g1_unreal_student_DAgger \
        --motion_dir src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/Unreal_Engine_stair_subset200_aligned \
        --num_envs 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path

from loguru import logger

from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment


_CKPT_RE = re.compile(r"^model_(\d+)\.pt$")


def _parse_iteration(name: str) -> int | None:
    m = _CKPT_RE.match(name)
    return int(m.group(1)) if m else None


def main():
    init_eval_logging()
    parser = argparse.ArgumentParser(description="Sweep SuccessRateCallback over a log dir's checkpoints")
    parser.add_argument("--log_dir", required=True, help="Directory containing model_*.pt files")
    parser.add_argument("--motion_dir", default=None, help="Override motion_dir for evaluation")
    parser.add_argument("--split_file", default=None, help="Optional split_file override")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Parallel envs during eval (default: same as max_motions from saved config).",
    )
    parser.add_argument(
        "--max_motions",
        type=int,
        default=None,
        help="Override max_motions / pool_size / obj_max_tiles. Defaults to the value saved in the training config.",
    )
    parser.add_argument(
        "--reference_ckpt",
        default=None,
        help="Checkpoint path used to load the saved ExperimentConfig. Defaults to the latest model_*.pt in --log_dir.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Where to save the sweep JSON + plot. Defaults to <log_dir>/sweep_success_rate.",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip checkpoints already in saved JSON.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.is_dir():
        raise FileNotFoundError(f"log_dir not found: {log_dir}")

    # Discover checkpoints (sorted by iteration).
    ckpt_paths: list[tuple[int, Path]] = []
    for p in sorted(log_dir.glob("model_*.pt")):
        it = _parse_iteration(p.name)
        if it is not None:
            ckpt_paths.append((it, p))
    if not ckpt_paths:
        raise FileNotFoundError(f"No model_*.pt files in {log_dir}")
    ckpt_paths.sort(key=lambda x: x[0])
    logger.info(f"Found {len(ckpt_paths)} checkpoints (iter {ckpt_paths[0][0]}–{ckpt_paths[-1][0]})")

    out_dir = Path(args.out_dir) if args.out_dir else log_dir / "sweep_success_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "success_rate.json"

    # Load saved config from the reference (latest by default) checkpoint.
    reference_ckpt = Path(args.reference_ckpt) if args.reference_ckpt else ckpt_paths[-1][1]
    logger.info(f"Loading saved config from reference checkpoint: {reference_ckpt}")
    saved_cfg, saved_wandb_path = load_saved_experiment_config(CheckpointConfig(checkpoint=str(reference_ckpt)))

    # Resolve max_motions / pool_size / num_envs / obj_max_tiles. Multi-terrain
    # training uses pool_ordered=True + bind_terrain_tiles=True which together
    # require pool_size == #motion files == terrain tile count, so all three
    # need to agree. Pull max_motions from the saved config unless overridden.
    saved_motion_cfg = saved_cfg.command.setup_terms["motion_command"].params["motion_config"]
    saved_max_motions = getattr(saved_motion_cfg, "max_motions", None)
    if isinstance(saved_motion_cfg, dict):
        saved_max_motions = saved_motion_cfg.get("max_motions", saved_max_motions)
    n_motions = args.max_motions if args.max_motions is not None else saved_max_motions
    if not n_motions or n_motions <= 0:
        raise ValueError(
            "Could not infer max_motions from saved config; pass --max_motions explicitly."
        )
    n_envs = args.num_envs if args.num_envs is not None else n_motions
    logger.info(
        f"Eval alignment: max_motions={n_motions}, pool_size={n_motions}, "
        f"obj_max_tiles={n_motions}, num_envs={n_envs}"
    )

    # Apply eval overrides — same shape as eval_success_rate.py.
    saved_cfg = dataclasses.replace(
        saved_cfg,
        eval_overrides=dataclasses.replace(
            saved_cfg.eval_overrides,
            num_envs=n_envs,
            headless=True,
        ),
    )
    eval_cfg = saved_cfg.get_eval_config()

    # Override motion_dir / split_file / max_motions / pool_size if needed.
    # Mirror eval_success_rate.py's in-place mutation pattern.
    motion_config = eval_cfg.command.setup_terms["motion_command"].params["motion_config"]
    if isinstance(motion_config, dict):
        if args.motion_dir is not None:
            motion_config["motion_dir"] = args.motion_dir
        if args.split_file is not None:
            motion_config["split_file"] = args.split_file
        motion_config["max_motions"] = n_motions
        motion_config["pool_size"] = n_motions
    else:
        updates = {"max_motions": n_motions, "pool_size": n_motions}
        if args.motion_dir is not None:
            updates["motion_dir"] = args.motion_dir
        if args.split_file is not None:
            updates["split_file"] = args.split_file
        new_motion_config = dataclasses.replace(motion_config, **updates)
        eval_cfg.command.setup_terms["motion_command"].params["motion_config"] = new_motion_config
    if args.motion_dir:
        logger.info(f"Overriding motion_dir -> {args.motion_dir}")

    # Cap terrain tile count to match motion count (multi-mesh terrain).
    try:
        terrain_term = eval_cfg.terrain.terrain_term
        if hasattr(terrain_term, "obj_max_tiles"):
            eval_cfg = dataclasses.replace(
                eval_cfg,
                terrain=dataclasses.replace(
                    eval_cfg.terrain,
                    terrain_term=dataclasses.replace(terrain_term, obj_max_tiles=n_motions),
                ),
            )
    except (AttributeError, TypeError) as e:
        logger.warning(f"Could not override terrain.obj_max_tiles ({e}); relying on motion count match.")

    # Build env + algo once.
    env, device, simulation_app = setup_simulation_environment(eval_cfg)

    eval_log_dir = get_experiment_dir(
        eval_cfg.logger, eval_cfg.training, get_timestamp(), task_name="eval_success_rate_sweep"
    ).resolve()
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Sweep eval log dir: {eval_log_dir}")

    algo_class = get_class(eval_cfg.algo._target_)
    algo = algo_class(
        device=device,
        env=env,
        config=eval_cfg.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)

    # Lazy-import the callback so the sim app has fully initialised.
    from holosoma.agents.callbacks.success_rate_callback import SuccessRateCallback

    # Resume previous sweep results if requested.
    results: dict[str, dict[str, float]] = {}
    if args.skip_existing and json_path.is_file():
        with open(json_path) as f:
            results = json.load(f)
        logger.info(f"Resuming with {len(results)} previously-evaluated iterations.")

    try:
        for idx, (iteration, ckpt_path) in enumerate(ckpt_paths):
            key = str(iteration)
            if args.skip_existing and key in results:
                logger.info(f"[{idx + 1}/{len(ckpt_paths)}] iter {iteration}: cached, skipping.")
                continue

            logger.info(f"[{idx + 1}/{len(ckpt_paths)}] iter {iteration}: loading {ckpt_path.name}")
            algo.load(str(ckpt_path))

            # Fresh callback per-iteration (its internal state is per-motion).
            algo.eval_callbacks = [SuccessRateCallback(config=None, training_loop=algo)]
            algo.evaluate_policy(max_eval_steps=None)

            metrics = {}
            for cb in algo.eval_callbacks:
                if hasattr(cb, "metrics") and cb.metrics:
                    metrics.update(cb.metrics)
            sr_05 = float(metrics.get("Eval/success_rate_0.5m", float("nan")))
            sr_025 = float(metrics.get("Eval/success_rate_0.25m", float("nan")))
            num_motions = float(metrics.get("Eval/num_motions", 0.0))
            results[key] = {
                "iteration": iteration,
                "success_rate_0.5m": sr_05,
                "success_rate_0.25m": sr_025,
                "num_motions": num_motions,
            }
            logger.info(
                f"  iter {iteration}: success_rate_0.5m={sr_05:.4f}, success_rate_0.25m={sr_025:.4f} "
                f"(n={num_motions:.0f})"
            )

            # Write incrementally so a crash mid-sweep doesn't lose everything.
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)
    finally:
        # Always plot whatever we have, even if we crash mid-sweep.
        _plot(results, out_dir / "success_rate.png")
        if simulation_app:
            close_simulation_app(simulation_app)


def _plot(results: dict[str, dict[str, float]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    if not results:
        logger.warning("No results to plot.")
        return

    rows = sorted(results.values(), key=lambda r: r["iteration"])
    iters = [r["iteration"] for r in rows]
    sr05 = [r["success_rate_0.5m"] for r in rows]
    sr025 = [r["success_rate_0.25m"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, sr05, marker="o", label="success_rate @ 0.5m")
    ax.plot(iters, sr025, marker="x", linestyle="--", alpha=0.6, label="success_rate @ 0.25m")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"DAgger student success rate over training (n={int(rows[0]['num_motions'])} motions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
