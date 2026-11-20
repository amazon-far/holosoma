"""Run a recipe's curriculum phases, each in its own subprocess.

Phases end early once they meet their graduation condition (e.g. average
episode length >= 990), parameters can ramp within a phase (sigma annealing),
and a checkpoint is always saved on early termination. The next phase starts
from the previous phase's checkpoint.

Entry point: ``python -m holosoma.train_recipe <recipe>``, which builds the
phases from :mod:`holosoma.recipes.catalog` and calls :func:`run_renew_pipeline`.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import signal
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from holosoma.agents.early_stop import EarlyStopCfg
from holosoma.agents.sigma_scheduler import SigmaSchedulerCfg
from holosoma.training_pipeline import (
    PhaseConfig,
    _kill_process_tree,
    _read_checkpoint_global_step,
    build_config_for_phase,
    discover_latest_checkpoint,
)


@dataclass
class RenewPhaseConfig:
    """A single training phase with optional early stop and sigma scheduling."""

    name: str
    """Human-readable phase name."""

    num_iterations: int
    """Maximum number of training iterations (acts as cap with early stop)."""

    overrides: dict[str, Any] = field(default_factory=dict)
    """CLI parameter overrides applied for this phase."""

    early_stop: EarlyStopCfg | None = None
    """If set, phase can terminate early when condition is met."""

    sigma_schedule: SigmaSchedulerCfg | None = None
    """If set, reward params are annealed in-process during this phase."""


def _generate_renew_phase_script(
    config: Any,
    script_path: Path,
    early_stop_cfg: EarlyStopCfg | None = None,
    sigma_schedule_cfg: SigmaSchedulerCfg | None = None,
) -> None:
    """Write a self-contained Python script that trains with early stop / sigma scheduling.

    The config and optional early_stop/sigma_schedule are serialized via pickle.
    The generated script loads them at runtime, attaches them to the agent, and
    calls train() directly.
    """
    pkl_path = script_path.with_suffix(".pkl")
    payload = {
        "config": config,
        "early_stop_cfg": early_stop_cfg,
        "sigma_schedule_cfg": sigma_schedule_cfg,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    script = textwrap.dedent(f"""\
        #!/usr/bin/env python
        \"\"\"Auto-generated renewed pipeline phase script.\"\"\"
        import os
        import pickle
        import signal
        import sys

        def main():
            with open({str(pkl_path)!r}, "rb") as f:
                payload = pickle.load(f)

            config = payload["config"]
            early_stop_cfg = payload["early_stop_cfg"]
            sigma_schedule_cfg = payload["sigma_schedule_cfg"]

            from holosoma.train_agent import train
            train(
                config,
                early_stop_cfg=early_stop_cfg,
                sigma_schedule_cfg=sigma_schedule_cfg,
            )

            # Force-kill to release GPU resources
            os.kill(os.getpid(), signal.SIGKILL)

        if __name__ == "__main__":
            main()
    """)
    script_path.write_text(script)
    script_path.chmod(0o755)


def build_renew_config_for_phase(
    base_args: list[str],
    phase: RenewPhaseConfig,
    phase_idx: int,
    checkpoint_path: Path | None = None,
    phase_log_dir: Path | None = None,
    wandb_run_id: str | None = None,
) -> Any:
    """Build ExperimentConfig for a renewed phase.

    Delegates to the existing build_config_for_phase with a temporary PhaseConfig.
    """
    # Convert RenewPhaseConfig to PhaseConfig for compatibility
    legacy_phase = PhaseConfig(
        name=phase.name,
        num_iterations=phase.num_iterations,
        overrides=phase.overrides,
    )
    return build_config_for_phase(
        base_args, legacy_phase, phase_idx, checkpoint_path, phase_log_dir,
        wandb_run_id=wandb_run_id,
    )


# A phase is killed as hung once nothing under its log dir (train log,
# tfevents, W&B files, checkpoints) has been written for this long. Logging
# every 100 iterations writes about once a minute.
PHASE_STALL_SECONDS = 1800.0


_POLL_SECONDS = 10.0


_ACTIVITY_CHECK_SECONDS = 60.0


def _newest_mtime(root: Path) -> float:
    """Latest modification time of any file under ``root``; 0 if there is none."""
    newest = 0.0
    for path in root.rglob("*"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            pass
    return newest


def run_renew_phase(
    phase_idx: int,
    phase: RenewPhaseConfig,
    base_args: list[str],
    checkpoint_path: Path | None = None,
    phase_log_dir: Path | None = None,
    wandb_run_id: str | None = None,
    python: str = sys.executable,
    shutdown_grace_seconds: float = 30.0,
    num_gpus: int = 1,
) -> dict[str, Any]:
    """Launch a single renewed phase as a subprocess.

    Handles early stop detection via marker file and sigma scheduling via
    pickle injection.
    """
    # Build the ExperimentConfig
    experiment_config = build_renew_config_for_phase(
        base_args, phase, phase_idx, checkpoint_path, phase_log_dir,
        wandb_run_id=wandb_run_id,
    )

    # Prepare sigma schedule with correct phase_start_step
    sigma_cfg = None
    if phase.sigma_schedule is not None:
        # Set phase_start_step from checkpoint
        phase_start_step = 0
        if checkpoint_path is not None:
            phase_start_step = _read_checkpoint_global_step(checkpoint_path)
        sigma_cfg = dataclasses.replace(
            phase.sigma_schedule,
            phase_start_step=phase_start_step,
        )

    # Write phase script
    scripts_dir = Path(phase_log_dir) if phase_log_dir else Path("logs/training_pipeline_renew")
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / f"_phase_{phase_idx}_runner.py"
    _generate_renew_phase_script(
        experiment_config, script_path,
        early_stop_cfg=phase.early_stop,
        sigma_schedule_cfg=sigma_cfg,
    )

    # Build command
    if num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--standalone",
            "--nnodes=1",
            str(script_path),
        ]
    else:
        cmd = [python, str(script_path)]

    logger.info(
        f"[Phase {phase_idx}: {phase.name}] Starting "
        f"(max {phase.num_iterations} iterations"
        f"{', early_stop enabled' if phase.early_stop else ''}"
        f"{', sigma_schedule enabled' if sigma_cfg else ''})"
    )
    if checkpoint_path:
        logger.info(f"[Phase {phase_idx}] Checkpoint: {checkpoint_path}")

    start_time = time.time()

    proc = subprocess.Popen(cmd, start_new_session=True)

    # Poll for completion. Two guards against a hung phase, and both count as a
    # failure: a total budget, generous because it has to cover the slowest
    # run on the slowest machine (3 s per iteration, several times what a
    # single-step FastSAC iteration takes), and a stall check that catches a
    # real hang much sooner.
    phase_timeout = max(phase.num_iterations * 3.0 + 600, 1200)
    phase_start = time.time()
    checkpoint_detected = False
    killed_reason: str | None = None
    last_activity = next_activity_check = phase_start

    try:
        while True:
            try:
                proc.wait(timeout=_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                elapsed_so_far = time.time() - phase_start

                # Check for graduation marker (early stop)
                if phase_log_dir is not None and not checkpoint_detected:
                    markers = list(Path(phase_log_dir).glob("**/_graduated.marker"))
                    if markers:
                        checkpoint_detected = True
                        logger.info(
                            f"[Phase {phase_idx}] Graduation marker detected. "
                            f"Waiting {shutdown_grace_seconds}s for clean exit..."
                        )

                # Check for final checkpoint (normal completion)
                if not checkpoint_detected and phase_log_dir is not None:
                    expected_step = experiment_config.algo.config.num_learning_iterations
                    final_name = f"model_{expected_step:07d}.pt"
                    finals = list(Path(phase_log_dir).glob(f"**/{final_name}"))
                    if finals:
                        checkpoint_detected = True
                        logger.info(
                            f"[Phase {phase_idx}] Final checkpoint detected. "
                            f"Waiting {shutdown_grace_seconds}s for clean exit..."
                        )

                if checkpoint_detected:
                    try:
                        proc.wait(timeout=shutdown_grace_seconds)
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            f"[Phase {phase_idx}] Force-killing hung process (pid={proc.pid})"
                        )
                        _kill_process_tree(proc.pid)
                        proc.wait()
                    break

                now = time.time()
                if phase_log_dir is not None and now >= next_activity_check:
                    last_activity = max(last_activity, _newest_mtime(Path(phase_log_dir)))
                    next_activity_check = now + _ACTIVITY_CHECK_SECONDS
                    if now - last_activity > PHASE_STALL_SECONDS:
                        killed_reason = f"stalled, nothing written for {now - last_activity:.0f}s"
                if elapsed_so_far > phase_timeout:
                    killed_reason = f"timeout after {phase_timeout:.0f}s"
                if killed_reason is not None:
                    logger.warning(f"[Phase {phase_idx}] {killed_reason}. Force-killing...")
                    _kill_process_tree(proc.pid)
                    proc.wait()
                    break
    except KeyboardInterrupt:
        logger.warning(f"[Phase {phase_idx}] Ctrl+C. Killing subprocess...")
        _kill_process_tree(proc.pid)
        proc.wait()
        raise

    elapsed = time.time() - start_time
    returncode = proc.returncode
    if killed_reason is not None:
        returncode = 124  # killed here as hung: a failure, whatever the signal
    elif returncode == -signal.SIGKILL:
        returncode = 0  # the phase runner SIGKILLs itself after train() returns

    # Check if graduated early
    graduated = False
    if phase_log_dir is not None:
        markers = list(Path(phase_log_dir).glob("**/_graduated.marker"))
        graduated = len(markers) > 0

    if graduated:
        status = "graduated early"
    elif returncode == 0:
        status = "success"
    else:
        status = f"failed (rc={returncode}{', ' + killed_reason if killed_reason else ''})"
    logger.info(
        f"[Phase {phase_idx}: {phase.name}] Finished — {status} in {elapsed:.1f}s"
    )

    return {
        "phase_idx": phase_idx,
        "phase_name": phase.name,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "graduated_early": graduated,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }


def run_renew_pipeline(
    base_args: list[str],
    phases: list[RenewPhaseConfig],
    results_dir: str = "logs/training_pipeline_renew",
    shutdown_grace_seconds: float = 30.0,
    python: str = sys.executable,
    start_checkpoint: str | None = None,
    start_phase: int | None = None,
    num_gpus: int = 1,
) -> list[dict[str, Any]]:
    """Execute all renewed phases sequentially."""
    from holosoma.utils.experiment_paths import get_timestamp

    logger.info(f"Renewed pipeline: {len(phases)} phase(s)")

    results_base = Path(results_dir)
    results_base.mkdir(parents=True, exist_ok=True)

    run_timestamp = get_timestamp()
    run_dir = results_base / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_run_id = uuid.uuid4().hex[:8]
    logger.info(f"Shared wandb run ID: {wandb_run_id}")

    def _phase_dir(idx: int, phase: RenewPhaseConfig) -> Path:
        safe_name = phase.name.replace(" ", "_").replace(":", "").replace("/", "_")
        return run_dir / f"phase_{idx}_{safe_name}"

    results: list[dict[str, Any]] = []
    checkpoint_path: Path | None = None

    # Handle start checkpoint / phase skipping
    start_phase_idx = start_phase if start_phase is not None else 0
    if start_checkpoint is not None:
        checkpoint_path = Path(start_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Start checkpoint not found: {checkpoint_path}")
        if start_phase is None:
            start_phase_idx = 1
        logger.info(
            f"Skipping phases 0..{start_phase_idx - 1} — starting from "
            f"Phase {start_phase_idx} with checkpoint: {checkpoint_path}"
        )
    elif start_phase_idx > 0:
        raise ValueError(f"start_phase={start_phase_idx} requires a start_checkpoint.")

    for idx, phase in enumerate(phases):
        if idx < start_phase_idx:
            logger.info(f"Skipping Phase {idx} ({phase.name})")
            continue

        phase_log_dir = _phase_dir(idx, phase)
        phase_log_dir.mkdir(parents=True, exist_ok=True)

        result = run_renew_phase(
            idx, phase, base_args,
            checkpoint_path=checkpoint_path,
            phase_log_dir=phase_log_dir,
            wandb_run_id=wandb_run_id,
            python=python,
            shutdown_grace_seconds=shutdown_grace_seconds,
            num_gpus=num_gpus,
        )
        results.append(result)

        # Persist results incrementally
        results_file = run_dir / "pipeline_results.json"
        results_file.write_text(json.dumps(results, indent=2))

        if result["returncode"] != 0:
            logger.error(
                f"Phase '{phase.name}' failed with exit code {result['returncode']}. Aborting."
            )
            break

        # Discover checkpoint for next phase
        if idx < len(phases) - 1:
            try:
                checkpoint_path = discover_latest_checkpoint(phase_log_dir)
                logger.info(f"[Phase {idx}] Next checkpoint: {checkpoint_path}")
            except FileNotFoundError as exc:
                logger.error(f"Cannot find checkpoint: {exc}")
                break

    successes = sum(1 for r in results if r["returncode"] == 0)
    logger.info(f"Pipeline complete: {successes}/{len(results)} phases succeeded")
    logger.info(f"Results: {run_dir / 'pipeline_results.json'}")

    return results

