#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "wandb",
# ]
# ///
"""Writes per-run performance aggregates into wandb run summaries.

The `Perf/*` metrics are logged once per training iteration, so they only exist as
per-run time series. wandb's cross-run panels read run summary and config, not
history, so plotting throughput across nights needs one value per run stored at the
run level. This script derives those values and writes them back.

Safe to re-run: runs that already carry the aggregates are skipped unless --force is
given, so it can backfill history once and then run after each nightly build.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from datetime import datetime, timezone
from os import getenv
from typing import Any, Iterable, NamedTuple

import wandb

WANDB_ENTITY = getenv("WANDB_ENTITY", "amazon-far")
WANDB_PROJECT = "nightly-holosoma-runs"

# Per-iteration metrics written by LoggingHelper._logging_to_writer. All three are
# logged in the same wandb.log call, so scan_history can request them together.
PERF_METRICS = ("Perf/total_fps", "Perf/collection_time", "Perf/learning_time")

# Summary keys are written under their own prefix rather than reusing the `Perf/`
# history namespace, so a run's history and its aggregates can't shadow each other.
SUMMARY_PREFIX = "perf/"
STATS = ("min", "mean", "max")

# Numeric run-start time and wall-clock duration, copied to explicit keys because
# wandb's own `_timestamp`/`_runtime` are internal and not reliably selectable as
# panel axes.
RUN_DATE_KEY = f"{SUMMARY_PREFIX}run_date"
RUN_DURATION_KEY = f"{SUMMARY_PREFIX}run_duration_s"

# Marks a run as processed. Keyed on a version rather than on the presence of the
# aggregates themselves, because a run that logged no Perf/* history (a crash, say)
# would otherwise look unprocessed forever and be rescanned on every pass. Bump this
# to reprocess every run after changing what gets written.
BACKFILL_VERSION_KEY = f"{SUMMARY_PREFIX}backfill_version"
BACKFILL_VERSION = 1


class Aggregate(NamedTuple):
    """Min/mean/max of one metric over a single run."""

    minimum: float
    mean: float
    maximum: float


def summary_key(metric: str, stat: str) -> str:
    """Summary key for one metric/stat pair, e.g. `perf/total_fps_mean`."""
    return f"{SUMMARY_PREFIX}{metric.split('/')[-1]}_{stat}"


def aggregate_keys() -> list[str]:
    """The per-metric aggregate keys this script writes when history is available."""
    return [summary_key(metric, stat) for metric in PERF_METRICS for stat in STATS]


def aggregate_history(rows: Iterable[dict[str, Any]]) -> dict[str, Aggregate]:
    """Reduces history rows to per-metric min/mean/max in a single pass.

    Accumulates rather than materializing the history, because the fast-SAC configs
    log tens of thousands of iterations per run. Non-finite values are ignored.
    """
    totals: dict[str, list[float]] = {metric: [math.inf, -math.inf, 0.0, 0.0] for metric in PERF_METRICS}

    for row in rows:
        for metric in PERF_METRICS:
            value = row.get(metric)
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            acc = totals[metric]
            acc[0] = min(acc[0], float(value))
            acc[1] = max(acc[1], float(value))
            acc[2] += float(value)
            acc[3] += 1

    aggregates: dict[str, Aggregate] = {}
    for metric, (lo, hi, total, count) in totals.items():
        if count:
            aggregates[metric] = Aggregate(minimum=lo, mean=total / count, maximum=hi)
    return aggregates


def run_start_epoch(run: Any) -> float | None:
    """Unix timestamp of a run's creation, or None when wandb reports no value.

    wandb returns `created_at` as a UTC ISO-8601 string, usually without an offset.
    """
    raw = getattr(run, "created_at", None)
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        print(f"  could not parse created_at {raw!r}")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def needs_backfill(run: Any) -> bool:
    """Whether a run has yet to be processed at the current version."""
    return (run.summary or {}).get(BACKFILL_VERSION_KEY) != BACKFILL_VERSION


def write_summary(run: Any, updates: dict[str, float | int]) -> None:
    """Persists summary keys without touching the rest of the run.

    Uses `run.summary.update()` rather than `run.update()`: the latter also re-upserts
    tags, config, display name, notes, group and job type from the in-memory run, which
    risks overwriting metadata this script has no business changing.
    `summary.update()` issues a summary-only mutation.
    """
    for key, value in updates.items():
        run.summary[key] = value
    run.summary.update()


def backfill_run(run: Any, dry_run: bool, backup: Any = None) -> bool:
    """Computes and stores one run's aggregates. Returns whether anything was written.

    Runs with no usable history are still marked processed, so they aren't rescanned
    on every subsequent pass. When `backup` is an open file, each run's prior summary
    is appended to it as JSON before anything is written.
    """
    aggregates = aggregate_history(run.scan_history(keys=list(PERF_METRICS)))
    if not aggregates:
        print("  no Perf/* history, marking processed")

    updates: dict[str, float | int] = {BACKFILL_VERSION_KEY: BACKFILL_VERSION}
    for metric, aggregate in aggregates.items():
        updates[summary_key(metric, "min")] = aggregate.minimum
        updates[summary_key(metric, "mean")] = aggregate.mean
        updates[summary_key(metric, "max")] = aggregate.maximum

    start_epoch = run_start_epoch(run)
    if start_epoch is not None:
        updates[RUN_DATE_KEY] = start_epoch

    runtime = (run.summary or {}).get("_runtime")
    if isinstance(runtime, (int, float)) and math.isfinite(runtime):
        updates[RUN_DURATION_KEY] = float(runtime)

    fps_mean = updates.get(summary_key("Perf/total_fps", "mean"))
    print(f"  fps mean {fps_mean:.0f}" if fps_mean is not None else "  no fps data")

    if dry_run:
        print(f"  [dry-run] would write {len(updates)} keys")
        return False

    if backup is not None:
        backup.write(json.dumps({"id": run.id, "path": str(run.path), "summary": dict(run.summary or {})}) + "\n")
        backup.flush()

    write_summary(run, updates)
    print(f"  wrote {len(updates)} keys")
    return True


def nightly_projects(api: wandb.Api, project: str | None, all_projects: bool) -> list[str]:
    """Project names to process.

    Defaults to the project the nightly build writes to. The older per-experiment
    `nightly-<exp>` projects are only included with `all_projects`, since scanning
    their history is slow and they no longer receive runs.
    """
    if project:
        return [project]
    if all_projects:
        return [p.name for p in api.projects(WANDB_ENTITY) if p.name.startswith("nightly")]
    return [WANDB_PROJECT]


def main() -> None:
    parser = argparse.ArgumentParser(
        "backfill_perf_summary",
        description="Writes per-run Perf/* min/mean/max into wandb run summaries",
    )
    parser.add_argument("--project", default=None, help=f"Single project to process (default: {WANDB_PROJECT})")
    parser.add_argument(
        "--all-projects", action="store_true", help="Also process the legacy per-experiment nightly projects"
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many runs per project")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing")
    parser.add_argument("--force", action="store_true", help="Recompute runs that are already marked processed")
    parser.add_argument(
        "--backup", default=None, help="Append each run's prior summary to this file as JSON before writing"
    )
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    written = 0
    skipped = 0

    with contextlib.ExitStack() as stack:
        backup = stack.enter_context(open(args.backup, "a")) if args.backup else None

        for project_name in nightly_projects(api, args.project, args.all_projects):
            print(f"\n=== {WANDB_ENTITY}/{project_name}")
            runs = api.runs(path=f"{WANDB_ENTITY}/{project_name}", order="-created_at")

            processed = 0
            for run in runs:
                if args.limit is not None and processed >= args.limit:
                    break
                processed += 1

                if not args.force and not needs_backfill(run):
                    skipped += 1
                    continue

                print(f"{run.name} ({run.id})")
                try:
                    if backfill_run(run, args.dry_run, backup):
                        written += 1
                except Exception as e:  # keep going; one bad run shouldn't stop the pass
                    print(f"  failed: {e}")

    print(f"\nUpdated {written} run(s), skipped {skipped} already-current run(s)")


if __name__ == "__main__":
    main()
