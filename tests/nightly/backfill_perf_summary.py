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
import io
import itertools
import json
import math
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from os import getenv
from typing import Any, NamedTuple

import wandb

WANDB_ENTITY = getenv("WANDB_ENTITY", "amazon-far")
WANDB_PROJECT = "nightly-holosoma-runs"

# Per-iteration metrics written by LoggingHelper._logging_to_writer. All three are
# logged in the same wandb.log call, so scan_history can request them together.
PERF_METRICS = ("Perf/total_fps", "Perf/collection_time", "Perf/learning_time")

# Iteration zero pays for JIT compilation and CUDA graph capture, and is dramatically
# slower than the rest: one measured run reported 53s of learning time and a fifth of its
# steady-state fps on that row, then settled from the next one onward. Left in, it becomes
# the reported fps minimum and timing maximum for the whole run. Only the first row is
# dropped: trimming further discards the genuine early-training ramp, which is real
# behaviour worth seeing.
WARMUP_ROWS = 1

# `history` with a sample cap above a run's length returns every row in one request, in
# step order, identical to a paged scan -- measured at 0.7s against 18s for the same
# 10k-row run. The cap is a ceiling rather than a bucket count, so this is exact only
# while it exceeds the row count: asking for fewer samples than a run has rows silently
# downsamples and corrupts the extremes (a 100-sample read of one run reported a minimum
# of 55607 against a true 50651). Any run that reaches the cap is therefore re-read with
# the paged scan, which never samples.
HISTORY_SAMPLE_CAP = 200_000
HISTORY_PAGE_SIZE = 10_000

# Runs in these states have not finished logging. Aggregating one would store figures
# from a partial history and mark it processed, so the real numbers would never be
# recorded. They are left for a later pass instead.
UNFINISHED_STATES = frozenset({"running", "pending"})

# Summary keys are written under their own prefix rather than reusing the `Perf/`
# history namespace, so a run's history and its aggregates can't shadow each other.
SUMMARY_PREFIX = "perf/"
STATS = ("min", "mean", "max")

# Numeric run-start time and wall-clock duration, copied to explicit keys because
# wandb's own `_timestamp`/`_runtime` are internal and not reliably selectable as
# panel axes.
RUN_DATE_KEY = f"{SUMMARY_PREFIX}run_date"
RUN_DURATION_KEY = f"{SUMMARY_PREFIX}run_duration_s"

# Flat identifier for the nightly matrix cell a run belongs to, so the dashboard can
# group by it. Throughput is only comparable within a cell. It has to be a scalar
# summary key: run config is entirely nested dicts, and grouping a saved view by tags
# does not survive being written (see perf_workspace.py).
CELL_KEY = f"{SUMMARY_PREFIX}cell"

# Tag vocabulary written by nightly.py. Keep in sync with it and with run_summary.py.
KNOWN_SIMULATORS = {"isaacgym", "isaacsim", "mujoco"}
RESERVED_TAG_PREFIXES = ("nightly-", "gha-run-id-", "gpus-")
RESERVED_TAGS = KNOWN_SIMULATORS | {
    "singlegpu",
    "multigpu",
    "nightly_test_passed",
    "nightly_test_failed",
    "nightly_perf_warning",
}

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

    Skips the first `WARMUP_ROWS` rows, whose compilation cost is not representative of
    the run. Accumulates rather than materializing the history, because the fast-SAC
    configs log tens of thousands of iterations per run. Non-finite values are ignored.
    """
    totals: dict[str, list[float]] = {metric: [math.inf, -math.inf, 0.0, 0.0] for metric in PERF_METRICS}

    for row in itertools.islice(rows, WARMUP_ROWS, None):
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


def run_cell(run: Any) -> str:
    """Matrix cell label for a run, as "<experiment> | <simulator> | <gpus>".

    Derived from the tags nightly.py applies. Missing components are reported rather
    than dropped, so an untagged run groups somewhere visible instead of silently
    joining another cell.
    """
    tags = list(run.tags or [])

    gpus = next((t for t in tags if t.startswith("gpus-")), None)
    if gpus is None:
        gpus = "gpus-4" if "multigpu" in tags else "gpus-1" if "singlegpu" in tags else "gpus-?"

    simulator = next((t for t in tags if t in KNOWN_SIMULATORS), "sim-?")
    experiment = next(
        (t for t in tags if not t.startswith(RESERVED_TAG_PREFIXES) and t not in RESERVED_TAGS),
        "exp-?",
    )
    return f"{experiment} | {simulator} | {gpus}"


def read_history(run: Any) -> list[dict[str, Any]]:
    """Every history row carrying the Perf metrics, in step order.

    Takes the single-request path when the run fits under the sample cap, and falls back
    to the paged scan otherwise so the result is never sampled.
    """
    rows = run.history(keys=list(PERF_METRICS), samples=HISTORY_SAMPLE_CAP, pandas=False)
    if len(rows) < HISTORY_SAMPLE_CAP:
        return rows

    print(f"  run has at least {HISTORY_SAMPLE_CAP} rows; re-reading with a paged scan")
    return list(run.scan_history(keys=list(PERF_METRICS), page_size=HISTORY_PAGE_SIZE))


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


def write_summary(run: Any, updates: dict[str, Any]) -> None:
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
    aggregates = aggregate_history(read_history(run))
    if not aggregates:
        print("  no Perf/* history, marking processed")

    updates: dict[str, Any] = {BACKFILL_VERSION_KEY: BACKFILL_VERSION, CELL_KEY: run_cell(run)}
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


def backfill_cell_only(run: Any, dry_run: bool) -> str | None:
    """Writes just the matrix-cell label, returning it, or None if nothing was written.

    The label comes from a run's tags, so this needs no history read -- which lets runs
    aggregated before the label existed be topped up cheaply, rather than reprocessed.
    Reporting is left to the caller so each run's name and label stay on one line.
    """
    if CELL_KEY in (run.summary or {}):
        return None

    cell = run_cell(run)
    if dry_run:
        print(f"[dry-run] {run.name} -> {cell}")
        return None

    write_summary(run, {CELL_KEY: cell})
    return cell


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
    parser.add_argument(
        "--cells-only",
        action="store_true",
        help=f"Only add the missing {CELL_KEY} label, without reading history",
    )
    args = parser.parse_args()

    # A full pass takes a while and is usually watched through a redirected log, which
    # would otherwise block-buffer and show nothing for minutes at a time. Guarded
    # because stdout is not always a file wrapper, such as when it has been captured.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)

    api = wandb.Api(timeout=60)
    written = 0
    skipped = 0

    with contextlib.ExitStack() as stack:
        backup = stack.enter_context(open(args.backup, "a")) if args.backup else None

        for project_name in nightly_projects(api, args.project, args.all_projects):
            print(f"\n=== {WANDB_ENTITY}/{project_name}")
            # Have the backend skip runs that are already current. The in-code checks
            # catch them anyway, but only after fetching each run in full, which is
            # essentially the whole cost of a pass: without this, finding one night's
            # runs among a project's history means fetching and discarding all of it.
            # Matching on inequality rather than absence also re-selects runs left at
            # an older version. --force wants everything, so it filters nothing.
            filters: dict[str, dict[str, Any]] | None
            if args.force:
                filters = None
            elif args.cells_only:
                filters = {f"summary_metrics.{CELL_KEY}": {"$exists": False}}
            else:
                filters = {f"summary_metrics.{BACKFILL_VERSION_KEY}": {"$ne": BACKFILL_VERSION}}
            runs = api.runs(path=f"{WANDB_ENTITY}/{project_name}", filters=filters, order="-created_at")

            processed = 0
            for run in runs:
                if args.limit is not None and processed >= args.limit:
                    break
                processed += 1

                # One guard around everything this run touches. Reading a run's summary
                # loads it lazily and can raise on its own -- a project can list runs
                # that no longer load individually -- so a check placed outside the
                # guard lets a single unreadable run end the pass.
                try:
                    if run.state in UNFINISHED_STATES:
                        print(f"{run.name}: still {run.state}, leaving for a later pass")
                        skipped += 1
                        continue

                    if args.cells_only:
                        cell = backfill_cell_only(run, args.dry_run)
                        if cell is None:
                            skipped += 1
                        else:
                            print(f"{run.name} -> {cell}")
                            written += 1
                        continue

                    if not args.force and not needs_backfill(run):
                        skipped += 1
                        continue

                    print(f"{run.name} ({run.id})")
                    if backfill_run(run, args.dry_run, backup):
                        written += 1
                except Exception as e:  # keep going; one bad run shouldn't stop the pass
                    print(f"{run.name}: failed: {e}")

    print(f"\nUpdated {written} run(s), skipped {skipped} already-current run(s)")


if __name__ == "__main__":
    main()
