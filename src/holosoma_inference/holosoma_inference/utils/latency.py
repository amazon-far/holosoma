"""Latency measurement utility."""

from __future__ import annotations

import atexit
import statistics
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass

# Reporting order; also the column order of the per-cycle CSV.
STAGE_ORDER = (
    "fresh_state_wait",
    "read_state",
    "preprocessing",
    "inference",
    "postprocessing",
    "action_pub",
    "total",
)

_CSV_BUFFER_BYTES = 1 << 16


@dataclass
class LatencyStats:
    """Statistics for a stage over multiple measurements."""

    stage: str
    count: int = 0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0


class LatencyTracker:
    """Minimal latency measurement system."""

    def __init__(self, window_size: int = 50, csv_path: str | None = None):
        self.window_size = window_size
        self.measurements: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.current_cycle: dict[str, float] = {}
        self.cycle_start_time: float | None = None
        self.last_cycle_start_time: float | None = None
        self.fps_measurements: deque = deque(maxlen=window_size)
        self.cycle_index = 0
        # Block-buffered so the per-cycle write is a memcpy, not a syscall.
        self._csv = open(csv_path, "w", buffering=_CSV_BUFFER_BYTES) if csv_path else None  # noqa: SIM115 - lives for the run
        if self._csv is not None:
            self._csv.write("iter," + ",".join(STAGE_ORDER) + "\n")
            atexit.register(self._csv.close)

    @contextmanager
    def measure(self, stage: str):
        """Context manager for measuring a single stage."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            self.measurements[stage].append(duration_ms)
            self.current_cycle[stage] = duration_ms

    def start_cycle(self):
        """Start a new measurement cycle."""
        current_time = time.perf_counter()

        # Calculate FPS if we have a previous cycle
        if self.last_cycle_start_time is not None:
            cycle_duration = current_time - self.last_cycle_start_time
            fps = 1.0 / cycle_duration if cycle_duration > 0 else 0.0
            self.fps_measurements.append(fps)

        self.last_cycle_start_time = current_time
        self.cycle_start_time = current_time
        self.current_cycle.clear()

    def end_cycle(self) -> dict[str, float]:
        """End current cycle and return measurements."""
        if self.cycle_start_time:
            total_time = (time.perf_counter() - self.cycle_start_time) * 1000
            self.current_cycle["total"] = total_time
            self.measurements["total"].append(total_time)

        if self._csv is not None:
            self._write_csv_row()

        return self.current_cycle.copy()

    def _write_csv_row(self):
        """Append this cycle's per-stage durations to the CSV."""
        row = ",".join(f"{self.current_cycle.get(stage, float('nan')):.4f}" for stage in STAGE_ORDER)
        self._csv.write(f"{self.cycle_index},{row}\n")
        self.cycle_index += 1

    def get_stats(self, stages: list[str] | None = None) -> dict[str, LatencyStats]:
        """Get statistics for specified stages or all stages."""
        if stages is None:
            stages = list(self.measurements.keys())

        stats = {}
        for stage in stages:
            if self.measurements.get(stage):
                data = list(self.measurements[stage])
                stats[stage] = LatencyStats(
                    stage=stage,
                    count=len(data),
                    mean_ms=statistics.mean(data),
                    std_ms=statistics.stdev(data) if len(data) > 1 else 0.0,
                    min_ms=min(data),
                    max_ms=max(data),
                )
            else:
                raise ValueError(f"No {stage=} in {stages=}!")
        return stats

    def get_stats_str(self) -> str:
        """Get formatted one-line latency statistics string."""
        stats = self.get_stats()
        if not stats:
            return ""

        # Create one-line latency report
        latency_parts = []
        for stage in STAGE_ORDER:
            if stage in stats:
                stat = stats[stage]
                latency_parts.append(f"{stage}: {stat.mean_ms:.3f}±{stat.std_ms:.3f} max {stat.max_ms:.3f}ms")

        if latency_parts:
            return " | ".join(latency_parts)
        return ""

    def get_fps(self) -> float:
        """Get current FPS (frames per second) based on cycle timing."""
        if not self.fps_measurements:
            return 0.0
        return statistics.mean(self.fps_measurements)

    def reset(self):
        """Reset all measurements."""
        self.measurements.clear()
        self.current_cycle.clear()
