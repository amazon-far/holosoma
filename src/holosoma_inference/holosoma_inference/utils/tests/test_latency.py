#!/usr/bin/env python3
"""Unit tests for the latency tracking implementation."""

import csv
import tempfile
import time
import unittest
from pathlib import Path

from holosoma_inference.utils.latency import STAGE_ORDER, LatencyStats, LatencyTracker


class TestLatencyTracker(unittest.TestCase):
    """Test cases for LatencyTracker functionality."""

    def test_latency_tracker_basic(self):
        """Test basic latency tracker functionality."""
        tracker = LatencyTracker(window_size=10)

        # Test measurement context manager
        with tracker.measure("test_stage"):
            time.sleep(0.001)  # 1ms sleep

        # Check that measurement was recorded
        assert "test_stage" in tracker.measurements
        assert len(tracker.measurements["test_stage"]) == 1
        assert tracker.measurements["test_stage"][0] >= 1.0  # Should be at least 1ms

    def test_latency_tracker_cycle(self):
        """Test cycle measurement functionality."""
        tracker = LatencyTracker()

        tracker.start_cycle()

        with tracker.measure("stage1"):
            time.sleep(0.001)

        with tracker.measure("stage2"):
            time.sleep(0.002)

        cycle_results = tracker.end_cycle()

        # Check cycle results
        assert "stage1" in cycle_results
        assert "stage2" in cycle_results
        assert "total" in cycle_results
        assert cycle_results["stage1"] >= 1.0
        assert cycle_results["stage2"] >= 2.0
        assert cycle_results["total"] >= 3.0

    def test_latency_stats(self):
        """Test statistics calculation."""
        tracker = LatencyTracker()

        # Add multiple measurements
        for i in range(5):
            with tracker.measure("test_stage"):
                time.sleep(0.001 * (i + 1))  # Variable sleep times

        stats = tracker.get_stats(["test_stage"])

        assert "test_stage" in stats
        stat = stats["test_stage"]
        assert isinstance(stat, LatencyStats)
        assert stat.count == 5
        assert stat.mean_ms > 0
        assert stat.min_ms > 0
        assert stat.max_ms > stat.min_ms

    def test_fps_tracking(self):
        """Test FPS tracking functionality."""
        tracker = LatencyTracker()

        # Simulate multiple cycles
        for _ in range(3):
            tracker.start_cycle()
            time.sleep(0.01)  # ~100 FPS
            tracker.end_cycle()

        fps = tracker.get_fps()
        assert fps > 0
        # Should be roughly around 100 FPS (allowing for timing variations)
        assert fps < 200  # Upper bound check

    def test_get_stats_str(self):
        """Test formatted statistics string output."""
        tracker = LatencyTracker()

        # Add some measurements
        with tracker.measure("read_state"):
            time.sleep(0.001)
        with tracker.measure("inference"):
            time.sleep(0.002)

        stats_str = tracker.get_stats_str()

        # Should contain stage names and measurements
        assert "read_state:" in stats_str
        assert "inference:" in stats_str
        assert "ms" in stats_str
        assert "|" in stats_str  # Pipe separator

    def test_get_stats_str_reports_max(self):
        """The one-line report carries a max, not just mean±std."""
        tracker = LatencyTracker()

        for sleep_s in (0.001, 0.005):
            with tracker.measure("inference"):
                time.sleep(sleep_s)

        stats_str = tracker.get_stats_str()
        stat = tracker.get_stats(["inference"])["inference"]

        assert f"max {stat.max_ms:.3f}ms" in stats_str

    def test_csv_dump(self):
        """csv_path writes one row per cycle with a column per stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "cycles.csv"
            tracker = LatencyTracker(csv_path=str(csv_path))

            for _ in range(3):
                tracker.start_cycle()
                with tracker.measure("inference"):
                    time.sleep(0.001)
                tracker.end_cycle()
            tracker._csv.flush()

            with csv_path.open() as handle:
                rows = list(csv.DictReader(handle))

        assert [row["iter"] for row in rows] == ["0", "1", "2"]
        assert list(rows[0]) == ["iter", *STAGE_ORDER]
        assert float(rows[0]["inference"]) >= 1.0
        # Stages that did not run this cycle are nan, not zero.
        assert rows[0]["read_state"] == "nan"

    def test_no_csv_by_default(self):
        """No csv_path means no file handle and no per-cycle write."""
        tracker = LatencyTracker()
        tracker.start_cycle()
        tracker.end_cycle()

        assert tracker._csv is None

    def test_reset_functionality(self):
        """Test reset functionality."""
        tracker = LatencyTracker()

        # Add some measurements
        with tracker.measure("test_stage"):
            time.sleep(0.001)

        # Verify measurements exist
        assert "test_stage" in tracker.measurements

        # Reset and verify clean state
        tracker.reset()
        assert len(tracker.measurements) == 0
        assert len(tracker.current_cycle) == 0


if __name__ == "__main__":
    unittest.main()
