"""Paths for data and generated artifacts owned by the retargeting package."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEMO_RESULTS_DIR = PACKAGE_ROOT / "demo_results"
DEMO_RESULTS_PARALLEL_DIR = PACKAGE_ROOT / "demo_results_parallel"
