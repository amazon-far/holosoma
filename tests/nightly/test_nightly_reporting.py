"""Unit tests for the pure reporting helpers in run_summary.py.

The nightly scripts are standalone `uv run --script` files rather than a package, so the
module under test is loaded by path. Only helpers that need no wandb access are covered;
the fetch is stubbed where a caller would make one.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.no_sim

ISAACSIM_CELL = "g1_29dof_manager | isaacsim | gpus-1"
MJWARP_CELL = "g1_29dof_manager | mjwarp | gpus-1"


def _load_run_summary() -> Any:
    path = Path(__file__).parent / "run_summary.py"
    spec = importlib.util.spec_from_file_location("nightly_run_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rs() -> Any:
    return _load_run_summary()


def _series(count: int, value: float, spacing_days: float = 1.0) -> list[tuple[datetime, float]]:
    """`count` points newest first, one every `spacing_days`, alternating around `value`.

    The alternation gives the series a non-zero spread, without which the 1d comparison has
    no scale to judge against and every row reports flat.
    """
    now = datetime.now(timezone.utc)
    return [(now - timedelta(days=i * spacing_days), value + (i % 2) * 0.1) for i in range(count)]


# --- tag parsing -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["gpus-1"], "1gpu"),
        (["gpus-4"], "4gpu"),
        (["multigpu"], "4gpu"),  # legacy runs predate the explicit tag
        (["singlegpu"], "1gpu"),
        (["gpus-4", "singlegpu"], "4gpu"),  # explicit tag wins over the legacy one
        ([], "unknown-gpu"),
    ],
)
def test_gpus_from_tags(rs: Any, tags: list[str], expected: str) -> None:
    assert rs._gpus_from_tags(tags) == expected


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["g1_29dof_manager", "isaacsim", "gpus-1"], "isaacsim"),
        (["g1_29dof_manager", "mjwarp", "gpus-1"], "mjwarp"),
        (["g1_29dof_manager", "mujoco", "gpus-1"], "mujoco"),
        (["g1_29dof_manager", "isaacgym", "gpus-1"], "isaacgym"),  # archived runs only
        (["g1_29dof_manager", "gpus-1"], "unknown-sim"),
    ],
)
def test_simulator_from_tags(rs: Any, tags: list[str], expected: str) -> None:
    assert rs._simulator_from_tags(tags) == expected


class _StubRun:
    """Enough of a wandb run for the experiment-name helper."""

    def __init__(self, tags: list[str], url: str) -> None:
        self.tags = tags
        self.url = url


@pytest.mark.parametrize(
    "name",
    [
        "nightly-g1_29dof_manager-mjwarp-20261022_220755",
        "nightly-g1_29dof_manager-isaacsim-multigpu-20261022_220755",
        # Names without a simulator, still present on older runs.
        "nightly-g1_29dof_manager-20261022_220755",
        "nightly-g1_29dof_manager-multigpu-20261022_220755",
    ],
)
def test_exp_name_falls_back_to_the_url_for_untagged_runs(rs: Any, name: str) -> None:
    """Untagged runs are read from the URL, for both the current and the older name shapes.

    The URL carries no path prefix: the fallback indexes a fixed path segment, so a deployment
    serving the app under one does not reach this path.
    """
    run = _StubRun(tags=[], url=f"https://host/entity/nightly-holosoma-runs/runs/{name}")
    assert rs._run_exp_name(run) == "g1_29dof_manager"


def test_exp_name_prefers_the_tag_over_the_url(rs: Any) -> None:
    run = _StubRun(
        tags=["g1_29dof_manager", "mjwarp", "gpus-1"],
        url="https://host/wandb/entity/nightly-holosoma-runs/runs/nightly-something_else-mjwarp-1",
    )
    assert rs._run_exp_name(run) == "g1_29dof_manager"


def test_every_tracked_cell_names_a_known_simulator(rs: Any) -> None:
    """A cell label whose simulator has drifted would silently never match a run."""
    for cell in rs.TREND_CELLS:
        simulator = cell.split("|")[1].strip()
        assert simulator in rs.KNOWN_SIMULATORS, cell


# --- per-cell view links -----------------------------------------------------------


def test_trend_view_url_is_per_cell(rs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERF_TREND_VIEW_URL_G1_29DOF_MANAGER_ISAACSIM_GPUS_1", "url-isaac")
    monkeypatch.setenv("PERF_TREND_VIEW_URL_G1_29DOF_MANAGER_MJWARP_GPUS_1", "url-mjwarp")

    assert rs._trend_view_url(ISAACSIM_CELL) == "url-isaac"
    assert rs._trend_view_url(MJWARP_CELL) == "url-mjwarp"


def test_trend_view_url_empty_when_unset(rs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERF_TREND_VIEW_URL_G1_29DOF_MANAGER_ISAACSIM_GPUS_1", raising=False)
    # The un-suffixed variable names no particular cell and must not be used for one.
    monkeypatch.setenv("PERF_TREND_VIEW_URL", "url-bare")

    assert rs._trend_view_url(ISAACSIM_CELL) == ""


# --- trend report ------------------------------------------------------------------


def test_report_is_none_when_no_cell_has_history(rs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rs, "_fetch_trend_series", lambda _cell: ([], None))
    assert rs.format_trend_report() is None


def test_report_skips_the_cell_without_history(rs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def fetch(cell: str) -> tuple[list[tuple[datetime, float]], str | None]:
        return (_series(8, 3.3), "project-url") if cell == ISAACSIM_CELL else ([], None)

    monkeypatch.setattr(rs, "_fetch_trend_series", fetch)
    report = rs.format_trend_report()

    assert report is not None
    assert ISAACSIM_CELL in report
    assert MJWARP_CELL not in report


def test_each_block_carries_only_its_own_view_link(rs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rs, "_fetch_trend_series", lambda _cell: (_series(8, 3.3), "project-url"))
    monkeypatch.setenv("PERF_TREND_VIEW_URL_G1_29DOF_MANAGER_ISAACSIM_GPUS_1", "url-isaac")
    monkeypatch.setenv("PERF_TREND_VIEW_URL_G1_29DOF_MANAGER_MJWARP_GPUS_1", "url-mjwarp")

    report = rs.format_trend_report()
    assert report is not None

    blocks = report.split("\n\n")
    assert len(blocks) == len(rs.TREND_CELLS)

    isaac_block = next(b for b in blocks if ISAACSIM_CELL in b)
    mjwarp_block = next(b for b in blocks if MJWARP_CELL in b)
    assert "url-isaac" in isaac_block and "url-mjwarp" not in isaac_block
    assert "url-mjwarp" in mjwarp_block and "url-isaac" not in mjwarp_block


def test_block_falls_back_to_project_link(rs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rs, "_fetch_trend_series", lambda _cell: (_series(8, 3.3), "project-url"))
    for cell in rs.TREND_CELLS:
        monkeypatch.delenv(f"PERF_TREND_VIEW_URL_{cell.upper().replace(' | ', '_').replace('-', '_')}", raising=False)

    report = rs.format_trend_report()
    assert report is not None
    assert "project-url" in report


def test_blocks_are_separated_by_a_blank_line(rs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without it a cell's heading renders directly under the previous cell's link."""
    monkeypatch.setattr(rs, "_fetch_trend_series", lambda _cell: (_series(8, 3.3), "project-url"))

    report = rs.format_trend_report()
    assert report is not None
    assert f"\n\n⏱️ {rs.TREND_METRIC_LABEL} · {MJWARP_CELL}" in report


def test_block_reports_all_three_timescales_earlier_first(rs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rs, "_fetch_trend_series", lambda _cell: (_series(80, 3.3), "project-url"))

    block = rs._format_trend_block(ISAACSIM_CELL)
    assert block is not None
    for label in ("1d:", "7d:", "30d:"):
        assert label in block
    # The arrow reads earlier -> later, so every row carries one.
    assert block.count("→") == 3


def test_block_is_none_rather_than_raising_on_a_fetch_failure(rs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The trend is a nicety; it must never stop the summary posting."""

    def boom(cell: str) -> tuple[list[tuple[datetime, float]], str | None]:
        raise RuntimeError("wandb unavailable")

    monkeypatch.setattr(rs, "_fetch_trend_series", boom)
    assert rs._format_trend_block(ISAACSIM_CELL) is None
    assert rs.format_trend_report() is None
