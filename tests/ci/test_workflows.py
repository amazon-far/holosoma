"""Read the workflow files and bind the values CI depends on."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.no_sim

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
GITHUB_DEFAULT_TIMEOUT_MINUTES = 360


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _simulator_timeout_minutes(multigpu: str) -> int:
    """Evaluate the simulator-tests timeout for one matrix value."""
    expression = _load("test-simulator-image.yaml")["jobs"]["simulator-tests"]["timeout-minutes"]
    match = re.fullmatch(r"\$\{\{\s*matrix\.multigpu == 'True' && (\d+) \|\| (\d+)\s*\}\}", str(expression))
    assert match, f"unrecognised timeout-minutes expression: {expression!r}"
    multi, single = (int(group) for group in match.groups())
    return multi if multigpu == "True" else single


def test_selected_images_run_the_simulator_suites_through_the_bounded_workflow():
    jobs = _load("test-images.yaml")["jobs"]
    callers = [job for job in jobs.values() if str(job.get("uses", "")).endswith("/test-simulator-image.yaml")]
    assert callers, "test-images.yaml no longer calls test-simulator-image.yaml"


@pytest.mark.parametrize(
    ("multigpu", "expected"),
    [
        # A green multi-gpu run is ~3 minutes, so a hang fails in 20 rather than 6 hours.
        ("True", 20),
        # The single-gpu Isaac Sim suite is ~40 minutes green.
        ("False", 90),
    ],
)
def test_simulator_tests_job_has_a_timeout(multigpu, expected):
    minutes = _simulator_timeout_minutes(multigpu)
    assert minutes == expected
    assert minutes < GITHUB_DEFAULT_TIMEOUT_MINUTES
