"""The test jobs in .github/workflows/pytest.yaml must be able to fail the workflow run (pure, no simulator).

``continue-on-error: true`` at JOB level makes a job's failure invisible to the run that contains it:
the job's own check-run concludes ``failure`` while the ``Pytests`` run concludes ``success``. Both
simulator jobs (``tests``, ``e2e-tests``) carried it from 75691e1, a runner-selection PR, so every
isaacgym / isaacsim / mujoco / requires_inference test in this repo ran where a red result could not
fail the run. Observed in production, not inferred: PR #170 merged while three sim job check-runs
from run 29469569515 had already concluded ``failure`` and that same ``Pytests`` run read ``success``.

Read this file with a YAML parser rather than by eye. The neighbouring ``fail-fast: false`` under
``strategy`` is about the matrix and invites misreading the indentation, while ``continue-on-error``
sits at job level and is what decouples the job's result from the run's.

NOT COVERED HERE: a step-level ``continue-on-error`` reaches the same end (the step fails, the job
succeeds), and this asserts only the job-level flag. Nor does it say anything about which checks are
required to merge -- that lives in branch protection, outside this repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "pytest.yaml"


def _jobs() -> dict:
    """Return pytest.yaml's ``jobs`` mapping, refusing to answer off an empty or wrong-file parse."""
    assert WORKFLOW.is_file(), f"no workflow at {WORKFLOW} -- repo layout moved, so nothing was checked"
    text = WORKFLOW.read_text()
    jobs = (yaml.safe_load(text) or {}).get("jobs")
    # Both assertions are vacuity refusals: an empty parse, or a file that no longer runs the
    # suite, would satisfy the check below by construction rather than by being correct.
    assert isinstance(jobs, dict) and jobs, f"no jobs parsed out of {WORKFLOW}"
    assert "pytest" in text, f"{WORKFLOW} no longer runs pytest -- is this still the test workflow?"
    return jobs


@pytest.mark.no_sim
def test_no_pytest_job_swallows_its_own_failure():
    swallowing = sorted(name for name, job in _jobs().items() if job.get("continue-on-error", False) is not False)
    assert not swallowing, (
        f"job-level 'continue-on-error' on {WORKFLOW.name} job(s): {', '.join(swallowing)}. Those jobs' "
        "check-runs go red while the workflow run concludes success, so their tests cannot fail the run. "
        "Remove the flag, or set it to false."
    )
