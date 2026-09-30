"""IsaacSim actuator-grouping equivalence (drives a headless harness in a subprocess)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "actuator_group_equivalence_assert.py"


def test_actuator_group_equivalence(tmp_path):
    result_file = tmp_path / "actuator_group_result.txt"
    run_harness(
        _HARNESS,
        "--result-file",
        str(result_file),
        label="actuator-group equivalence",
        timeout=600,
        result_file=result_file,
    )
