"""IsaacSim spawn-cfg invariants (boots a minimal headless app in a subprocess).

``object_spawner`` imports ``isaaclab.sim`` (needs ``carb``, only available once a
``SimulationApp`` runs), so this drives the ``object_spawner_cfg_assert.py`` harness in its own
process — which boots a minimal headless app, then asserts the cfg-layer invariants the spawner
refactor is responsible for (URDF/USD honor the same PhysicsConfig; the material pre-bake never
mutates the shared source asset). It does NOT build a full sim, so it's the cheapest IsaacSim
check in the suite.

Marked ``isaacsim`` so only the IsaacSim CI job (``-m isaacsim``) collects it;
``importorskip("isaaclab")`` makes a stray unfiltered run skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "object_spawner_cfg_assert.py"


def test_object_spawner_cfg_invariants(tmp_path):
    # IsaacSim app teardown can mask the exit code, so trust the result-file the harness writes
    # AFTER all checks pass (same convention as test_dr_matrix_isaacsim.py).
    result_file = tmp_path / "spawner_cfg_result.txt"
    run_harness(
        _HARNESS,
        "--result-file",
        str(result_file),
        label="spawner-cfg invariants",
        timeout=600,
        result_file=result_file,
    )
