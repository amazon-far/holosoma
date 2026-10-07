"""IsaacSim render-cadence guard: a camera run must refuse an interval that beats the control step.

IsaacSim's cameras are read out of the periodic render pass, which draws when the backend's substep
counter reaches ``render_interval``. A driving loop closes its frame on the control step's last
substep, so an interval that does not divide the decimation puts the draw and the frame's state on
different substeps and a consumer pairs an image with a pose from another one. ``IsaacSim.__init__``
rejects that pairing, and does so before building the ``SimulationContext`` so a rejected config
leaves no process-global context behind. Both halves are asserted here.

The rule itself is arithmetic; what needs a live backend is the wiring — that the constructor applies
it at all, and applies it early enough. So this file is both the pytest wrapper and the harness it
runs: the checks live in ``_main`` and execute in a subprocess (``python <this file>``), because
importing the backend launches the app and ``SimulationContext`` is a process singleton. Nothing here
builds a scene — the guard raises during construction, before any of that.

Run the harness half directly with:
    python tests/simulators/isaacsim/test_render_cadence_guard.py [--result-file /tmp/r.txt]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# Shadow-fix: keep this file's dir from hiding the real isaacsim package.
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).parent.resolve():
    sys.path.pop(0)

BEATING_INTERVAL = 3
"""Does not divide the g1 preset's control decimation of 4, so the frame's last draw precedes it."""


def _run_config(render_interval: int):
    """A one-env headless IsaacSim config with one mounted camera and the given render interval."""
    from tests.simulators._camera_presets import front_cam
    from tests.simulators._sim_harness import build_run_sim_config

    config = build_run_sim_config("isaacsim", "empty", "g1-29dof", "terrain_locomotion_plane", sensors=front_cam)
    simulator = dataclasses.replace(
        config.simulator,
        config=dataclasses.replace(
            config.simulator.config,
            sim=dataclasses.replace(config.simulator.config.sim, render_interval=render_interval),
        ),
    )
    return dataclasses.replace(
        config,
        simulator=simulator,
        device="cuda:0",
        training=dataclasses.replace(config.training, num_envs=1, headless=True),
    )


def _construct(config) -> None:
    """Construct the IsaacSim backend for ``config``, launching the app but building no scene.

    Deliberately not ``setup_simulation_environment``: its cleanup calls the Isaac app's ``close()``,
    which exits the process itself rather than returning, so a construction error there never reaches
    the caller and the run ends silently with status 0. Launching the app and calling the backend
    directly keeps the exception observable, which is the whole point of this check.
    """
    from holosoma.config_types.full_sim import FullSimConfig
    from holosoma.managers.terrain import TerrainManager
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import setup_isaaclab_launcher, setup_simulator_imports

    setup_simulator_imports(config)
    setup_isaaclab_launcher(config, device=config.device)

    full_config = FullSimConfig(
        simulator=config.simulator.config,
        robot=config.robot,
        scene=config.scene,
        sensors=dict(config.sensor),
        training=config.training,
        logger=config.logger,
        plugin=config.plugin,
        experiment_dir=None,
    )

    class _EnvProxy:
        num_envs = config.training.num_envs
        device = config.device

    terrain_manager = TerrainManager(config.terrain, env=_EnvProxy(), device=config.device)
    get_class(config.simulator._target_)(full_config, terrain_manager, config.device)


def _main() -> int:
    """The harness: construct with a beating cadence and assert the constructor refuses it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", default=None, help="write 'OK' here after PASS (teardown-robust)")
    args = parser.parse_args()

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    config = _run_config(BEATING_INTERVAL)
    sim_config = config.simulator.config.sim
    assert sim_config.render_interval_steps == BEATING_INTERVAL
    decimation = sim_config.control_decimation_steps
    assert decimation % BEATING_INTERVAL, f"decimation {decimation} is divisible; pick another interval"

    error: Exception | None = None
    try:
        _construct(config)
    except Exception as raised:  # a failure of another kind is a FAIL, not a pass
        error = raised

    # Both halves matter: that construction failed at all, and that it failed on THIS rule. Without
    # the second, any later construction error — a missing asset, a bad device — would read as a pass.
    check("construction-refused", error is not None, f"raised {type(error).__name__}")
    check(
        "refusal-is-the-cadence-rule",
        isinstance(error, ValueError) and "must divide control_decimation" in str(error),
        str(error)[:200],
    )

    # The guard runs before the context is built, so nothing global survives the refusal — which is
    # what lets a corrected config be constructed in this same process.
    from isaaclab.sim import SimulationContext

    check("no-simulation-context-leaked", SimulationContext.instance() is None)

    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("ALL PASS")
    if args.result_file:
        Path(args.result_file).write_text("OK\n")
    return 0


if __name__ == "__main__":
    from tests.simulators._sim_harness import run_and_hard_exit

    run_and_hard_exit(_main)


import pytest  # noqa: E402  (the harness half above must not depend on pytest)

pytestmark = pytest.mark.isaacsim


def test_camera_run_refuses_a_beating_render_cadence(tmp_path):
    """Drive the harness above in its own process and require every check to pass."""
    pytest.importorskip("isaaclab")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("IsaacSim requires a CUDA device")

    from tests.simulators._run_harness import run_harness

    result_file = tmp_path / "render_cadence_isaacsim.txt"
    run_harness(
        Path(__file__).resolve(),
        "--result-file",
        str(result_file),
        label="isaacsim/render-cadence-guard",
        timeout=400,
        result_file=result_file,
    )
