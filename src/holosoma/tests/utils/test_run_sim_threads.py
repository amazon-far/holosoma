"""Scope of the classic-MuJoCo CPU thread limit applied by ``holosoma.run_sim``.

The limit is a performance guard with no observable output, so what needs pinning is exactly WHEN
it fires: only the single-env CPU backend, never another backend, and never over an operator who
has already set a thread count in the environment.

Two deliberate choices, both because the alternative cannot fail:

* Cases carry a REAL ``SIMULATOR_REGISTRY`` preset. ``mujoco_backend`` lives on the shared
  ``SimulatorInitConfig`` and defaults to ``CLASSIC``, so every preset reports ``CLASSIC`` unless it
  overrides it -- a hand-built stand-in that simply omits the field cannot catch a gate keyed on it.
* The preset names and the environment-variable names are spelled out here rather than read from
  production. Parametrizing over the production constant makes a case vanish when the constant
  shrinks, which looks like a pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import cast

import pytest

import holosoma.config_values.simulator
from holosoma import run_sim
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_values.simulator import SIMULATOR_REGISTRY
from holosoma.run_sim import limit_cpu_math_threads
from holosoma.utils.safe_torch_import import torch

# Set explicitly rather than leaning on the root conftest's fallback: these cases are deselected by
# ``-m no_sim`` whenever pytest resolves its rootdir to src/holosoma, which puts the repo-root
# conftest outside confcutdir.
pytestmark = pytest.mark.no_sim

CLASSIC_PRESET = "mujoco"  # the only single-env CPU-physics backend
OTHER_PRESETS = ["isaacsim", "isaacsim3", "mjwarp"]
THREAD_LIMIT_VARIABLES = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]


@pytest.fixture(autouse=True)
def _no_inherited_thread_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every case from an environment with no thread limit, whatever the runner exported."""
    for name in THREAD_LIMIT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def requested(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record ``torch.set_num_threads`` calls instead of mutating the test process's pool."""
    calls: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    return calls


def _config(preset: str, device: str | None) -> RunSimConfig:
    """A real simulator preset plus the device. The rest of RunSimConfig the guard never reads."""
    return cast("RunSimConfig", SimpleNamespace(simulator=SIMULATOR_REGISTRY.get(preset), device=device))


def test_thread_limit_variables_match_production() -> None:
    # The parametrisation below deliberately spells the names out, which catches the tuple
    # SHRINKING (a case fails rather than vanishing). This catches it GROWING: a new variable with
    # no case would otherwise be unconstrained.
    assert list(run_sim._THREAD_LIMIT_VARIABLES) == THREAD_LIMIT_VARIABLES


def _in_tree_presets() -> set[str]:
    """The simulator presets holosoma itself declares, ignoring anything added from outside.

    ``SIMULATOR_REGISTRY`` is an entry-point-backed registry (and ``--import-file`` can add to it),
    so its key set is not closed: a downstream package that ships a simulator preset would appear
    here once something in the session triggers entry-point loading. The registry is a plain dict
    and keeps no provenance -- ``ingest`` only logs its ``source`` -- so in-tree membership is
    decided by object identity against the module that declares them.
    """
    declared = {
        id(value) for value in vars(holosoma.config_values.simulator).values() if isinstance(value, SimulatorConfig)
    }
    return {key for key, value in SIMULATOR_REGISTRY.items() if id(value) in declared}


def test_known_backends_are_the_ones_this_file_decides_for() -> None:
    # A new IN-TREE backend must fail here, so somebody chooses its pool policy instead of
    # inheriting whatever the shared mujoco_backend default happens to give it. Deliberately not
    # asserted over the whole registry: that would also fail for a third-party preset, which is
    # none of this file's business and would make the result depend on test order.
    assert _in_tree_presets() == {CLASSIC_PRESET, *OTHER_PRESETS}


@pytest.mark.parametrize("device", ["cpu", "cpu:0"])
def test_classic_mujoco_on_cpu_is_serialized(requested: list[int], device: str) -> None:
    # "cpu:0" is why the check is a prefix match rather than equality; Torch accepts the indexed
    # form and an exact comparison would silently skip the cap for it.
    limit_cpu_math_threads(_config(CLASSIC_PRESET, device))
    assert requested == [1]


@pytest.mark.parametrize("preset", OTHER_PRESETS)
def test_other_backends_keep_the_default_pool_even_on_cpu(requested: list[int], preset: str) -> None:
    # IsaacSim is the trap: it shares SimulatorInitConfig, so it reports mujoco_backend=CLASSIC,
    # and it falls back to cpu when CUDA is absent -- with num_envs > 1 under CPU PhysX.
    limit_cpu_math_threads(_config(preset, "cpu"))
    assert requested == []


@pytest.mark.parametrize("device", ["cuda:0", None])
def test_classic_mujoco_off_cpu_keeps_the_default_pool(requested: list[int], device: str | None) -> None:
    # Only the classic preset is worth varying the device over; the others are already excluded by
    # the backend half of the gate, so they would pass here however the device check behaved.
    limit_cpu_math_threads(_config(CLASSIC_PRESET, device))
    assert requested == []


@pytest.mark.parametrize("name", THREAD_LIMIT_VARIABLES)
def test_an_empty_environment_value_is_not_a_limit(
    requested: list[int], monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    # OpenMP and friends ignore an empty value, so it must not be mistaken for an operator's
    # choice; presence alone is the wrong test.
    monkeypatch.setenv(name, "")
    limit_cpu_math_threads(_config(CLASSIC_PRESET, "cpu"))
    assert requested == [1]


@pytest.mark.parametrize("name", THREAD_LIMIT_VARIABLES)
def test_explicit_environment_limit_wins(requested: list[int], monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    # Nothing in this repo sets these; the FAR-pi sim launcher sets them on the container child,
    # and an operator may export them by hand. Either way the process must not override them.
    monkeypatch.setenv(name, "8")
    limit_cpu_math_threads(_config(CLASSIC_PRESET, "cpu"))
    assert requested == []


# Calling the guard directly cannot show that run_simulation still calls it, nor where. These
# children drive the real entry point and read the pool back out of Torch at two points.
#
# The config is a REAL RunSimConfig from the production parser, not a stand-in: run_simulation
# resolves an unset device through ``dataclasses.replace``, which rejects a SimpleNamespace, so a
# stubbed config cannot reach the ``device is None`` branch at all -- and that branch is how
# ``python -m holosoma.run_sim simulator:mujoco`` actually runs.
_CHILD = textwrap.dedent(
    """
    import dataclasses

    import holosoma.run_sim as runner
    from holosoma.config_types.run_sim import RunSimConfig
    from holosoma.utils.config_registry import parse_config
    from holosoma.utils.safe_torch_import import torch

    class Env:
        sim = object()

        def close(self):
            pass

    def setup(_config, *, device):
        # Reported here too: the cap has to land before the environment is constructed, since that
        # is where the backend allocates its buffers. Checking only inside run() would let the call
        # slide past setup_simulation_environment unnoticed.
        print(f"AT_SETUP={torch.get_num_threads()}", flush=True)
        return Env(), device, None

    def initialize(self):
        pass

    def run(self):
        print(f"AT_RUN={torch.get_num_threads()}", flush=True)

    runner.setup_simulation_environment = setup
    runner.DirectSimulation.initialize = initialize
    runner.DirectSimulation.run = run

    config = parse_config(RunSimConfig, args=[f"simulator:{PRESET}"])
    if DEVICE is not None:
        config = dataclasses.replace(config, device=DEVICE)
    runner.run_simulation(config)
    """
)


def _child(source: str) -> dict[str, int]:
    """Run ``source`` in a child with every math-thread variable scrubbed, and parse its KEY=int lines.

    The environment is built here rather than inherited so that an operator's exported
    OMP_NUM_THREADS cannot shift one child's baseline relative to another's.
    """
    env = {k: v for k, v in os.environ.items() if k not in THREAD_LIMIT_VARIABLES}
    done = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=False, env=env)
    assert done.returncode == 0, f"child failed: {done.stderr[-2000:]}"
    reported = dict(
        line.split("=", 1) for line in done.stdout.splitlines() if line.partition("=")[0].isupper() and "=" in line
    )
    return {key: int(value) for key, value in reported.items()}


def _pools_after_run_simulation(preset: str, device: str | None) -> dict[str, int]:
    """The real ``run_simulation``, reporting the pool at env construction and at loop start."""
    observed = _child(f"PRESET = {preset!r}\nDEVICE = {device!r}\n{_CHILD}")
    assert set(observed) == {"AT_SETUP", "AT_RUN"}, observed
    return observed


@pytest.fixture(scope="session")
def untouched_pool() -> int:
    """Torch's own default pool in a fresh child, so the comparison holds on any core count."""
    return _child("from holosoma.utils.safe_torch_import import torch\nprint(f'POOL={torch.get_num_threads()}')")[
        "POOL"
    ]


# ``None`` is the default invocation: run_simulation resolves it to cpu for the classic backend.
@pytest.mark.parametrize("device", ["cpu", None])
def test_run_simulation_applies_the_limit_before_building_the_env(device: str | None) -> None:
    assert _pools_after_run_simulation(CLASSIC_PRESET, device) == {"AT_SETUP": 1, "AT_RUN": 1}


@pytest.mark.parametrize("preset", OTHER_PRESETS)
@pytest.mark.parametrize("device", ["cpu", None])
def test_run_simulation_leaves_other_backends_alone(preset: str, device: str | None, untouched_pool: int) -> None:
    # Guards the wiring from the other side: a cap applied unconditionally would show up here.
    # Comparing against Torch's own default rather than the literal 1 avoids a spurious failure on
    # a machine with a different core count -- but it only discriminates while that default is
    # above 1, so require it explicitly rather than let a cgroup-restricted runner pass everything.
    # With device=None and no CUDA present, IsaacSim resolves to cpu, which is precisely the
    # configuration that must stay uncapped.
    assert untouched_pool > 1, "cannot distinguish capped from uncapped when Torch defaults to 1"
    assert _pools_after_run_simulation(preset, device) == {
        "AT_SETUP": untouched_pool,
        "AT_RUN": untouched_pool,
    }
