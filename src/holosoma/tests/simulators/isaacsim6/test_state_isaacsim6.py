"""Headless Isaac Lab 3.0 state, scene and playback regressions.

Each harness runs in a separate process. Both PhysX and Newton exercise the same
state contract and multi-environment scenarios; completion markers guard against
Kit shutdown masking an early exit."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim6

pytest.importorskip("isaaclab")
pytest.importorskip("isaaclab_physx")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Isaac Sim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS_DIR = Path(__file__).resolve().parents[1]

BACKENDS = ["isaacsim6", "isaacsim6-newton"]
"""The constructible 6.2 presets. ``isaacsim6-ovphysx`` is deliberately absent.

OvPhysX is Isaac Lab 3.0's *Kit-free* PhysX runtime and holosoma always launches Kit, so
``capabilities.validate`` rejects it at startup -- a test here would assert the rejection, not the
capability, and that is already covered by ``test_capabilities.py`` without a GPU.
"""


@pytest.mark.parametrize("simulator", BACKENDS)
@pytest.mark.parametrize("num_envs", ["1", "4"])
def test_static_move(simulator: str, num_envs: str) -> None:
    """A box rests on a static pillar, the pillar teleports, the box then falls.

    Covers three things at once that are easy to get wrong independently: that a static body is
    solid at its spawned pose, that the kinematic write lands on the live backend, and that
    ``get_actor_states`` reports the relocated pose. Run at 4 envs as well as 1 because a bare
    ``InteractiveSceneCfg`` in Isaac Lab 3.0 replicates *nothing* while still reporting correct
    ``env_origins``, so a one-env-shaped cloning bug passes silently at ``num_envs=1``.
    """
    result = run_harness(
        _HARNESS_DIR / "static_move_assert.py",
        "--simulator",
        simulator,
        "--num-envs",
        num_envs,
        label=f"{simulator}/static-move (num_envs={num_envs})",
        timeout=900,
    )
    assert "STATIC-MOVE OK:" in result.stdout, result.stdout[-3000:] + result.stderr[-2000:]


@pytest.mark.parametrize("simulator", BACKENDS)
def test_subset_write(simulator: str) -> None:
    """Writing a subset of environments must leave the others untouched."""
    run_harness(
        _HARNESS_DIR / "subset_write_assert.py",
        "--simulator",
        simulator,
        "--num-envs",
        "4",
        label=f"{simulator}/subset-write",
        timeout=900,
    )


@pytest.mark.parametrize("simulator", BACKENDS)
def test_all_root_states_unified(simulator: str) -> None:
    """The unified root-state accessor must agree across robot and object assets.

    Worth running per backend: Newton reports these through an ``ArticulationView`` while PhysX
    uses a tensor view, and the two disagree on layout in ways the adapter has to reconcile.
    """
    run_harness(
        _HARNESS_DIR / "all_root_states_unified_assert.py",
        "--simulator",
        simulator,
        "--num-envs",
        "4",
        label=f"{simulator}/all-root-states-unified",
        timeout=900,
    )


@pytest.mark.parametrize("simulator", BACKENDS)
@pytest.mark.parametrize("scene", ["multibody", "multibody-override"])
def test_scene_spawn(simulator: str, scene: str) -> None:
    """Multi-body USD scene spawn, including the light rig.

    A wrong light *orientation* does not raise -- it aims the lights elsewhere and the scene
    renders black -- which is why this probes the authored light prims rather than trusting that
    spawning succeeded.
    """
    run_harness(
        _HARNESS_DIR / "scene_spawn_assert.py",
        "--simulator",
        simulator,
        "--scene",
        scene,
        "--num-envs",
        "4",
        label=f"{simulator}/scene-spawn",
        timeout=900,
    )


@pytest.mark.parametrize("simulator", BACKENDS)
def test_lights(simulator: str) -> None:
    result = run_harness(
        _HARNESS_DIR / "scene_spawn_assert.py",
        "--simulator",
        simulator,
        "--probe-lights",
        "--num-envs",
        "4",
        label=f"{simulator}/lights",
        timeout=900,
    )
    assert "PASS: SceneConfig.lights" in result.stdout, result.stdout[-3000:] + result.stderr[-2000:]


@pytest.mark.parametrize("simulator", BACKENDS)
def test_spawn_configs(simulator: str, tmp_path: Path) -> None:
    sentinel = tmp_path / "spawn-config-result.txt"
    run_harness(
        _HARNESS_DIR / "object_spawner_cfg_assert.py",
        "--simulator",
        simulator,
        "--result-file",
        str(sentinel),
        result_file=sentinel,
        label=f"{simulator}/spawn-configs",
        timeout=900,
    )


@pytest.mark.parametrize("simulator", BACKENDS)
def test_kinematic_playback(simulator: str) -> None:
    result = run_harness(
        _HARNESS_DIR / "playback_assert.py",
        "--simulator",
        simulator,
        "--num-envs",
        "4",
        label=f"{simulator}/playback",
        timeout=900,
    )
    assert "ALL PASS" in result.stdout, result.stdout[-3000:] + result.stderr[-2000:]
