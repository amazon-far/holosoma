"""Check LiDAR profile validation and typed config at the external launch boundary."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from holosoma.simulator.isaacsim import backends
from tests.simulators import lidar_assert

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("simulator", ["mujoco", "mjwarp", "isaacsim"])
def test_lidar_newton_rejects_other_presets(monkeypatch, capsys, simulator):
    monkeypatch.setattr(sys, "argv", ["lidar_assert.py", "--simulator", simulator, "--physics-backend", "newton"])
    with pytest.raises(SystemExit) as error:
        lidar_assert.main()
    assert error.value.code == 2
    assert "--physics-backend newton requires --simulator isaaclab3" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("simulator", "physics_option", "expected_backend", "expected_physics"),
    [
        ("isaacsim", None, "isaaclab2", "physx"),
        ("isaaclab3", None, "isaaclab3", "physx"),
        ("isaaclab3", "newton", "isaaclab3", "newton"),
    ],
)
def test_lidar_profile_reaches_launch_as_typed_config(
    monkeypatch, tmp_path, simulator, physics_option, expected_backend, expected_physics
):
    argv = ["lidar_assert.py", "--simulator", simulator, "--num-envs", "3"]
    if physics_option is not None:
        argv += ["--physics-backend", physics_option]
    monkeypatch.setattr(sys, "argv", argv)
    # These are runtime availability boundaries; the real config builders and backend selection
    # remain in use. No native simulation is launched by this CPU test.
    monkeypatch.setattr(backends, "_installed_isaaclab_major", lambda: None)
    ray_module = tmp_path / "sensors" / "ray_caster" / "multi_mesh_ray_caster.py"
    ray_module.parent.mkdir(parents=True)
    ray_module.touch()
    monkeypatch.setattr(
        lidar_assert, "find_spec", lambda _name: SimpleNamespace(submodule_search_locations=[str(tmp_path)])
    )
    captured = []

    class LaunchBoundaryError(Exception):
        pass

    def capture_launch(config, *, device):
        captured.append((config, device))
        raise LaunchBoundaryError

    monkeypatch.setattr(lidar_assert, "launch_simulator", capture_launch)
    with pytest.raises(LaunchBoundaryError):
        lidar_assert.main()
    assert len(captured) == 1
    config, device = captured[0]
    assert config.simulator.config.isaaclab_backend.value == expected_backend
    assert config.simulator.config.sim.isaaclab3.physics_backend == expected_physics
    assert config.simulator.config.sim.isaaclab3.renderer == "isaac_rtx"
    assert config.training.num_envs == 3
    assert device == "cuda:0"
    assert {"downward", "cyclic", "cyclic_target", "tracked_target"} <= config.sensor.keys()
    assert ("disabled_child_target" in config.sensor) == (simulator == "isaaclab3")
