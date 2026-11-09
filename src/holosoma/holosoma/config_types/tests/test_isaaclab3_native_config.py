"""Native options stay portable data and preserve targeting precedence."""

from __future__ import annotations

import dataclasses
import sys
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.mujoco import MujocoConfig, MujocoWarpConfig
from holosoma.config_types.newton import IsaacLab3RendererConfig, NewtonCollisionConfig
from holosoma.config_types.scene import MujocoPhysicsConfig, PhysicsConfig
from holosoma.config_types.simulator import (
    IsaacLab3Config,
    IsaacLab3NewtonConfig,
    IsaacSimBackend,
    MujocoBackend,
    SimulatorInitConfig,
)
from holosoma.simulator.isaacsim.backends.native_config import build_config
from holosoma.simulator.isaacsim.backends.native_properties import schema_rules
from holosoma.simulator.isaacsim.backends.newton_config import solver_settings
from holosoma.utils.config_registry import parse_config
from holosoma.utils.eval_utils import _experiment_config_from_saved

pytestmark = pytest.mark.no_sim


def test_cli_and_saved_config_preserve_shared_options_and_link_precedence(tmp_path: Any) -> None:
    cfg = parse_config(
        ExperimentConfig,
        args=[
            "simulator:isaacsim3",
            "--simulator.config.sim.isaaclab3.physics-backend",
            "newton",
            "--simulator.config.sim.mujoco.iterations",
            "17",
            "--simulator.config.sim.mujoco.cone",
            "elliptic",
            "--simulator.config.sim.mujoco.warp.nconmax-per-env",
            "256",
            "--simulator.config.sim.isaaclab3.renderer-options.enable-shadows",
            "False",
        ],
    )
    overrides = {
        "pelvis": PhysicsConfig(mass=42.0, mujoco=MujocoPhysicsConfig(gravity_compensation=0.25)),
        "*": PhysicsConfig(mujoco=MujocoPhysicsConfig(friction=[0.45, 0.07, 0.03])),
    }
    cfg = dataclasses.replace(
        cfg,
        robot=dataclasses.replace(
            cfg.robot, asset=dataclasses.replace(cfg.robot.asset, link_physics_overrides=overrides)
        ),
    )
    path = tmp_path / "typed.yaml"
    cfg.save_config(str(path))
    restored = _experiment_config_from_saved(yaml.safe_load(path.read_text()))
    assert restored == cfg
    assert restored.robot.asset.link_physics_overrides is not None
    assert list(restored.robot.asset.link_physics_overrides) == ["pelvis", "*"]
    settings = solver_settings(restored.simulator.config.sim.mujoco, restored.simulator.config.sim.isaaclab3.newton)
    assert settings["iterations"] == 17
    assert settings["nconmax"] == 256
    assert settings["cone"] == "elliptic"
    assert restored.simulator.config.sim.isaaclab3.renderer_options.enable_shadows is False


def test_shared_solver_controls_preserve_defaults_and_explicit_zero_without_sdk_imports() -> None:
    before = set(sys.modules)
    defaults = solver_settings(MujocoConfig(), IsaacLab3NewtonConfig())
    assert defaults == {"nconmax": 128, "use_mujoco_contacts": True, "disable_sensors": False}
    settings = solver_settings(
        MujocoConfig(
            iterations=0,
            ls_iterations=0,
            tolerance=0,
            ls_tolerance=0,
            jacobian="sparse",
            warp=MujocoWarpConfig(nconmax_per_env=256, njmax_per_env=512),
        ),
        IsaacLab3NewtonConfig(),
    )
    assert settings["iterations"] == settings["ls_iterations"] == settings["tolerance"] == settings["ls_tolerance"] == 0
    assert settings["jacobian"] == "sparse"
    assert (settings["nconmax"], settings["njmax"]) == (256, 512)
    assert not any(name.startswith(("isaaclab.", "isaaclab_newton.")) for name in set(sys.modules) - before)


@pytest.mark.parametrize(
    ("shared", "path"),
    [
        (MujocoConfig(solver="pgs"), "solver"),
        (MujocoConfig(integrator="implicit"), "integrator"),
        (MujocoConfig(noslip_iterations=1), "noslip_iterations"),
        (MujocoConfig(warp=MujocoWarpConfig(naconmax=1024)), "naconmax"),
    ],
)
def test_unsupported_shared_inputs_report_the_config_path(shared: MujocoConfig, path: str) -> None:
    with pytest.raises(ValueError, match=path):
        solver_settings(shared, IsaacLab3NewtonConfig())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_typed_renderer_rejects_invalid_distance(value: float) -> None:
    with pytest.raises(ValidationError):
        IsaacLab3RendererConfig(max_distance=value)


def test_collision_options_have_one_selector() -> None:
    with pytest.raises(ValidationError, match="requires collision_source"):
        IsaacLab3NewtonConfig(collision=NewtonCollisionConfig(broad_phase="sap"))
    settings = solver_settings(MujocoConfig(), IsaacLab3NewtonConfig(collision_source="newton"))
    assert settings["use_mujoco_contacts"] is False


def test_public_options_reject_sdk_constructor_dictionaries() -> None:
    kwargs: dict[str, Any] = {"physics": {"solver_cfg": {"iterations": 17}}}
    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        IsaacLab3Config(**kwargs)
    kwargs = {"_type_": "isaaclab_ov.renderers:OVRTXRendererCfg"}
    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        IsaacLab3RendererConfig(**kwargs)


def test_schema_rules_do_not_import_the_sdk_and_keep_empty_fragment_rules() -> None:
    before = set(sys.modules)
    rules = [{"match": "/z.*", "fragments": []}, {"match": "/.*", "fragments": []}]
    assert schema_rules(rules, location="mass_props") == [("/z.*", []), ("/.*", [])]
    assert not any(name.startswith(("isaaclab.", "isaaclab_newton.")) for name in set(sys.modules) - before)


@pytest.mark.parametrize(
    "value",
    [
        [{"match": "/foo", "fragments": []}, {"_type_": "isaaclab.sim.schemas:MassCfg"}],
        [{"match": "/[", "fragments": []}],
        [{"match": "/foo", "fragments": {}}],
    ],
)
def test_invalid_rules_fail_with_config_location(value: Any) -> None:
    with pytest.raises(ValueError, match="physics.mass_props"):
        schema_rules(value, location="physics.mass_props")


def test_nested_native_overrides_merge_and_do_not_mutate_defaults() -> None:
    @dataclasses.dataclass
    class Solver:
        iterations: int = 100
        cone: str = "pyramidal"

    @dataclasses.dataclass
    class Physics:
        solver_cfg: Solver = dataclasses.field(default_factory=Solver)

    solver = Solver(iterations=13)
    native = build_config(
        Physics, {"solver_cfg": {"cone": "elliptic"}}, defaults={"solver_cfg": solver}, location="physics"
    )
    assert native.solver_cfg == Solver(iterations=13, cone="elliptic")
    assert solver == Solver(iterations=13)
    assert build_config(Physics, {"solver_cfg": {"iterations": 7}}, location="physics").solver_cfg == Solver(
        iterations=7
    )
    with pytest.raises(ValueError, match="physics.solver_cfg.*unknown.*iterations_typo"):
        build_config(Physics, {"solver_cfg": {"iterations_typo": 7}}, location="physics")


def test_historical_newton_settings_load_without_rewriting_user_inputs() -> None:
    saved = ExperimentConfig().to_serializable_dict()
    lab3 = saved["simulator"]["config"]["sim"]["isaaclab3"]
    lab3["physics_backend"] = "newton"
    lab3["newton"] = {
        "use_cuda_graph": False,
        "nconmax": None,
        "njmax": 512,
        "solver_iterations": 17,
        "default_shape_margin": 0.003,
    }
    restored = _experiment_config_from_saved(saved)
    newton = restored.simulator.config.sim.isaaclab3.newton
    resolved = solver_settings(MujocoConfig(), newton)
    assert {name: resolved[name] for name in ("nconmax", "njmax", "iterations")} == {
        "nconmax": None,
        "njmax": 512,
        "iterations": 17,
    }
    assert newton.use_cuda_graph is False
    assert newton.default_shape_margin == 0.003
    serialized = restored.to_serializable_dict()
    for name, value in lab3["newton"].items():
        assert serialized["simulator"]["config"]["sim"]["isaaclab3"]["newton"][name] == value
    assert _experiment_config_from_saved(serialized) == restored
    with pytest.raises(ValueError, match="Conflicting"):
        solver_settings(MujocoConfig(iterations=9), restored.simulator.config.sim.isaaclab3.newton)


def test_inactive_historical_newton_defaults_do_not_become_physx_kwargs() -> None:
    saved = ExperimentConfig().to_serializable_dict()
    lab3 = saved["simulator"]["config"]["sim"]["isaaclab3"]
    lab3["newton"] = {
        "use_cuda_graph": True,
        "nconmax": 128,
        "njmax": None,
        "solver_iterations": None,
        "default_shape_margin": 0.01,
    }
    restored = _experiment_config_from_saved(saved).simulator.config.sim.isaaclab3
    assert restored.physics_backend == "physx"
    assert restored.newton == IsaacLab3NewtonConfig()


def test_legacy_python_constructor_and_repeated_replace_preserve_settings() -> None:
    before = set(sys.modules)
    legacy = IsaacLab3NewtonConfig(use_cuda_graph=False, nconmax=None, solver_iterations=17)
    # Preserve the pre-existing positional constructor as well as keyword input.
    config = IsaacLab3Config("newton", "newton_warp", False, legacy, False, True, False, "total")
    assert config.newton == legacy
    assert (config.use_fabric, config.create_stage_in_memory, config.use_newton_actuators) == (False, True, False)
    config = dataclasses.replace(config, newton=dataclasses.replace(config.newton, solver_iterations=23))
    config = dataclasses.replace(config, newton=dataclasses.replace(config.newton, njmax=512))
    resolved = solver_settings(MujocoConfig(), config.newton)
    assert (resolved["nconmax"], resolved["iterations"], resolved["njmax"]) == (None, 23, 512)
    assert config.newton.use_cuda_graph is False
    assert not any(name.startswith(("isaaclab.", "isaaclab_newton.")) for name in set(sys.modules) - before)


@pytest.mark.parametrize("style", ["separate", "equals"])
def test_legacy_cli_flags_remain_accepted(style: str) -> None:
    values = {
        "use-cuda-graph": "False",
        "nconmax": "None",
        "njmax": "512",
        "solver-iterations": "17",
        "default-shape-margin": "0.003",
    }
    args = ["simulator:isaacsim3", "--simulator.config.sim.isaaclab3.physics-backend=newton"]
    for name, value in values.items():
        flag = f"--simulator.config.sim.isaaclab3.newton.{name}"
        args.extend([f"{flag}={value}"] if style == "equals" else [flag, value])
    config = parse_config(ExperimentConfig, args=args)
    assert config.simulator.config.sim.isaaclab3.newton == IsaacLab3NewtonConfig(
        use_cuda_graph=False,
        nconmax=None,
        njmax=512,
        solver_iterations=17,
        default_shape_margin=0.003,
    )
    restored = _experiment_config_from_saved(config.to_serializable_dict())
    assert restored.simulator.config.sim.isaaclab3 == config.simulator.config.sim.isaaclab3


def test_shared_settings_coexist_with_untouched_legacy_defaults() -> None:
    shared = MujocoConfig(iterations=17, warp=MujocoWarpConfig(nconmax_per_env=256))
    settings = solver_settings(shared, IsaacLab3NewtonConfig())
    assert (settings["iterations"], settings["nconmax"]) == (17, 256)
    settings["iterations"] = 99
    assert shared.iterations == 17
    with pytest.raises(ValueError, match="Conflicting.*solver_iterations"):
        solver_settings(shared, IsaacLab3NewtonConfig(solver_iterations=9))
    with pytest.raises(ValueError, match="Conflicting.*nconmax"):
        solver_settings(shared, IsaacLab3NewtonConfig(nconmax=512))
    assert solver_settings(shared, IsaacLab3NewtonConfig(solver_iterations=17))["iterations"] == 17


def test_inactive_legacy_customization_survives_save_and_backend_switch() -> None:
    saved = ExperimentConfig().to_serializable_dict()
    lab3 = saved["simulator"]["config"]["sim"]["isaaclab3"]
    lab3["newton"] = {"solver_iterations": 17}
    config = _experiment_config_from_saved(saved)
    inactive = config.simulator.config.sim.isaaclab3
    assert inactive.physics_backend == "physx"
    restored = _experiment_config_from_saved(config.to_serializable_dict()).simulator.config.sim.isaaclab3
    assert restored.newton.solver_iterations == 17
    switched = dataclasses.replace(restored, physics_backend="newton")
    assert solver_settings(MujocoConfig(), switched.newton)["iterations"] == 17


def test_pre_lab3_simulator_constructor_keeps_positional_field_order() -> None:
    preset = ExperimentConfig().simulator.config
    config = SimulatorInitConfig(
        preset.name,
        preset.sim,
        preset.debug_viz,
        preset.viewer,
        preset.reset_manager,
        preset.robot_mjcf_filter,
        preset.isaacsim,
        MujocoBackend.WARP,
        preset.bridge,
        preset.virtual_gantry,
    )
    assert config.mujoco_backend == MujocoBackend.WARP
    assert config.bridge == preset.bridge
    assert config.virtual_gantry == preset.virtual_gantry
    assert config.isaacsim_backend == IsaacSimBackend.ISAACLAB2
