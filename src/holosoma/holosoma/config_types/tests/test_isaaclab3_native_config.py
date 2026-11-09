"""Native options stay portable data and preserve targeting precedence."""

from __future__ import annotations

import dataclasses
import sys
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.scene import PhysicsConfig
from holosoma.config_types.simulator import (
    IsaacLab3Config,
    IsaacLab3NewtonConfig,
    IsaacSimBackend,
    MujocoBackend,
    SimulatorInitConfig,
)
from holosoma.simulator.isaacsim.backends.native_config import build_config
from holosoma.simulator.isaacsim.backends.native_properties import schema_rules
from holosoma.utils.config_registry import parse_config
from holosoma.utils.eval_utils import _experiment_config_from_saved

pytestmark = pytest.mark.no_sim


def test_cli_and_saved_config_preserve_native_values_and_rule_order(tmp_path: Any) -> None:
    settings = {"solver_cfg": {"cone": "elliptic", "impratio": 5.0}, "use_cuda_graph": False}
    cfg = parse_config(
        ExperimentConfig,
        args=["simulator:isaacsim3", "--simulator.config.sim.isaaclab3.physics", repr(settings)],
    )
    rules = [
        {"match": "/z.*", "fragments": []},
        {"match": "/.*", "fragments": []},
    ]
    cfg = dataclasses.replace(
        cfg,
        robot=dataclasses.replace(
            cfg.robot,
            asset=dataclasses.replace(cfg.robot.asset, link_physics=PhysicsConfig(isaaclab3={"mass_props": rules})),
        ),
    )
    path = tmp_path / "native.yaml"
    cfg.save_config(str(path))
    restored = _experiment_config_from_saved(yaml.safe_load(path.read_text()))
    assert restored.simulator.config.sim.isaaclab3.physics == settings
    assert restored.robot.asset.link_physics is not None
    assert restored.robot.asset.link_physics.isaaclab3["mass_props"] == rules
    assert "newton" not in restored.to_serializable_dict()["simulator"]["config"]["sim"]["isaaclab3"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), (1, 2)])
def test_native_options_reject_values_that_cannot_roundtrip(value: Any) -> None:
    with pytest.raises(ValidationError, match="finite JSON"):
        IsaacLab3Config(physics={"value": value})


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
    expected = {
        "use_cuda_graph": False,
        "solver_cfg": {"nconmax": None, "njmax": 512, "iterations": 17},
        "default_shape_cfg": {"margin": 0.003},
    }
    restored = _experiment_config_from_saved(saved)
    assert restored.simulator.config.sim.isaaclab3.physics_settings == expected
    assert restored.simulator.config.sim.isaaclab3.newton.solver_iterations == 17
    assert restored.to_serializable_dict() == saved
    lab3["physics"] = {"solver_cfg": {"iterations": 9}}
    with pytest.raises(ValueError, match="Conflicting"):
        _experiment_config_from_saved(saved)


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
    assert restored.physics == {}


def test_legacy_python_constructor_and_repeated_replace_preserve_settings() -> None:
    before = set(sys.modules)
    legacy = IsaacLab3NewtonConfig(use_cuda_graph=False, nconmax=None, solver_iterations=17)
    # Preserve the pre-existing positional constructor as well as keyword input.
    config = IsaacLab3Config("newton", "newton_warp", False, legacy, False, True, False, "total")
    assert config.newton == legacy
    assert (config.use_fabric, config.create_stage_in_memory, config.use_newton_actuators) == (False, True, False)
    config = dataclasses.replace(config, newton=dataclasses.replace(config.newton, solver_iterations=23))
    config = dataclasses.replace(config, newton=dataclasses.replace(config.newton, njmax=512))
    assert config.physics_settings == {
        "use_cuda_graph": False,
        "solver_cfg": {"nconmax": None, "iterations": 23, "njmax": 512},
    }
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
    assert config.simulator.config.sim.isaaclab3.physics_settings == {
        "use_cuda_graph": False,
        "solver_cfg": {"nconmax": None, "njmax": 512, "iterations": 17},
        "default_shape_cfg": {"margin": 0.003},
    }
    assert config.simulator.config.sim.isaaclab3.newton.solver_iterations == 17
    restored = _experiment_config_from_saved(config.to_serializable_dict())
    assert restored.simulator.config.sim.isaaclab3 == config.simulator.config.sim.isaaclab3


def test_native_settings_coexist_with_untouched_legacy_defaults() -> None:
    settings = {"use_cuda_graph": False, "solver_cfg": {"nconmax": 256, "iterations": 17}}
    config = IsaacLab3Config(physics_backend="newton", physics=settings)
    assert config.physics_settings == settings
    config.physics_settings["solver_cfg"]["iterations"] = 99
    assert config.physics == settings
    combined = dataclasses.replace(config, newton=IsaacLab3NewtonConfig(default_shape_margin=0.003))
    assert combined.physics_settings == {**settings, "default_shape_cfg": {"margin": 0.003}}
    with pytest.raises(ValidationError, match="Conflicting.*solver_iterations"):
        dataclasses.replace(config, newton=IsaacLab3NewtonConfig(solver_iterations=9))
    with pytest.raises(ValidationError, match="cannot merge"):
        IsaacLab3Config(
            physics_backend="newton",
            newton=IsaacLab3NewtonConfig(solver_iterations=9),
            physics={"solver_cfg": None},
        )


def test_inactive_legacy_customization_survives_save_and_backend_switch() -> None:
    saved = ExperimentConfig().to_serializable_dict()
    lab3 = saved["simulator"]["config"]["sim"]["isaaclab3"]
    lab3["newton"] = {"solver_iterations": 17}
    config = _experiment_config_from_saved(saved)
    inactive = config.simulator.config.sim.isaaclab3
    assert inactive.physics_settings == {}
    restored = _experiment_config_from_saved(config.to_serializable_dict()).simulator.config.sim.isaaclab3
    assert restored.newton.solver_iterations == 17
    assert dataclasses.replace(restored, physics_backend="newton").physics_settings == {
        "solver_cfg": {"iterations": 17}
    }


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
