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
from holosoma.config_types.simulator import IsaacLab3Config
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


def test_historical_newton_settings_migrate_without_duplicate_user_options() -> None:
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
    assert _experiment_config_from_saved(saved).simulator.config.sim.isaaclab3.physics == expected
    lab3["physics"] = {"solver_cfg": {"iterations": 9}}
    with pytest.raises(ValueError, match="conflicting"):
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
