"""The depth-distillation algo and terrain presets are registered, not bare module globals.

A preset that is only a module global is invisible to the CLI menu, and a registry whose
``config_type`` omits a member of the field's union drops matching entry-point presets in
``ConfigRegistry.ingest``. A module-level ``DEFAULTS`` also shadows the PEP 562 deprecated alias.
"""

from __future__ import annotations

import importlib
import pkgutil
import typing
import warnings

import pytest

import holosoma.config_values
import holosoma.config_values.algo as algo_values
import holosoma.config_values.terrain as terrain_values
from holosoma.config_types.algo import AlgoConfig, DistillationAlgoConfig, DistillationPPOAlgoConfig
from holosoma.config_values.experiment import get_annotated_experiment_config
from holosoma.utils.config_registry import parse_config

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize(
    ("token", "key", "config_type"),
    [
        ("algo:distillation-ppo", "distillation_ppo", DistillationPPOAlgoConfig),
        ("algo:distillation", "distillation", DistillationAlgoConfig),
    ],
)
def test_distillation_algo_is_selectable_from_the_cli(token: str, key: str, config_type: type) -> None:
    cfg = parse_config(get_annotated_experiment_config, args=["exp:g1-29dof", token])

    assert isinstance(cfg.algo, config_type)
    assert cfg.algo == algo_values.ALGO_REGISTRY[key]


def test_stairs_and_slope_eval_terrain_is_selectable_from_the_cli() -> None:
    key = "terrain_locomotion_stairs_and_slope_eval"
    cfg = parse_config(get_annotated_experiment_config, args=["exp:g1-29dof", "terrain:" + key.replace("_", "-")])

    assert cfg.terrain == terrain_values.TERRAIN_REGISTRY[key]
    assert cfg.terrain.terrain_term.fixed_step_height == 0.15


def test_algo_registry_accepts_every_algo_config_type() -> None:
    """Otherwise ingest() drops an entry-point preset of the missing type (logged at debug only)."""
    assert set(algo_values.ALGO_REGISTRY.config_types) == set(typing.get_args(AlgoConfig))
    for value in (algo_values.distillation_ppo, algo_values.distillation):
        name = f"_external_{type(value).__name__}"
        assert algo_values.ALGO_REGISTRY.ingest(name, value, source="test")
        del algo_values.ALGO_REGISTRY[name]


@pytest.mark.parametrize(
    ("module", "registry_name"),
    [(algo_values, "ALGO_REGISTRY"), (terrain_values, "TERRAIN_REGISTRY")],
)
def test_deprecated_defaults_alias_returns_the_registry(module: object, registry_name: str) -> None:
    with pytest.warns(DeprecationWarning, match="DEFAULTS is deprecated"):
        assert getattr(module, "DEFAULTS") is getattr(module, registry_name)  # noqa: B009


def test_no_config_values_module_shadows_the_deprecated_defaults_alias() -> None:
    """The alias is a module ``__getattr__``, which Python skips when ``DEFAULTS`` is a real global."""
    shadowing = []
    for info in pkgutil.iter_modules(holosoma.config_values.__path__):
        if info.ispkg:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            module = importlib.import_module(f"{holosoma.config_values.__name__}.{info.name}")
        if "__getattr__" in vars(module) and "DEFAULTS" in vars(module):
            shadowing.append(module.__name__)
    assert shadowing == []
