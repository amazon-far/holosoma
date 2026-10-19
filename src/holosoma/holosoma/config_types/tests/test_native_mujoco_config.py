"""Public parsing/serialization/validation contracts for the focused native schema."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.mujoco import MujocoConfig, MujocoWarpConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.scene import MujocoPhysicsConfig
from holosoma.config_types.simulator import SimulatorInitConfig
from holosoma.config_values.simulator import mujoco as preset
from holosoma.utils.config_registry import parse_config

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("kind", [RunSimConfig, ExperimentConfig])
@pytest.mark.parametrize("backend", ["mujoco", "mjwarp"])
def test_shared_entrypoint_cli_yaml_and_preset_roundtrip(
    kind: type[RunSimConfig | ExperimentConfig], backend: str
) -> None:
    args = [
        f"simulator:{backend}",
        "--simulator.config.sim.mujoco.integrator",
        "rk4",
        "--simulator.config.sim.mujoco.iterations",
        "0",
        "--simulator.config.sim.mujoco.disableflags",
        "('contact', 'gravity')",
    ]
    if backend == "mjwarp":
        args += [
            "--simulator.config.sim.mujoco.warp.nconmax-per-env",
            "128",
            "--simulator.config.sim.mujoco.warp.naconmax",
            "300",
            "--simulator.config.sim.mujoco.warp.graph-conditional",
            "False",
        ]
    parsed = parse_config(
        kind,
        args=args,
    )
    native = parsed.simulator.config.sim.mujoco
    assert native.integrator == "rk4"
    assert native.iterations == 0
    assert native.disableflags == ("contact", "gravity")
    if backend == "mjwarp":
        assert native.warp.nconmax_per_env == 128
        assert native.warp.naconmax == 300
        assert native.warp.graph_conditional is False
    else:
        assert native.warp.nconmax_per_env is native.warp.naconmax is native.warp.graph_conditional is None
    adapter: TypeAdapter[RunSimConfig | ExperimentConfig] = TypeAdapter(kind)
    encoded = adapter.dump_python(parsed, mode="json")
    if kind is ExperimentConfig:
        assert parsed.to_serializable_dict() == encoded
    restored = adapter.validate_python(yaml.safe_load(yaml.safe_dump(encoded)))
    assert restored.simulator.config.sim.mujoco == native
    assert "mujoco_warp" not in encoded["simulator"]["config"]
    assert preset.config.sim.mujoco.integrator is None


def test_extra_cli_false_and_yaml_values_roundtrip() -> None:
    cfg = parse_config(
        MujocoConfig,
        args=[
            "--extra",
            "{'ccd_iterations': 0, 'wind': [0, 0, 0], 'gravity': [0, 0, -2]}",
            "--warp.graph-conditional",
            "False",
            "--warp.extras",
            '{"run_collision_detection": false, "nccdmax": 0}',
        ],
    )
    assert cfg.extra["ccd_iterations"] == 0
    assert cfg.extra["wind"] == [0, 0, 0]
    assert cfg.extra["gravity"] == [0, 0, -2]
    assert cfg.warp.extras["run_collision_detection"] is False
    assert cfg.warp.extras["nccdmax"] == 0
    assert cfg.warp.graph_conditional is False
    adapter = TypeAdapter(MujocoConfig)
    assert adapter.validate_python(yaml.safe_load(yaml.safe_dump(adapter.dump_python(cfg, mode="json")))) == cfg


@pytest.mark.parametrize(
    ("kind", "values"),
    [
        (
            MujocoPhysicsConfig,
            {"friction": [0.4, 0.01, 0.001], "solref": [-3000, -3], "solimp": [0.8, 0.9, 0.001, 0.5, 2]},
        ),
    ],
)
def test_native_vectors_preserve_existing_list_cli_yaml_and_preset_conventions(
    kind: type, values: dict[str, list[float]]
) -> None:
    args = [token for name, value in values.items() for token in (f"--{name}", repr(value))]
    parsed = parse_config(kind, args=args)
    for name, value in values.items():
        assert getattr(parsed, name) == value
        assert isinstance(getattr(parsed, name), list)
    assert replace(kind(), **values) == parsed
    adapter: TypeAdapter[Any] = TypeAdapter(kind)
    assert adapter.validate_python(yaml.safe_load(yaml.safe_dump(adapter.dump_python(parsed, mode="json")))) == parsed


@pytest.mark.parametrize(
    ("kind", "field"),
    [(MujocoConfig, "extra"), (MujocoWarpConfig, "extras"), (MujocoPhysicsConfig, "extra")],
)
@pytest.mark.parametrize("name", ["option.tolerance", "_private"])
def test_extra_name_validation_is_shared_by_all_native_configs(kind: type, field: str, name: str) -> None:
    with pytest.raises(ValidationError, match="bare public native name"):
        TypeAdapter(kind).validate_python({field: {name: 0}})


@pytest.mark.parametrize("path", ["extra", "warp.extras"])
def test_cli_timestep_override_fails_during_config_parsing(path: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises((ValidationError, SystemExit)) as error:
        parse_config(MujocoConfig, args=[f"--{path}", '{"timestep": 0.005}'])
    if isinstance(error.value, SystemExit):
        assert error.value.code == 2
    assert "sim.fps owns timestep" in str(error.value) + capsys.readouterr().err


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"integrator": "bad"}, "integrator"),
        ({"iterations": -1}, "iterations"),
        ({"gravity": [0, 0, -2]}, "gravity"),
        ({"extra": {"gravity": [0, 0, float("inf")]}}, "finite"),
        ({"tolerance": float("nan")}, "tolerance"),
        ({"extra": {"wind": [0, float("nan"), 0]}}, "finite"),
        ({"extra": {"timestep": 0.005}}, "fps"),
        ({"extra": {"integrator": "rk4"}}, "explicit"),
        ({"extra": {"_private": 0}}, "public"),
        ({"timstep": 0.005}, "timstep"),
        ({"warp": {"nconmax_per_env": -1}}, "nconmax"),
        ({"warp": {"naconmax": -1}}, "naconmax"),
        ({"warp": {"graph_conditional": 0}}, "graph_conditional"),
        ({"warp": {"extras": {"nconmax": 10}}}, "nconmax_per_env"),
        ({"warp": {"extras": {"nworld": 4}}}, "training.num_envs"),
        ({"warp": {"extras": {"graph_conditional": False}}}, "explicit"),
        ({"warp": {"extras": {"option.graph_conditional": False}}}, "without prefixes"),
        ({"warp": {"extras": {"data.naconmax": 300}}}, "without prefixes"),
    ],
)
def test_pydantic_rejects_invalid_values_and_duplicate_owners(kwargs: dict[str, Any], reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        MujocoConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"friction": [-0.1, 0, 0]}, "friction"),
        ({"friction": [0.1, 0]}, "friction"),
        ({"friction": [0.1, 0, 0, 0]}, "friction"),
        ({"friction": [0.1, float("nan"), 0]}, "finite"),
        ({"solref": [0.02, -1]}, "solref"),
        ({"solref": [0.02]}, "solref"),
        ({"solref": [0.02, 1, 1]}, "solref"),
        ({"solref": [float("inf"), 1]}, "solref"),
        ({"solimp": [0.8, 0.9, 0.001, 0.5]}, "solimp"),
        ({"solimp": [0.8, 0.9, 0.001, 0.5, 2, 0]}, "solimp"),
        ({"solimp": [0.8, 0.9, float("inf"), 0.5, 2]}, "finite"),
        ({"solimp": [-0.1, 0.9, 0.001, 0.5, 2]}, "solimp"),
        ({"solimp": [0.8, 1.1, 0.001, 0.5, 2]}, "solimp"),
        ({"solimp": [0.8, 0.9, -0.001, 0.5, 2]}, "solimp"),
        ({"solimp": [0.8, 0.9, 0.001, 1.1, 2]}, "solimp"),
        ({"solimp": [0.8, 0.9, 0.001, 0.5, 0.5]}, "solimp"),
        ({"condim": 2}, "condim"),
        ({"contype": 2147483648}, "contype"),
    ],
)
def test_geom_contact_config_rejects_invalid_native_parameters(kwargs: dict[str, Any], reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        MujocoPhysicsConfig(**kwargs)


def test_old_warp_capacity_location_fails_with_replacement_in_yaml_and_python() -> None:
    old = TypeAdapter(SimulatorInitConfig).dump_python(preset.config, mode="json")
    old["mujoco_warp"] = {"nconmax_per_env": 128}
    with pytest.raises(ValidationError, match=r"sim.mujoco\."):
        SimulatorInitConfig(**yaml.safe_load(yaml.safe_dump(old)))
    with pytest.raises(ValidationError, match=r"sim.mujoco\."):
        # The invalid legacy keyword is intentional; this checks the migration error.
        replace(preset.config, mujoco_warp={"nconmax_per_env": 128})  # type: ignore[call-arg]


def test_old_cli_path_fails_with_replacement() -> None:
    with pytest.raises(SystemExit, match=r"sim.mujoco\."):
        parse_config(RunSimConfig, args=["--simulator.config.mujoco-warp.nconmax-per-env", "128"])


def test_nested_warp_uses_pydantic_and_dataclass_preset_replacement() -> None:
    native = replace(MujocoConfig(), warp=replace(MujocoWarpConfig(), graph_conditional=False, naconmax=0))
    assert native.warp.graph_conditional is False
    assert native.warp.naconmax == 0
    with pytest.raises(ValidationError, match="graph_conditional"):
        replace(native.warp, graph_conditional=0)  # type: ignore[arg-type]  # Deliberately invalid input.
