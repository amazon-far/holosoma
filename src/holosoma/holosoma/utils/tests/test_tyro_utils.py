"""Tests for tyro CLI helper functions."""

from __future__ import annotations

import dataclasses

import pytest
import tyro
from typing_extensions import Annotated

from holosoma.utils.tyro_utils import TYRO_CONIFG, find_dynamic_dict_fields, pop_dynamic_dict_args


@dataclasses.dataclass(frozen=True)
class _RgbCamera:
    width: int = 640
    fov: float = 90.0


@dataclasses.dataclass(frozen=True)
class _DepthCamera:
    width: int = 320
    near: float = 0.1


# A variant menu holds default *instances*, the same shape as a DEFAULTS dict / plugin menu.
_SENSOR_VARIANTS = {"rgb": _RgbCamera(), "depth": _DepthCamera()}


@dataclasses.dataclass(frozen=True)
class _MujocoSim:
    fps: int = 2000


@dataclasses.dataclass(frozen=True)
class _IsaacSim:
    fps: int = 200


_SIM_DEFAULTS = {"mujoco": _MujocoSim(), "isaacsim": _IsaacSim()}


@dataclasses.dataclass(frozen=True)
class _RunConfig:
    """Stand-in for RunSimConfig: a subcommand field plus a dynamic dict field."""

    simulator: Annotated[
        object,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(_SIM_DEFAULTS)),
    ] = _SIM_DEFAULTS["mujoco"]
    sensors: dict[str, object] = dataclasses.field(default_factory=dict)
    num_envs: int = 1


def _cli(argv: list[str]) -> _RunConfig:
    """Build the CLI the way run_sim.main() would: extract sensor decls, inject as the default."""
    sensors = pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS, argv=argv)
    remaining = [t for t in argv if not t.startswith("sensors.") or ":" not in t]
    default = dataclasses.replace(_RunConfig(), sensors=sensors)
    return tyro.cli(_RunConfig, args=remaining, default=default, config=TYRO_CONIFG)


def test_no_declarations_gives_empty_dict():
    assert pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS, argv=[]) == {}
    assert pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS, argv=["--num-envs=4"]) == {}


def test_builds_instances_keyed_by_declaration():
    built = pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS, argv=["sensors.front:rgb", "sensors.belly:depth"])
    assert built == {"front": _RgbCamera(), "belly": _DepthCamera()}


def test_last_writer_wins_on_repeated_key():
    built = pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS, argv=["sensors.front:rgb", "sensors.front:depth"])
    assert built == {"front": _DepthCamera()}


def test_unknown_variant_fails_loud():
    with pytest.raises(SystemExit, match="Unknown sensors variant 'lidar'"):
        pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS, argv=["sensors.front:lidar"])


def test_shared_variant_instances_are_isolated():
    # Two keys of the same variant must be distinct objects (deep-copied), so a per-key leaf
    # override never mutates the shared menu entry or the sibling key.
    built = pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS, argv=["sensors.a:rgb", "sensors.b:rgb"])
    assert built["a"] == built["b"] == _RgbCamera()
    assert built["a"] is not built["b"]
    assert built["a"] is not _SENSOR_VARIANTS["rgb"]  # menu entry untouched


def test_config_file_added_variant_flows_through():
    # A variant merged into the menu by the import-file / entry-point loader (load_file_presets
    # mutates the DEFAULTS dict in place) is selectable with no change to this helper.
    menu = dict(_SENSOR_VARIANTS)
    menu["thermal"] = _DepthCamera(width=160, near=0.02)  # stand-in for a plugin-supplied preset
    built = pop_dynamic_dict_args("sensors", menu, argv=["sensors.ir:thermal"])
    assert built == {"ir": _DepthCamera(width=160, near=0.02)}


def test_field_prefix_scoping():
    # A different field's declaration is left untouched (returned to the remainder, not consumed).
    built = pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS, argv=["loggers.a:rgb", "sensors.b:rgb"])
    assert built == {"b": _RgbCamera()}


def test_sys_argv_rewrite(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["prog", "sensors.front:rgb", "--num-envs=4"])
    built = pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS)  # argv=None -> uses/rewrites sys.argv
    assert built == {"front": _RgbCamera()}
    assert sys.argv == ["prog", "--num-envs=4"]  # declaration stripped, rest preserved


def test_end_to_end_with_subcommand_and_overrides():
    cfg = _cli(
        [
            "simulator:isaacsim",
            "sensors.front:rgb",
            "sensors.belly:depth",
            "--simulator.fps=240",
            "--sensors.front.width=1280",
            "--sensors.belly.near=0.05",
            "--num-envs=4",
        ]
    )
    assert cfg.simulator == _IsaacSim(fps=240)
    assert cfg.sensors == {
        "front": _RgbCamera(width=1280),
        "belly": _DepthCamera(near=0.05),
    }
    assert cfg.num_envs == 4


def test_end_to_end_no_sensors_defaults_clean():
    cfg = _cli(["simulator:mujoco", "--num-envs=8"])
    assert cfg.simulator == _MujocoSim()
    assert cfg.sensors == {}
    assert cfg.num_envs == 8


# --- find_dynamic_dict_fields -------------------------------------------------------------------


def test_scan_finds_plain_dict_field():
    assert find_dynamic_dict_fields(_RunConfig) == {"sensors": object}


def test_scan_finds_annotated_and_preserves_value_metadata():
    import typing

    @dataclasses.dataclass(frozen=True)
    class Cfg:
        rgb_only: dict[str, _RgbCamera] = dataclasses.field(default_factory=dict)
        # Annotated on the outer dict is stripped; Annotated on the VALUE type is kept so a
        # UseRegistry marker there survives the scan.
        wrapped: Annotated[dict[str, Annotated[_DepthCamera, "vmeta"]], "outer"] = dataclasses.field(
            default_factory=dict
        )
        n: int = 1

    found = find_dynamic_dict_fields(Cfg)
    assert set(found) == {"rgb_only", "wrapped"}
    assert found["rgb_only"] is _RgbCamera  # plain value type returned as-is
    # The value hint keeps its Annotated metadata rather than being unwrapped to _DepthCamera.
    assert getattr(found["wrapped"], "__metadata__", ()) == ("vmeta",)
    assert typing.get_args(found["wrapped"])[0] is _DepthCamera


def test_scan_ignores_non_str_key_and_non_dict():
    @dataclasses.dataclass(frozen=True)
    class Cfg:
        int_keyed: dict[int, _RgbCamera] = dataclasses.field(default_factory=dict)
        listy: list[_RgbCamera] = dataclasses.field(default_factory=list)
        scalar: int = 0

    assert find_dynamic_dict_fields(Cfg) == {}


def test_scan_no_op_on_non_dataclass():
    # The Annotated[Union[...]] aliases from get_annotated_*_config() are not dataclasses.
    alias = Annotated[_RunConfig, "subcommand-ish"]
    assert find_dynamic_dict_fields(alias) == {"sensors": object}  # alias unwraps to the dataclass
    assert find_dynamic_dict_fields(int) == {}
    assert find_dynamic_dict_fields("not a type") == {}
