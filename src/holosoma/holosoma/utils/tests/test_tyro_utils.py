"""Tests for tyro CLI helper functions."""

from __future__ import annotations

import dataclasses
from typing import Annotated

import pytest
import tyro

from holosoma.utils.tyro_utils import find_dynamic_dict_fields, pop_dynamic_dict_args


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


def test_builds_instances_keyed_by_declaration() -> None:
    built = pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS, argv=["sensors.front:rgb", "sensors.belly:depth"])
    assert built == {"front": _RgbCamera(), "belly": _DepthCamera()}


def test_last_writer_wins_on_repeated_key() -> None:
    built = pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS, argv=["sensors.front:rgb", "sensors.front:depth"])
    assert built == {"front": _DepthCamera()}


def test_shared_variant_instances_are_isolated() -> None:
    # Two keys of the same variant must be distinct objects (deep-copied), so a per-key leaf
    # override never mutates the shared menu entry or the sibling key.
    built = pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS, argv=["sensors.a:rgb", "sensors.b:rgb"])
    assert built["a"] == built["b"] == _RgbCamera()
    assert built["a"] is not built["b"]
    assert built["a"] is not _SENSOR_VARIANTS["rgb"]  # menu entry untouched


def test_field_prefix_scoping() -> None:
    # A different field's declaration is left untouched (returned to the remainder, not consumed).
    built = pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS, argv=["loggers.a:rgb", "sensors.b:rgb"])
    assert built == {"b": _RgbCamera()}


def test_sys_argv_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setattr(sys, "argv", ["prog", "sensors.front:rgb", "--num-envs=4"])
    built = pop_dynamic_dict_args("sensors", _SENSOR_VARIANTS)  # argv=None -> uses/rewrites sys.argv
    assert built == {"front": _RgbCamera()}
    assert sys.argv == ["prog", "--num-envs=4"]  # declaration stripped, rest preserved


# --- find_dynamic_dict_fields -------------------------------------------------------------------


def test_scan_finds_annotated_and_preserves_value_metadata() -> None:
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


def test_scan_ignores_non_str_key_and_non_dict() -> None:
    @dataclasses.dataclass(frozen=True)
    class Cfg:
        int_keyed: dict[int, _RgbCamera] = dataclasses.field(default_factory=dict)
        listy: list[_RgbCamera] = dataclasses.field(default_factory=list)
        scalar: int = 0

    assert find_dynamic_dict_fields(Cfg) == {}


def test_scan_no_op_on_non_dataclass() -> None:
    # The Annotated[Union[...]] aliases from get_annotated_*_config() are not dataclasses.
    alias = Annotated[_RunConfig, "subcommand-ish"]
    assert find_dynamic_dict_fields(alias) == {"sensors": object}  # alias unwraps to the dataclass
    assert find_dynamic_dict_fields(int) == {}
    assert find_dynamic_dict_fields("not a type") == {}
