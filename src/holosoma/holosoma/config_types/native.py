"""Serializable native overrides without importing a simulator SDK."""

from __future__ import annotations

import ast
import json
import math
from typing import Annotated, Any

from pydantic import AfterValidator
from tyro.constructors import PrimitiveConstructorSpec


def _validate_json(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _validate_json(item, f"{path}.{key}")
        return
    raise ValueError(f"{path}: native settings require finite JSON values")


def _validate_settings(value: dict[str, Any]) -> dict[str, Any]:
    _validate_json(value, "native")
    for key in value:
        if not key.isidentifier() or key.startswith("_"):
            raise ValueError(f"{key}: use a public native field name")
    return value


def _from_cli(args: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(args[0])
    except ValueError:
        try:
            value = ast.literal_eval(args[0])
        except (ValueError, SyntaxError) as exc:
            raise ValueError("native settings expect a JSON object or Python dictionary") from exc
    if not isinstance(value, dict):
        raise ValueError("native settings expect a JSON object or Python dictionary")
    return value


NativeSettings = Annotated[
    dict[str, Any],
    AfterValidator(_validate_settings),
    PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=_from_cli,
        is_instance=lambda value: isinstance(value, dict),
        str_from_instance=lambda value: [repr(value)],
    ),
]
"""Native kwargs. Nested ``_type_`` records construct config classes lazily."""
