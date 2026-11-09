"""Resolve compact Lab 3 overrides only after runtime selection."""

from __future__ import annotations

import copy
import dataclasses
import importlib
from typing import Any


_OWNED_FIELDS = frozenset({"class_type", "launcher_type", "renderer_type", "cloning_contexts", "func"})


def _config_class(path: Any, location: str) -> type[Any]:
    if not isinstance(path, str) or ":" not in path:
        raise ValueError(f"{location}._type_: expected 'module:ConfigClass'")
    module, name = path.split(":", 1)
    if not module.startswith(("isaaclab.", "isaaclab_newton.", "isaaclab_physx.", "isaaclab_ov.")):
        raise ValueError(f"{location}._type_: expected an Isaac Lab configuration class")
    try:
        cls = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"{location}._type_: cannot resolve {path!r}") from exc
    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        raise ValueError(f"{location}._type_: {path!r} is not a configuration dataclass")
    return cls


def resolve_value(value: Any, *, location: str) -> Any:
    """Resolve explicitly tagged configs, preserving ordinary mappings and lists."""
    if isinstance(value, dict):
        if "_type_" in value:
            cls = _config_class(value["_type_"], location)
            return build_config(cls, {key: item for key, item in value.items() if key != "_type_"}, location=location)
        return {key: resolve_value(item, location=f"{location}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_value(item, location=f"{location}[{index}]") for index, item in enumerate(value)]
    return value


def build_config(
    cls: type[Any],
    settings: dict[str, Any],
    *,
    location: str,
    defaults: dict[str, Any] | None = None,
) -> Any:
    """Construct native kwargs, merging nested config overrides without mutating defaults."""
    fields = {field.name for field in dataclasses.fields(cls) if field.init}
    unknown = settings.keys() - fields
    if unknown:
        raise ValueError(f"{location}: unknown {cls.__name__} fields: {', '.join(sorted(unknown))}")
    owned = settings.keys() & _OWNED_FIELDS
    if "solver_type" in settings and isinstance(getattr(cls, "solver_type", None), str):
        owned = owned | {"solver_type"}
    if owned:
        raise ValueError(f"{location}: fields selected by the backend/config type: {', '.join(sorted(owned))}")
    kwargs = copy.deepcopy(defaults or {})
    for name, value in settings.items():
        current = kwargs.get(name)
        if current is None:
            field = next(field for field in dataclasses.fields(cls) if field.name == name)
            if field.default is not dataclasses.MISSING:
                current = field.default
            elif field.default_factory is not dataclasses.MISSING:
                current = field.default_factory()
        child_path = f"{location}.{name}"
        if isinstance(value, dict) and "_type_" not in value and dataclasses.is_dataclass(current):
            child_defaults = {
                field.name: getattr(current, field.name) for field in dataclasses.fields(current) if field.init
            }
            kwargs[name] = build_config(type(current), value, location=child_path, defaults=child_defaults)
        else:
            kwargs[name] = resolve_value(value, location=child_path)
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: invalid {cls.__name__}: {exc}") from exc
