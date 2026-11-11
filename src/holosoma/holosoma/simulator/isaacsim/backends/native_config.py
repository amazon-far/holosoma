"""Construct SDK configs from private backend mappings after runtime selection."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any


_OWNED_FIELDS = frozenset({"class_type", "launcher_type", "renderer_type", "cloning_contexts", "func"})


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
        if isinstance(value, dict) and dataclasses.is_dataclass(current):
            child_defaults = {
                field.name: getattr(current, field.name) for field in dataclasses.fields(current) if field.init
            }
            kwargs[name] = build_config(type(current), value, location=child_path, defaults=child_defaults)
        else:
            kwargs[name] = copy.deepcopy(value)
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: invalid {cls.__name__}: {exc}") from exc
