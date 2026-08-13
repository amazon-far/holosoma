from __future__ import annotations

# NOTE FOR AGENTS/MAINTAINERS: This module is intentionally mirrored with
# holosoma.utils.tyro_utils. Keep behavior changes in sync, with one deliberate
# exception: TYRO_CONIFG omits tyro.conf.FlagConversionOff here. Inference CLIs
# and docs use the --flag/--no-flag negation form because the old
# config/utils.py marker was nested one tuple too deep and silently ignored;
# training CLIs use --flag=True/False and keep FlagConversionOff.
import argparse
import collections.abc
import copy
import dataclasses
import re
import sys
import types
import typing
from typing import Any, Mapping

import typing_extensions
import tyro
import tyro.conf

TYRO_CONIFG = (
    tyro.conf.CascadeSubcommandArgs,
    tyro.conf.UsePythonSyntaxForLiteralCollections,
)


def split_import_file_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Return ``--import-file`` paths and the remaining CLI tokens."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--import-file", action="append", default=[], metavar="PATH", dest="import_file")
    known, remaining = pre.parse_known_args(argv)
    return known.import_file, remaining


def pop_import_file_args() -> list[str]:
    """Remove ``--import-file`` options from ``sys.argv`` and return their paths."""
    paths, remaining = split_import_file_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    return paths


def dynamic_dict_decl_pattern(field: str) -> re.Pattern[str]:
    """Return the matcher for ``<field>.<key>:<variant>`` tokens."""
    return re.compile(rf"^{re.escape(field)}\.([^.:=\s]+):([^.:=\s]+)$")


def pop_dynamic_dict_args(
    field: str,
    variants: Mapping[str, Any],
    *,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Build default dict entries from ``<field>.<key>:<variant>`` tokens.

    Args:
        field: Dict field name.
        variants: Variant name to default value.
        argv: Tokens to scan. ``None`` reads and rewrites ``sys.argv``.

    Returns:
        Mapping from declared key to a deep copy of the selected variant.
    """
    pattern = dynamic_dict_decl_pattern(field)
    rewrite = argv is None
    tokens = sys.argv[1:] if argv is None else argv

    built: dict[str, Any] = {}
    remaining: list[str] = []
    for token in tokens:
        match = pattern.match(token)
        if match is None:
            remaining.append(token)
            continue
        key, variant = match.group(1), match.group(2)
        if variant not in variants:
            raise SystemExit(f"Unknown {field} variant {variant!r} in '{token}'; choose from {sorted(variants)}.")
        # Copy so per-key leaf overrides don't mutate the shared variant instance.
        built[key] = copy.deepcopy(variants[variant])

    if rewrite:
        sys.argv = [sys.argv[0]] + remaining
    return built


def _strip_annotated(hint: Any) -> Any:
    """Return the underlying type of ``Annotated[T, ...]``."""
    return hint.__origin__ if hasattr(hint, "__metadata__") else hint


def _dict_value_hint(hint: Any) -> Any | None:
    """Return the value hint ``X`` for a ``dict[str, X]`` or ``Mapping[str, X]`` field.

    The returned hint keeps any ``Annotated`` metadata (so a ``UseRegistry`` marker on the
    value type survives); ``None`` means the field is not a string-keyed mapping.
    """
    inner = _strip_annotated(hint)
    origin = typing.get_origin(inner)
    if origin is typing.Union or origin is getattr(types, "UnionType", ()):
        members = [m for m in typing.get_args(inner) if m is not type(None)]
        if len(members) == 1:
            inner = _strip_annotated(members[0])
            origin = typing.get_origin(inner)
    if origin not in (dict, collections.abc.Mapping):
        return None
    args = typing.get_args(inner)
    if len(args) != 2 or args[0] is not str:
        return None
    return args[1]


def find_dynamic_dict_fields(config_type: Any) -> dict[str, Any]:
    """Return top-level string-keyed dict fields for a dataclass config.

    Args:
        config_type: Dataclass type or ``Annotated`` alias.

    Returns:
        Mapping from field name to the value hint (with ``Annotated`` metadata preserved).
    """
    base = _strip_annotated(config_type)
    if not (isinstance(base, type) and dataclasses.is_dataclass(base)):
        return {}

    try:
        hints = typing_extensions.get_type_hints(base, include_extras=True)
    except Exception:  # unresolvable annotations -> nothing to scan
        return {}

    field_names = {f.name for f in dataclasses.fields(base)}
    found: dict[str, Any] = {}
    for name, hint in hints.items():
        if name not in field_names:
            continue
        value_hint = _dict_value_hint(hint)
        if value_hint is not None:
            found[name] = value_hint
    return found
