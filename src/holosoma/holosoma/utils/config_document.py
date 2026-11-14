"""Expand ``--config PATH`` run documents into ``run_sim`` CLI tokens.

Document shape (YAML or JSON)::

    version: 2
    backend: isaacsim            # optional; same as arguments.simulator
    arguments:
      robot: g1-29dof            # <group>:<preset>
      sensor:                    # <family>.<key>:<preset>
        scan: g1-pelvis-lidar
      overrides:                 # --<dotted.path> <value>
        training.headless: true
        robot.init_state.pos: [0.0, 0.0, 0.8]

A file whose root has no ``version`` but has a ``holosoma_sim`` mapping reads that mapping;
``--config-path a.b`` selects any other nested mapping. Tokens on the command line take
precedence over the document. YAML lists encode as Python lists; write tuple-typed values as
quoted tuple literals.
"""

from __future__ import annotations

import argparse
import re
from typing import Any

import yaml

SUPPORTED_VERSION = 2
EMBEDDED_KEY = "holosoma_sim"
DOCUMENT_FIELDS = {"version", "backend", "arguments"}

_SUBCOMMAND_TOKEN = re.compile(r"^([A-Za-z_][\w.-]*):[^\s:]+$")


class ConfigDocumentError(ValueError):
    """A structural problem in a run document."""


def expand_config_args(argv: list[str]) -> list[str]:
    """Replace ``--config PATH [--config-path a.b]`` in ``argv`` with the document's tokens.

    Args:
        argv: CLI tokens, without the program name.

    Returns:
        Group selections, then flags, each with the document's before the CLI's. Document
        selections of a group the CLI also selects are dropped.
    """
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--config-path", default=None)
    known, remaining = pre.parse_known_args(argv)
    if known.config is None:
        if known.config_path is not None:
            raise ConfigDocumentError("--config-path requires --config")
        return argv
    try:
        with open(known.config) as f:
            raw = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigDocumentError(f"cannot load {known.config!r}: {exc}") from exc
    doc_selections, doc_flags = _document_args(_select_document(raw, known.config_path))
    cli_selections, cli_rest = _split_selections(remaining)
    shadowed = {_group(t) for t in cli_selections}
    kept = [t for t in doc_selections if _group(t) not in shadowed]
    return [*kept, *cli_selections, *doc_flags, *cli_rest]


def _select_document(raw: Any, path: str | None) -> dict[str, Any]:
    """Return the document mapping at dotted ``path``, or the embedded/root document."""
    if not isinstance(raw, dict):
        raise ConfigDocumentError("document root must be a mapping")
    if path is None:
        embedded = raw.get(EMBEDDED_KEY)
        return embedded if "version" not in raw and isinstance(embedded, dict) else raw
    selected: Any = raw
    for segment in path.split("."):
        if not isinstance(selected, dict) or segment not in selected:
            raise ConfigDocumentError(f"--config-path {path!r}: key {segment!r} not found")
        selected = selected[segment]
    if not isinstance(selected, dict):
        raise ConfigDocumentError(f"--config-path {path!r} must select a mapping")
    return selected


def _document_args(doc: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return a document's group-selection tokens and flag tokens."""
    unknown = set(doc) - DOCUMENT_FIELDS
    if unknown:
        raise ConfigDocumentError(f"unknown document fields: {sorted(unknown)}")
    if doc.get("version") != SUPPORTED_VERSION:
        raise ConfigDocumentError(f"unsupported document version {doc.get('version')!r} (expected {SUPPORTED_VERSION})")
    arguments = doc.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ConfigDocumentError("arguments must be a mapping")
    backend = doc.get("backend")
    if backend is not None:
        if "simulator" in arguments:
            raise ConfigDocumentError("set the simulator with backend or arguments.simulator, not both")
        arguments = {"simulator": backend, **arguments}

    tokens: list[str] = []
    flags: list[str] = []
    for name, value in arguments.items():
        if name == "overrides":
            if not isinstance(value, dict):
                raise ConfigDocumentError("arguments.overrides must map dotted paths to values")
            for path, leaf in value.items():
                flags.extend(_override_args(str(path), leaf))
        elif isinstance(value, str) and value:
            tokens.append(f"{name}:{value}")
        elif isinstance(value, dict) and all(isinstance(v, str) and v for v in value.values()):
            tokens.extend(f"{name}.{key}:{variant}" for key, variant in value.items())
        elif value is not None:
            raise ConfigDocumentError(f"arguments.{name}: expected a preset name or a mapping of keys to presets")
    return tokens, flags


def _override_args(path: str, value: Any) -> list[str]:
    if isinstance(value, dict):
        return [arg for key, sub in value.items() for arg in _override_args(f"{path}.{key}", sub)]
    return [f"--{path}", _encode(value)]


def _encode(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(repr(v) if isinstance(v, str) else _encode(v) for v in value) + "]"
    return str(value)


def _split_selections(argv: list[str]) -> tuple[list[str], list[str]]:
    selections: list[str] = []
    rest: list[str] = []
    after_flag = False
    for token in argv:
        if not after_flag and _SUBCOMMAND_TOKEN.match(token):
            selections.append(token)
        else:
            rest.append(token)
        after_flag = token.startswith("-") and "=" not in token
    return selections, rest


def _group(token: str) -> str:
    return token.split(":", 1)[0].replace("_", "-")
