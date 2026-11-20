"""Expand ``--config PATH`` run documents into CLI tokens.

Document shape (YAML or JSON)::

    version: 1
    arguments:
      exp: g1-29dof               # exp:<preset>, the base config; optional
      simulator: mjwarp           # <path>:<preset>
      sensor:                     # <path>.<key>:<preset>
        scan: g1-pelvis-lidar
      overrides:                  # --<dotted.path>=<value>
        run.num_envs: 64
        task: holosoma.envs.simulation_task:SimulationTask.simulate

A file whose root has no ``version`` but has a ``holosoma_sim`` mapping reads that mapping;
``--config-path a.b`` selects any other nested mapping. A selection applies before the selections
and values below its path, and tokens on the command line take precedence over the document's at
the same path. YAML scalars follow YAML 1.2: only ``true``/``false`` are booleans and dates stay
strings. Write tuple-typed or non-empty dict-typed values as quoted literals.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_VERSION = 1
EMBEDDED_KEY = "holosoma_sim"
DOCUMENT_FIELDS = {"version", "arguments"}

_SUBCOMMAND_TOKEN = re.compile(r"^[A-Za-z_][\w.-]*:[^\s:]+$")
_VALUELESS_FLAGS = {"--help"}


class ConfigDocumentError(ValueError):
    """A structural problem in a run document or its ``--config`` options."""


class _Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 core-schema scalars that rejects duplicate mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        mapping: dict[Any, Any] = super().construct_mapping(node, deep)
        return mapping

    def construct_core_int(self, node: yaml.ScalarNode) -> int:
        value = str(self.construct_scalar(node))
        try:
            return int(value, 0)
        except ValueError:
            return int(value, 10)  # leading zeros are decimal in YAML 1.2


_RETYPED = {"tag:yaml.org,2002:bool", "tag:yaml.org,2002:int", "tag:yaml.org,2002:float", "tag:yaml.org,2002:timestamp"}
_Loader.yaml_implicit_resolvers = {
    char: [(tag, regex) for tag, regex in resolvers if tag not in _RETYPED]
    for char, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:int", re.compile(r"^[-+]?(?:[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$"), list("-+0123456789")
)
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)
_Loader.add_constructor("tag:yaml.org,2002:int", _Loader.construct_core_int)


def expand_config_args(
    argv: list[str],
    root_selections: frozenset[str] = frozenset(),
    provenance: dict[str, str] | None = None,
) -> list[str]:
    """Replace ``--config PATH [--config-path a.b]`` in ``argv`` with the document's tokens.

    Args:
        argv: CLI tokens, without the program name.
        root_selections: Root subcommand tokens (e.g. ``exp:<name>``), hyphenated; emitted first.
        provenance: Filled with each document token mapped to the file and key it came from.

    Returns:
        Root selection, then group selections parents-first, then document flags, then the
        remaining CLI tokens. Document selections of a group the CLI also selects are dropped.
    """
    spec, path, remaining = _pop_config_options(argv)
    if spec is None:
        return argv
    cli_selections, cli_rest = _split_selections(remaining)
    try:
        doc_selections, doc_flags = _document_args(_select_document(_load(spec), path))
    except ConfigDocumentError as exc:
        raise ConfigDocumentError(f"--config {spec!r}: {exc}") from exc
    if provenance is not None:
        provenance.update({token: f"{spec}: {key}" for token, key in [*doc_selections, *doc_flags]})
    shadowed = {_group(t) for t in cli_selections}
    for token, _ in doc_selections:
        replacement = next((t for t in cli_selections if t.replace("_", "-") in root_selections), None)
        if token.replace("_", "-") in root_selections and replacement is not None and replacement != token:
            _warn_once(
                "exp",
                f"{replacement} replaces the document's {token}; the document's other selections and overrides "
                "still apply on top of it.",
            )
    selections = [t for t, _ in doc_selections if _group(t) not in shadowed] + cli_selections
    root = [t for t in selections if t.replace("_", "-") in root_selections]
    nested = sorted((t for t in selections if t not in root), key=lambda t: _group(t).count("."))
    return [*root, *nested, *(t for t, _ in doc_flags), *cli_rest]


# Fields that moved, by old dotted path (underscored) -> new dotted path. Paths under a moved
# prefix move with it (``training.seed`` -> ``run.seed``, ``logger.video.enabled`` ->
# ``run.logger.video.enabled``).
MOVED_FLAGS = {
    **{f"training.{name}": f"run.{name}" for name in ("num_envs", "headless", "seed", "torch_deterministic")},
    **{f"training.{name}": f"run.{name}" for name in ("project", "name")},
    "training.checkpoint": "algo.resume_from",
    **{f"training.{name}": f"algo.{name}" for name in ("teacher_checkpoint", "multigpu")},
    **{f"training.{name}": f"eval.{name}" for name in ("max_eval_steps", "export_onnx")},
    "checkpoint": "eval.checkpoint",
    **{name: f"run.{name}" for name in ("device", "time_scale", "viewer_update_decimation", "viewer_dt")},
    "logger.project": "run.project",
    "logger.name": "run.logger.display_name",
    "logger": "run.logger",
    "nightly": "algo.nightly",
    "eval_overrides.headless": "run.headless",
    "eval_overrides.num_envs": "run.num_envs",
    "eval_overrides.max_episode_length_s": "simulator.config.sim.max_episode_length_s",
    "eval_overrides.randomize_tiles": "terrain.terrain_term.spawn.randomize_tiles",
    "eval_overrides.xy_offset_range": "terrain.terrain_term.spawn.xy_offset_range",
}
# Flags that load but no longer do anything, with the reason.
REMOVED_FLAGS = {
    "eval_overrides.disable_logger": "evaluation always runs with the logger off; select run.logger: to change it",
}


def _moved(path: str, moves: dict[str, str]) -> str | None:
    """``path`` under its longest moved prefix, or None when it did not move."""
    for old in sorted(moves, key=len, reverse=True):
        if path == old or path.startswith(f"{old}."):
            return moves[old] + path[len(old) :]
    return None


def alias_legacy_flags(argv: list[str]) -> list[str]:
    """Rewrite flags and selections whose field moved, and drop removed flags, warning once each.

    ``--env_class=X`` becomes ``--task=X.train`` with ``X`` written as ``module:Class``.
    """
    moves = MOVED_FLAGS
    aliased: list[str] = []
    tokens = iter(argv)
    for token in tokens:
        head, eq, value = token.partition("=")
        if head.startswith("--"):
            path = head[2:].replace("-", "_")
            removed = next((key for key in REMOVED_FLAGS if path == key or path.startswith(f"{key}.")), None)
            if removed is not None:
                _warn_once(head, f"{head} has no effect and is ignored: {REMOVED_FLAGS[removed]}.")
                if not eq:
                    next(tokens, None)
                continue
            if path == "env_class":
                value = value if eq else next(tokens, "")
                module, _, cls = value.rpartition(".")
                _warn_once(head, f"{head} moved to --task=<module>:<Class>.<function>; the old spelling still works.")
                aliased.append(f"--task={module}:{cls}.train")
                continue
            new = _moved(path, moves)
            if new is not None:
                _warn_once(head, f"{head} moved to --{new}; the old spelling still works.")
                aliased.append(f"--{new}{eq}{value}")
                continue
        elif _SUBCOMMAND_TOKEN.match(token) and not token.startswith("exp:"):
            name, _, preset = token.partition(":")
            new = _moved(name.replace("-", "_"), moves)
            if new is not None:
                _warn_once(name, f"{name}:<preset> moved to {new}:<preset>; the old spelling still works.")
                aliased.append(f"{new}:{preset}")
                continue
        aliased.append(token)
    return aliased


_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    from loguru import logger

    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


def _pop_config_options(argv: list[str]) -> tuple[str | None, str | None, list[str]]:
    found: dict[str, str] = {}
    remaining: list[str] = []
    tokens = iter(argv)
    for token in tokens:
        name, eq, value = token.partition("=")
        if name not in ("--config", "--config-path"):
            remaining.append(token)
            continue
        if not eq:
            value = next(tokens, "")
        if not value or value.startswith("--"):
            raise ConfigDocumentError(f"{name} needs a value")
        if name in found:
            raise ConfigDocumentError(f"{name} given more than once")
        found[name] = value
    if "--config-path" in found and "--config" not in found:
        raise ConfigDocumentError("--config-path requires --config")
    return found.get("--config"), found.get("--config-path"), remaining


def _load(spec: str) -> Any:
    from holosoma.utils.path import resolve_path

    try:
        local = resolve_path(spec)
        with open(local, "rb") as f:
            content = f.read()
        if local.lower().endswith(".json"):
            return json.loads(content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        return yaml.load(content, Loader=_Loader)  # noqa: S506 - _Loader is a SafeLoader
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        raise ConfigDocumentError(f"cannot load: {exc}") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _select_document(raw: Any, path: str | None) -> dict[str, Any]:
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


def _document_args(doc: dict[str, Any]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return a document's group-selection and flag tokens, each with the key it came from."""
    if doc.get("version") != SUPPORTED_VERSION:
        raise ConfigDocumentError(f"unsupported document version {doc.get('version')!r} (expected {SUPPORTED_VERSION})")
    if "task" in doc:
        raise ConfigDocumentError("task is a field like any other: write it as `arguments: {overrides: {task: ...}}`")
    unknown = set(doc) - DOCUMENT_FIELDS
    if unknown:
        raise ConfigDocumentError(f"unknown document fields: {sorted(map(str, unknown))}")
    arguments = doc.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ConfigDocumentError("arguments must be a mapping")

    selections: list[tuple[str, str]] = []
    flags: list[tuple[str, str]] = []
    for name, value in arguments.items():
        if name == "overrides":
            if not isinstance(value, dict):
                raise ConfigDocumentError("arguments.overrides must map dotted paths to values")
            for path, leaf in value.items():
                flags.extend(_override_args(str(path), leaf, f"arguments.overrides.{path}"))
        elif isinstance(value, str) and value:
            selections.append((f"{name}:{value}", f"arguments.{name}"))
        elif isinstance(value, dict) and all(isinstance(v, str) and v for v in value.values()):
            selections.extend((f"{name}.{key}:{variant}", f"arguments.{name}.{key}") for key, variant in value.items())
        elif value is not None:
            raise ConfigDocumentError(f"arguments.{name}: expected a preset name or a mapping of keys to presets")
    return selections, flags


def _override_args(path: str, value: Any, source: str) -> list[tuple[str, str]]:
    if isinstance(value, dict) and value:
        return [pair for key, sub in value.items() for pair in _override_args(f"{path}.{key}", sub, f"{source}.{key}")]
    return [(f"--{path}={_encode(value, path)}", source)]


def _encode(value: Any, path: str) -> str:
    if isinstance(value, dict):
        return "{}"
    if isinstance(value, list):
        return _literal(value, path)
    return str(value)


def _literal(value: Any, path: str) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_literal(v, path) for v in value) + "]"
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ConfigDocumentError(f"overrides.{path}: nan cannot be written inside a list")
        return "1e999" if value > 0 else "-1e999"
    return repr(value)


def _split_selections(argv: list[str]) -> tuple[list[str], list[str]]:
    selections: list[str] = []
    rest: list[str] = []
    expects_value = False
    for token in argv:
        if not expects_value and _SUBCOMMAND_TOKEN.match(token):
            selections.append(token)
        else:
            rest.append(token)
        expects_value = (
            not expects_value and token.startswith("--") and "=" not in token and token not in _VALUELESS_FLAGS
        )
    return selections, rest


def _group(token: str) -> str:
    return token.split(":", 1)[0].replace("_", "-")


def args_to_document(argv: list[str]) -> dict[str, Any]:
    """Return the run document that expands to ``argv`` (CLI tokens, already expanded).

    Selections become ``arguments`` entries and flags become ``overrides``. Values are written as
    YAML scalars or lists when that round-trips to the same token, and as strings otherwise.
    """
    arguments: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    tokens = iter(argv)
    for token in tokens:
        if token.startswith("--"):
            head, eq, value = token.partition("=")
            if not eq:
                value = next(tokens, "")
            overrides[head[2:]] = _document_value(value)
        elif _SUBCOMMAND_TOKEN.match(token):
            name, _, preset = token.partition(":")
            arguments[name] = preset
    if overrides:
        arguments["overrides"] = overrides
    return {"version": SUPPORTED_VERSION, "arguments": arguments}


class _Dumper(yaml.SafeDumper):
    """Block mappings with lists written inline, the way they read on the command line."""


_Dumper.add_representer(
    list, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
)


def dump_document(document: dict[str, Any]) -> str:
    """Serialize a run document as YAML."""
    return str(yaml.dump(document, Dumper=_Dumper, sort_keys=False, default_flow_style=False))


def _document_value(text: str) -> Any:
    """``text`` as the YAML value that encodes back to the same token, else the string itself."""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return text
    if isinstance(value, (bool, int, float, list)) or value is None or value == {}:
        return value if _encode(value, "") == text else text
    return text


def write_document(path: str | Path, argv: list[str]) -> None:
    """Write the run document for ``argv`` to ``path``; ``holosoma --config <path>`` re-runs it."""
    text = dump_document(args_to_document(argv))
    Path(path).write_text(f"# Re-run with: holosoma --config {Path(path).name}\n{text}")
