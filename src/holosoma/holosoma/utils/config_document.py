"""Expand ``--config PATH`` run documents into CLI tokens.

Document shape (YAML or JSON)::

    version: 1
    task: sim                    # workflow the holosoma command runs
    arguments:
      simulator: sim-mujoco      # <group>:<preset>
      robot: g1-29dof
      sensor:                    # <family>.<key>:<preset>
        scan: g1-pelvis-lidar
      overrides:                 # --<dotted.path>=<value>
        training.headless: true
        robot.init_state.pos: [0.0, 0.0, 0.8]

A file whose root has no ``version`` but has a ``holosoma_sim`` mapping reads that mapping;
``--config-path a.b`` selects any other nested mapping. Tokens on the command line take
precedence over the document. YAML scalars follow YAML 1.2: only ``true``/``false`` are booleans
and dates stay strings. Write tuple-typed or non-empty dict-typed values as quoted literals.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import yaml

SUPPORTED_VERSION = 1
EMBEDDED_KEY = "holosoma_sim"
DOCUMENT_FIELDS = {"version", "task", "arguments"}

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
    selections = [t for t, _ in doc_selections if _group(t) not in shadowed] + cli_selections
    root = [t for t in selections if t.replace("_", "-") in root_selections]
    nested = sorted((t for t in selections if t not in root), key=lambda t: _group(t).count("."))
    return [*root, *nested, *(t for t, _ in doc_flags), *cli_rest]


def alias_task_flags(argv: list[str], task: str | None = None) -> list[str]:
    """Rewrite ``--<task>.<field>`` flags for the selected task to ``--task.<field>``.

    ``task`` defaults to the ``task:<name>`` selection in ``argv``, then ``sim``. A flag naming a
    different task is an error.
    """
    if task is None:
        task = next((t.split(":", 1)[1] for t in argv if t.startswith("task:") and _SUBCOMMAND_TOKEN.match(t)), None)
    selected = (task or "sim").replace("_", "-")
    aliased = []
    for token in argv:
        head = token.partition("=")[0]
        prefix = head[2:].split(".", 1)[0].replace("_", "-") if head.startswith("--") and "." in head else None
        if prefix is not None and prefix == selected:
            aliased.append(f"--task.{token.split('.', 1)[1]}")
            continue
        if prefix is not None and _is_task_name(prefix):
            raise ConfigDocumentError(f"{head} sets an argument of task {prefix}, but task {selected} is selected")
        aliased.append(token)
    return aliased


def select_task(argv: list[str]) -> tuple[str | None, list[str]]:
    """Return the selected task and ``argv`` without its ``task:<name>`` token.

    A ``task:<name>`` token on the command line takes precedence over the ``--config``
    document's ``task`` field. Returns ``None`` when neither selects a task.
    """
    task: str | None = None
    remaining: list[str] = []
    expects_value = False
    for token in argv:
        if not expects_value and token.startswith("task:") and _SUBCOMMAND_TOKEN.match(token):
            if task is not None:
                raise ConfigDocumentError("task: given more than once")
            task = token.split(":", 1)[1]
        else:
            remaining.append(token)
        expects_value = (
            not expects_value and token.startswith("--") and "=" not in token and token not in _VALUELESS_FLAGS
        )
    if task is not None:
        return task, remaining
    spec, path, _ = _pop_config_options(argv)
    if spec is None:
        return None, remaining
    try:
        doc = _select_document(_load(spec), path)
        _document_args(doc)
    except ConfigDocumentError as exc:
        raise ConfigDocumentError(f"--config {spec!r}: {exc}") from exc
    return doc.get("task"), remaining


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
    unknown = set(doc) - DOCUMENT_FIELDS
    if unknown:
        raise ConfigDocumentError(f"unknown document fields: {sorted(map(str, unknown))}")
    task = doc.get("task")
    if task is not None and not (isinstance(task, str) and task):
        raise ConfigDocumentError("task must be a task name")
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


def _is_task_name(key: Any) -> bool:
    from holosoma.config_values.task import TASK_REGISTRY

    names = {name.replace("_", "-") for name in TASK_REGISTRY}
    return str(key).replace("_", "-") in names


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
