"""Registry helpers for named config presets."""

# NOTE FOR AGENTS/MAINTAINERS: This module is intentionally mirrored with
# holosoma_inference.utils.config_registry. Keep behavior changes in sync unless a difference is
# package-specific.

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
import sys
import typing
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any, TypeVar, cast

import tyro
from loguru import logger

from holosoma.utils.pycompat import entry_points

T = TypeVar("T")


def deprecated_defaults_alias(module_name: str, registry: ConfigRegistry) -> Callable[[str], Any]:
    """Return a module ``__getattr__`` handler for deprecated ``DEFAULTS`` access."""

    def __getattr__(name: str) -> Any:  # noqa: N807 - PEP 562 module-level hook
        if name == "DEFAULTS":
            warnings.warn(
                f"{module_name}.DEFAULTS is deprecated; use the module's *_REGISTRY name instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return registry
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__


# Registry modules append instances here when imported. load_plugins() imports config value modules
# before loading external presets so this list contains every built-in registry.
ALL_REGISTRIES: list[ConfigRegistry] = []


class ConfigRegistry(dict[str, Any]):
    """Mapping of preset names to config values for one config family.

    Args:
        config_type: Accepted config class or tuple of classes.
        group: Entry-point group for external presets.
    """

    def __init__(self, config_type: type | tuple[type, ...], group: str | None = None) -> None:
        super().__init__()
        self.config_type = config_type
        self.group = group
        ALL_REGISTRIES.append(self)

    @property
    def config_types(self) -> tuple[type, ...]:
        """Return accepted config classes as a tuple."""
        return self.config_type if isinstance(self.config_type, tuple) else (self.config_type,)

    def add(self, name: str, value: T) -> T:
        """Register ``value`` under ``name`` and return it."""
        if value is not None and not isinstance(value, self.config_type):
            expected = getattr(self.config_type, "__name__", self.config_type)
            raise TypeError(f"config preset {name!r}: expected {expected}, got {type(value).__name__}")
        self[name] = value
        return value

    def ingest(self, name: str, value: Any, *, source: str = "") -> bool:
        """Register an external preset when it matches this registry's config type."""
        if value is not None and not isinstance(value, self.config_type):
            logger.debug(f"Skipping config preset '{name}' from {source or '?'}: {type(value).__name__} type mismatch.")
            return False
        if name in self and self[name] is not value:
            logger.warning(f"Config preset '{name}' from {source or '?'} overrides existing entry.")
        self[name] = value
        return True


# Loaded entry-point groups are keyed by registry object to allow packages to reuse group names.
# A group is recorded only after every entry point in it loads successfully.
_loaded_groups: set[tuple[str, int]] = set()

# Imported preset files are tracked by absolute path to avoid rerunning module-level registrations.
_loaded_files: set[str] = set()


def _discover_registries() -> None:
    """Import config value modules so their registries exist."""
    from holosoma import config_values

    for info in pkgutil.iter_modules(config_values.__path__):
        # Leaf modules import any package-specific presets they need.
        if info.ispkg or info.name == "registry":
            continue
        importlib.import_module(f"{config_values.__name__}.{info.name}")


def load_entrypoint_presets(registry: ConfigRegistry) -> None:
    """Load entry-point presets into ``registry``."""
    group = registry.group
    if group is None:
        return
    load_key = (group, id(registry))
    if load_key in _loaded_groups:
        return

    loaded_ok = True
    for ep in entry_points(group=group):
        try:
            value = ep.load()
        except Exception as exc:  # one bad plugin must not break the menu
            loaded_ok = False
            logger.warning(f"Skipping config preset '{ep.name}' from {ep.value!r}: failed to load ({exc}).")
            continue
        registry.ingest(ep.name, value, source=repr(ep.value))

    if loaded_ok:
        _loaded_groups.add(load_key)


def load_file_presets(paths: list[str]) -> None:
    """Import Python files that call registry ``add()`` methods."""
    for path in paths:
        _import_module_from_path(path)


def _import_module_from_path(path: str) -> ModuleType | None:
    """Import a Python file by path."""
    import importlib.util
    from pathlib import Path

    from holosoma.utils.path import resolve_path

    abspath = Path(resolve_path(path))
    if not abspath.is_file():
        raise FileNotFoundError(f"--import-file path does not exist: {path}")
    key = str(abspath)
    if key in _loaded_files:
        return sys.modules.get(f"_holosoma_config_file_{abs(hash(abspath))}")

    # Unique synthetic name so same-basename files don't collide in sys.modules.
    mod_name = f"_holosoma_config_file_{abs(hash(abspath))}"
    spec = importlib.util.spec_from_file_location(mod_name, key)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config file as a module: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(mod_name)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = previous
        raise
    _loaded_files.add(key)
    return module


def load_plugins(import_file_paths: list[str] | None = None) -> None:
    """Load built-in registries, entry-point presets, and optional preset files."""
    _discover_registries()
    for registry in ALL_REGISTRIES:
        load_entrypoint_presets(registry)
    if import_file_paths:
        load_file_presets(import_file_paths)


# Attribute stamped onto the tyro arg returned by UseRegistry so the dynamic-dict pre-parse can
# recover the bound registry from a field's Annotated metadata.
_REGISTRY_ATTR = "_holosoma_config_registry"


def UseRegistry(registry: ConfigRegistry) -> Any:  # noqa: N802 - annotation marker, reads as a type constructor
    """Bind a config field to ``registry``. Use as ``Annotated`` metadata in either position::

        simulator: Annotated[SimulatorConfig, UseRegistry(SIMULATOR_REGISTRY)]
        simulators: dict[str, Annotated[SimulatorConfig, UseRegistry(SIMULATOR_REGISTRY)]]

    A scalar field gets a tyro subcommand menu built from the registry's current contents. A
    dynamic ``dict[str, X]`` field is populated by ``parse_config`` from ``<field>.<key>:<variant>``
    tokens against the same registry. Under ``from __future__ import annotations`` the marker is
    rebuilt per parse (after ``load_plugins``), so entry-point and ``--import-file`` presets appear.
    """
    marker = tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(registry))
    object.__setattr__(marker, _REGISTRY_ATTR, registry)
    return marker


def registry_from_annotated(hint: Any) -> ConfigRegistry | None:
    """Return the registry bound to ``hint`` by ``UseRegistry``, or ``None`` if unmarked."""
    for meta in getattr(hint, "__metadata__", ()):
        bound = getattr(meta, _REGISTRY_ATTR, None)
        if bound is not None:
            return cast("ConfigRegistry", bound)
    return None


def parse_config(
    config: Any,
    *,
    args: list[str] | None = None,
    default: Any = None,
    record: list[str] | None = None,
    **tyro_kwargs: Any,
) -> Any:
    """Parse a config with plugin, preset-file, and dynamic-dict support.

    Recognizes repeatable ``--import-file PATH`` options, one ``--config PATH [--config-path a.b]``
    run document (see :mod:`holosoma.utils.config_document`), ``--<task>.<field>`` flags for the
    selected task, ``--print-args`` (print the equivalent command line and exit), ``--to-config``
    (print the equivalent run document and exit), and top-level ``<field>.<key>:<variant>`` dict
    declarations.

    Args:
        config: Dataclass type, ``Annotated`` alias, or zero-argument factory.
        args: CLI tokens. ``None`` reads from ``sys.argv``.
        default: Optional base config instance.
        record: Filled with the final CLI tokens (documents expanded, aliases applied).
        tyro_kwargs: Extra keyword arguments forwarded to ``tyro.cli``.

    Returns:
        The parsed config instance.
    """
    from holosoma.utils.config_document import (
        ConfigDocumentError,
        alias_legacy_flags,
        alias_task_flags,
        expand_config_args,
    )
    from holosoma.utils.tyro_utils import (
        TYRO_CONIFG,
        dynamic_dict_decl_pattern,
        find_dynamic_dict_fields,
        pop_dynamic_dict_args,
        split_import_file_args,
    )

    use_sys_argv = args is None
    argv = sys.argv[1:] if args is None else list(args)
    print_args = "--print-args" in argv
    to_config = "--to-config" in argv
    argv = [t for t in argv if t not in ("--print-args", "--to-config")]

    import_paths, argv = split_import_file_args(argv)
    load_plugins(import_paths)

    # Factories build tyro subcommand aliases from registries after external presets are loaded.
    config_type = _resolve_config(config)
    provenance: dict[str, str] = {}
    try:
        expanded = expand_config_args(argv, frozenset(_root_subcommands(config_type)), provenance)
        task = next((t.split(":", 1)[1] for t in expanded if t.startswith("task:")), None) or _default_task(config_type)
        argv = alias_task_flags(alias_legacy_flags(expanded), task)
        provenance = {
            alias_task_flags(renamed, task)[0]: source
            for token, source in provenance.items()
            if (renamed := alias_legacy_flags([token]))
        }
    except ConfigDocumentError as exc:
        raise SystemExit(str(exc)) from exc
    if print_args:
        import shlex

        print(shlex.join(["holosoma", *argv]))
        raise SystemExit(0)
    if to_config:
        from holosoma.utils.config_document import args_to_document, dump_document

        print(dump_document(args_to_document(argv)), end="")
        raise SystemExit(0)
    if record is not None:
        record[:] = argv
    _reject_legacy_mujoco_cli(argv)

    dynamic_defaults: dict[str, dict[str, Any]] = {}
    for field_name, value_hint in find_dynamic_dict_fields(config_type).items():
        registry = registry_from_annotated(value_hint)
        if registry is None:
            continue  # dict field without a UseRegistry marker stays an ordinary (empty) dict
        built = pop_dynamic_dict_args(field_name, registry, argv=argv)
        pattern = dynamic_dict_decl_pattern(field_name)
        argv = [t for t in argv if not pattern.match(t)]
        if built:
            dynamic_defaults[field_name] = built

    kwargs: dict[str, Any] = {"config": TYRO_CONIFG, "args": argv, **tyro_kwargs}
    merged_default = _merge_dynamic_default(config_type, default, dynamic_defaults, argv)
    if merged_default is not None:
        kwargs["default"] = merged_default
    if use_sys_argv:
        sys.argv = [sys.argv[0]] + argv
    if not provenance:
        return tyro.cli(config_type, **kwargs)
    return _cli_with_provenance(config_type, kwargs, provenance)


def _cli_with_provenance(config_type: Any, kwargs: dict[str, Any], provenance: dict[str, str]) -> Any:
    """Run ``tyro.cli``; on a usage error, name the document keys behind the tokens it rejected."""
    import contextlib
    import io

    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            result = tyro.cli(config_type, **kwargs)
    except SystemExit as exc:
        message = captured.getvalue()
        sys.stderr.write(message)
        if exc.code not in (0, None):
            flat = _normalize_cli_text(message)
            sources = [
                f"  {token.partition('=')[0]}  <- {source}"
                for token, source in provenance.items()
                if _normalize_cli_text(token.partition("=")[0]) in flat
            ]
            if sources:
                sys.stderr.write("From --config:\n" + "\n".join(sources) + "\n")
        raise
    sys.stderr.write(captured.getvalue())
    return result


def _normalize_cli_text(text: str) -> str:
    """Fold tyro's hyphen/underscore spellings and its box-drawing line wraps for matching."""
    return "".join(text.replace("_", "-").split()).replace("│", "")


def _default_task(config_type: Any) -> str | None:
    """Registry name of ``config_type``'s default ``task``, or ``None`` if it has no task field."""
    from holosoma.config_values.task import TASK_REGISTRY
    from holosoma.utils.tyro_utils import _strip_annotated

    base = _strip_annotated(config_type)
    if not dataclasses.is_dataclass(base):
        return None
    field = next((f for f in dataclasses.fields(base) if f.name == "task"), None)
    if field is None:
        return None
    return next((name for name, preset in TASK_REGISTRY.items() if preset == field.default), None)


def _reject_legacy_mujoco_cli(argv: list[str]) -> None:
    """Explain relocated CLI fields before Tyro reports an unknown argument."""
    aliases = (
        "--simulator.config.mujoco-warp.",
        "--simulator.config.mujoco_warp.",
    )
    canonical = "--simulator.config.sim.mujoco.warp."
    if any(token.startswith(aliases) for token in argv):
        raise SystemExit(
            f"--simulator.config.mujoco-warp.* moved to {canonical}*. "
            "Move the flags to that path; capacity field names and defaults are unchanged."
        )


def _resolve_config(config: Any) -> Any:
    """Return a config type or alias, resolving factories first."""
    is_factory = callable(config) and not isinstance(config, type) and not hasattr(config, "__metadata__")
    return config() if is_factory else config


def _root_subcommands(config_type: Any) -> dict[str, Any]:
    """Return root-level subcommand names (hyphenated) mapped to their preset defaults.

    A root alias like ``Annotated[ExperimentConfig, tyro.conf.arg(constructor=
    subcommand_type_from_defaults({'exp:name': preset, ...}))]`` carries each preset as
    the ``default`` of a ``tyro.conf.subcommand`` on the union members. Returns an empty
    mapping when the root is not such an alias.
    """
    found: dict[str, Any] = {}
    for marker in getattr(config_type, "__metadata__", ()):
        factory = getattr(marker, "constructor_factory", None)
        if factory is None:
            continue
        try:
            union = factory()
        except Exception as exc:  # malformed alias: fall back to base() below
            logger.debug(f"Ignoring unresolvable root-subcommand alias: {exc}")
            continue
        for member in typing.get_args(union):
            for sub in getattr(member, "__metadata__", ()):
                name = getattr(sub, "name", None)
                if isinstance(name, str):
                    found.setdefault(name.replace("_", "-"), getattr(sub, "default", None))
    return found


def _root_subcommand_default(config_type: Any, argv: list[str]) -> Any:
    """Return the preset selected by a root-level subcommand token in ``argv``, if any.

    When argv picks one (e.g. ``exp:g1-29dof-wbt``), that preset — not a bare instance — is
    the correct base for dynamic-dict injection. Tokens match with tyro's hyphen/underscore
    equivalence. Returns None when the root is not an alias or no token matches.
    """
    tokens = {t.replace("_", "-") for t in argv if not t.startswith("-")}
    for name, preset in _root_subcommands(config_type).items():
        if name in tokens:
            return preset
    return None


def _merge_dynamic_default(config_type: Any, default: Any, dynamic_defaults: dict[str, Any], argv: list[str]) -> Any:
    """Return a default config with dynamic-dict fields applied."""
    if not dynamic_defaults:
        return default

    from holosoma.utils.tyro_utils import _strip_annotated

    if default is not None:
        return dataclasses.replace(default, **dynamic_defaults)

    # Root subcommand selected on the CLI: seed from ITS preset. A bare base() here
    # would silently override every field of the selected preset (tyro applies the
    # explicit default over the subcommand's), turning e.g. `exp:g1-29dof-wbt
    # sensor.head:g1-head` into a default config with one sensor.
    selected = _root_subcommand_default(config_type, argv)
    if selected is not None:
        return dataclasses.replace(selected, **dynamic_defaults)

    base = _strip_annotated(config_type)
    try:
        instance = base()
    except TypeError as exc:
        raise TypeError(
            f"parse_config cannot inject dynamic dict keys {sorted(dynamic_defaults)} into "
            f"{getattr(base, '__name__', base)!r}: it has required fields, so no base instance can "
            f"be built. Pass an explicit default= to parse_config."
        ) from exc
    return dataclasses.replace(instance, **dynamic_defaults)
