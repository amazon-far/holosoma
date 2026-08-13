"""Registry helpers for named config presets."""

# NOTE FOR AGENTS/MAINTAINERS: This module is intentionally mirrored with
# holosoma_inference.utils.config_registry. Keep behavior changes in sync unless a difference is
# package-specific.

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
import sys
import warnings
from typing import Any, Callable, TypeVar

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


class ConfigRegistry(dict):
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


def _import_module_from_path(path: str):
    """Import a Python file by path."""
    import importlib.util
    from pathlib import Path

    abspath = Path(path).resolve()
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

        simulator: Annotated[SimulatorConfig, UseRegistry(RUN_SIM_REGISTRY)]
        simulators: dict[str, Annotated[SimulatorConfig, UseRegistry(RUN_SIM_REGISTRY)]]

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
            return bound
    return None


def parse_config(
    config: Any,
    *,
    args: list[str] | None = None,
    default: Any = None,
    **tyro_kwargs: Any,
) -> Any:
    """Parse a config with plugin, preset-file, and dynamic-dict support.

    Recognizes repeatable ``--import-file PATH`` options and top-level
    ``<field>.<key>:<variant>`` dict declarations.

    Args:
        config: Dataclass type, ``Annotated`` alias, or zero-argument factory.
        args: CLI tokens. ``None`` reads from ``sys.argv``.
        default: Optional base config instance.
        tyro_kwargs: Extra keyword arguments forwarded to ``tyro.cli``.

    Returns:
        The parsed config instance.
    """
    from holosoma.utils.tyro_utils import (
        TYRO_CONIFG,
        dynamic_dict_decl_pattern,
        find_dynamic_dict_fields,
        pop_dynamic_dict_args,
        split_import_file_args,
    )

    use_sys_argv = args is None
    argv = sys.argv[1:] if args is None else list(args)

    import_paths, argv = split_import_file_args(argv)
    load_plugins(import_paths)

    # Factories build tyro subcommand aliases from registries after external presets are loaded.
    config_type = _resolve_config(config)

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
    merged_default = _merge_dynamic_default(config_type, default, dynamic_defaults)
    if merged_default is not None:
        kwargs["default"] = merged_default
    if use_sys_argv:
        sys.argv = [sys.argv[0]] + argv
    return tyro.cli(config_type, **kwargs)


def _resolve_config(config: Any) -> Any:
    """Return a config type or alias, resolving factories first."""
    is_factory = callable(config) and not isinstance(config, type) and not hasattr(config, "__metadata__")
    return config() if is_factory else config


def _merge_dynamic_default(config_type: Any, default: Any, dynamic_defaults: dict[str, Any]) -> Any:
    """Return a default config with dynamic-dict fields applied."""
    if not dynamic_defaults:
        return default

    from holosoma.utils.tyro_utils import _strip_annotated

    if default is not None:
        return dataclasses.replace(default, **dynamic_defaults)

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
