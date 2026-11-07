"""Select the installed standalone importer namespace without bootstrapping Kit."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path


def select_standalone_importers() -> None:
    """Expose the real importer wheel when Kit's regular package shadows it.

    ``isaacsim-asset-isolated`` contributes a PEP 420 ``isaacsim.asset`` namespace.
    Kit's ``isaacsim/__init__.py`` overrides namespace portions and bootstraps its
    kernel in pip installations. Choose the standalone wheel's namespace before
    importing any ``isaacsim`` module; its actual importer implementation and
    dependencies still load through Python's normal import machinery.
    """
    existing = sys.modules.get("isaacsim")
    if existing is not None:
        if getattr(existing, "__file__", None) is not None:
            raise RuntimeError("Select OVRTX before importing the Kit isaacsim package.")
    else:
        distribution = importlib.metadata.distribution("isaacsim-asset-isolated")
        namespace_path = Path(str(distribution.locate_file("isaacsim")))
        if not (namespace_path / "asset/importer/urdf/__init__.py").is_file():
            raise RuntimeError("Standalone Isaac Sim importers are missing; rerun scripts/setup_isaacsim3.sh.")
        spec = ModuleSpec("isaacsim", loader=None, is_package=True)
        spec.submodule_search_locations = [str(namespace_path)]
        sys.modules["isaacsim"] = importlib.util.module_from_spec(spec)

    # The standalone URDF importer authors PhysX joint schemas even for
    # Newton. Load their actual codeless provider before USD creates a stage;
    # this registers schemas without creating an OVPhysX engine.
    import ovphysx
    from pxr import Plug

    registry = Plug.Registry()
    registered = {plugin.name.casefold() for plugin in registry.GetAllPlugins()}
    paths = [str(path) for path in ovphysx.codeless_schema_paths() if path.parent.name.casefold() not in registered]
    if paths:
        registry.RegisterPlugins(paths)
