"""Keep the real standalone URDF importer separate from Kit's Python package.

Our combined runtime exposes both Kit's regular ``isaacsim`` package and the
isolated importer's namespace wheel. Kit activation makes the regular package
shadow that namespace; pip installations also bootstrap Kit on root import.
This compatibility adapter selects the standalone namespace and its data-only
USD schemas before SDK launch. A clean standalone environment does not need the
namespace override. No importer implementation or physics bindings are replaced.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
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
        if vars(existing).get("__file__") is not None:
            raise RuntimeError("Select the standalone runtime before importing the Kit isaacsim package.")
    else:
        distribution = importlib.metadata.distribution("isaacsim-asset-isolated")
        namespace_path = Path(str(distribution.locate_file("isaacsim")))
        if not (namespace_path / "asset/importer/urdf/__init__.py").is_file():
            raise RuntimeError("Standalone Isaac Sim importers are missing; rerun scripts/setup_isaacsim6.sh.")
        spec = ModuleSpec("isaacsim", loader=None, is_package=True)
        spec.submodule_search_locations = [str(namespace_path)]
        sys.modules["isaacsim"] = importlib.util.module_from_spec(spec)

    # Binary-image activation adds Kit's kernel, bindings and extension paths
    # to sys.path. The importer wheel probes for omni.ext and, when those paths
    # remain visible, imports omni.kit.app even without a running Kit app.
    # Standalone wheels provide their own dependencies; keep Kit's bundled
    # Python modules out of this process while retaining the real asset namespace.
    roots = [Path(path).resolve() for path in sys.modules["isaacsim"].__path__]
    roots.extend(
        Path(path).resolve() for name in ("ISAACSIM_ROOT_PATH", "ISAAC_PATH") if (path := os.environ.get(name))
    )
    runtime_paths = [
        root / relative for root in roots for relative in ("kit/kernel", "kit/plugins", "exts", "extscache")
    ]
    # The image's venv uses the bundled interpreter: keep its Python standard
    # library under kit/python, which is independent of the Kit runtime.
    sys.path[:] = [
        path for path in sys.path if not any(Path(path).resolve().is_relative_to(root) for root in runtime_paths)
    ]


def register_standalone_schemas() -> None:
    """Register the full PhysX schema catalog before standalone stage creation.

    The Isaac Sim 6.2 URDF importer authors PhysX joint attributes even when
    Newton supplies physics. Creating a typed USD physics scene before the
    importer's schema initialization can make conversion fail with an unknown
    ``PhysxJointAPI``. Register these resources before SDK launch initializes
    the scene; schema metadata does not load the PhysX engine.

    The importer supplies minimal fallback schemas when the full catalog is
    absent. Registering that fallback first conflicts with OVPhysX's later full
    catalog, which declares the same types. Our combined installation includes
    OVPhysX, so registering its codeless resources first gives both standalone
    engines the same catalog without importing OVPhysX's native bindings.
    """
    from pxr import Plug

    # The standalone URDF importer authors PhysX joint schemas even for
    # Newton. Locate their codeless resources through wheel metadata, so schema
    # registration cannot initialize the OVPhysX native bindings.
    distribution = importlib.metadata.distribution("ovphysx")
    schema_root = Path(str(distribution.locate_file("ovphysx/schemas/physx")))
    resources = sorted(schema_root.glob("*/resources/plugInfo.json"))
    if not resources:
        raise RuntimeError("Standalone PhysX USD schemas are missing; rerun scripts/setup_isaacsim6.sh.")
    registry = Plug.Registry()
    registered = {plugin.name.casefold() for plugin in registry.GetAllPlugins()}
    paths = [str(path.parent) for path in resources if path.parent.parent.name.casefold() not in registered]
    if paths:
        registry.RegisterPlugins(paths)
