"""Prepare data-only USD schemas before standalone simulation stage creation."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path


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
