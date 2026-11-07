"""Inspect actual process dependencies after native scene/sensor initialization."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict


class NativeRuntimeDependencies(TypedDict):
    uses_kit: bool
    kit_libraries: list[str]
    physx_libraries: list[str]
    libraries: list[str]


def assert_native_runtime(*, physics: str, uses_kit: bool) -> NativeRuntimeDependencies:
    from isaaclab.utils.version import has_kit

    assert has_kit() == uses_kit, (uses_kit, has_kit())
    libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in line and "/" in line.split()[-1]
        }
    )
    # Carbonite is also a dependency of standalone OV/importer wheels. The
    # activated image may supply its libcarb.so from the bundled SDK directory;
    # loading that shared framework library does not initialize the Kit app.
    kit_libraries = [
        path
        for path in libraries
        if "omni.kit" in path.lower() or ("/kit/lib" in path.lower() and Path(path).name != "libcarb.so")
    ]
    physx_libraries = [path for path in libraries if "physx" in Path(path).name.lower()]
    if not uses_kit:
        assert "omni.kit.app" not in sys.modules, "Standalone startup imported the Kit app"
        assert not kit_libraries, f"Standalone startup loaded Kit libraries: {kit_libraries}"
        assert "isaaclab_physx.physics.physx_manager" not in sys.modules
        if physics == "newton":
            assert "ovphysx._bindings" not in sys.modules, "Newton loaded OVPhysX native bindings"
            assert not physx_libraries, f"Newton loaded PhysX libraries: {physx_libraries}"
        else:
            assert physx_libraries, "OVPhysX scene did not load its selected physics engine"
    return {
        "uses_kit": uses_kit,
        "kit_libraries": kit_libraries,
        "physx_libraries": physx_libraries,
        "libraries": libraries,
    }
