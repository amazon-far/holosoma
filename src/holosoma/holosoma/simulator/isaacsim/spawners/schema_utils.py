"""Schema utilities for USD spawners.

This module provides utilities for ensuring USD schemas (APIs) exist before modifying
their properties, handling cases where USD files don't have pre-existing APIs applied.

Problem Statement:
IsaacLab's schemas.modify_*() functions assume that the required USD APIs already exist
in the source files. This works for most IsaacLab assets (like sim_ready assets) but fails
with certain USD files that only have partial APIs applied.

For example:
- SM_* office props only have PhysicsCollisionAPI but no RigidBodyAPI
- Some assets have MeshCollisionAPI but no MassAPI
- Custom USD files may have inconsistent API coverage

When schemas.modify_rigid_body_properties() is called on such assets, it silently fails
because there's no RigidBodyAPI to modify, leading to objects without physics properties.

Solution:
This module provides the ensure_api_and_modify() pattern that:
1. Checks if the required API exists anywhere in the prim subtree
2. If not found, applies the API to the root prim first
3. Then calls the original modify function which now succeeds

This approach preserves IsaacLab's nested API handling while ensuring robustness
for USD files with missing APIs.

Usage Example:
```python
from holosoma.simulator.isaacsim.spawners.schema_utils import ensure_api_and_modify
from isaaclab.sim import schemas
from pxr import UsdPhysics

# Instead of:
# schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)  # May fail silently

# Use:
ensure_api_and_modify(
    prim_path,
    cfg.rigid_props,
    UsdPhysics.RigidBodyAPI,
    schemas.modify_rigid_body_properties
)  # Always succeeds
```
"""

from __future__ import annotations

from typing import Any, Callable

import isaacsim.core.utils.stage as stage_utils
import omni.log
import omni.usd
from pxr import Usd


def has_api_in_subtree(prim_path: str, api_class: type[Any], stage: Usd.Stage | None = None) -> bool:
    """Check if a USD API exists anywhere in the subtree of a prim.

    Recursively searches through the prim hierarchy starting from the specified
    prim path to determine if the given API class is applied to any prim in the subtree.

    Parameters
    ----------
    prim_path : str
        Path to the root prim to check.
    api_class : Type
        The USD API class to check for (e.g., UsdPhysics.RigidBodyAPI).
    stage : Usd.Stage, optional
        USD stage to use. Defaults to current stage if not provided.

    Returns
    -------
    bool
        True if the API is found anywhere in the subtree, False otherwise.
    """
    stage = stage or omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)

    if not root_prim.IsValid():
        return False

    # Check the root prim itself
    if root_prim.HasAPI(api_class):
        omni.log.verbose(f"Found {api_class.__name__} on root prim: {prim_path}")
        return True

    # Recursively check all descendants
    def _check_descendants(prim: Any) -> bool:
        for child in prim.GetChildren():
            if child.HasAPI(api_class):
                omni.log.verbose(f"Found {api_class.__name__} on descendant prim: {child.GetPath()}")
                return True
            if _check_descendants(child):
                return True
        return False

    return _check_descendants(root_prim)


def ensure_api_and_modify(
    prim_path: str,
    cfg: Any,
    api_class: type[Any],
    modify_func: Callable[[str, Any, Usd.Stage], bool],
    stage: Usd.Stage | None = None,
) -> bool:
    """Ensure a USD API exists before modifying its properties.

    This function checks if the specified API exists anywhere in the prim subtree.
    If not found, it applies the API to the root prim first, then calls the modify function.
    This ensures that property modification functions always have the required APIs available.

    Parameters
    ----------
    prim_path : str
        Path to the prim to modify.
    cfg : Any
        Configuration object containing the properties to set.
    api_class : Type
        The USD API class to ensure exists (e.g., UsdPhysics.RigidBodyAPI).
    modify_func : Callable[[str, Any, Usd.Stage], bool]
        The function to call to modify properties (e.g., schemas.modify_rigid_body_properties).
    stage : Usd.Stage, optional
        USD stage to use. Defaults to current stage if not provided.

    Returns
    -------
    bool
        True if properties were successfully modified, False otherwise.
    """
    stage = stage or stage_utils.get_current_stage()

    # Check if API exists anywhere in the subtree
    if not has_api_in_subtree(prim_path, api_class, stage):
        # Apply API to the root prim first
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            api_class.Apply(prim)
            omni.log.info(f"Applied {api_class.__name__} to prim '{prim_path}' before setting properties")
        else:
            omni.log.error(f"Invalid prim at path '{prim_path}', cannot apply {api_class.__name__}")
            return False
    else:
        omni.log.verbose(f"{api_class.__name__} already exists in subtree of {prim_path}")

    # Now call the modify function
    return modify_func(prim_path, cfg, stage)
