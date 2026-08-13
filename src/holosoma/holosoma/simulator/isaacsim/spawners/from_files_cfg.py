# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from Isaac Lab v2.0.0 (https://github.com/isaac-sim/IsaacLab)
# Contributors: https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md
#
# This file contains utilities adapted from Isaac Lab's spawner code.

"""Configuration for custom USD file spawners.

This module provides custom USD file configuration that supports source prim path filtering.
"""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from . import from_files


@configclass
class CustomUsdFileCfg(UsdFileCfg):
    """Custom USD file configuration for spawning assets with source prim path filtering.

    Extends IsaacLab's UsdFileCfg to add source prim path filtering capability,
    allowing loading of specific prims from USD files instead of the entire file.
    This is essential for the USD scene loader functionality.

    Parameters
    ----------
    source_path : str or None, optional
        The prim path within the USD file to load. Defaults to None (reference the whole file
        at its defaultPrim). Example: "/Scene/Items/Apple" to load only the Apple prim.
    disable_instanceable : bool, optional
        Whether to disable instanceable flag before applying physics properties.
        Defaults to False. Set to True to avoid USD instancing conflicts (required for assets whose
        physics is edited at spawn: descendant RemoveAPI/stamps no-op on instance proxies).
    physics_material : RigidBodyMaterialCfg, optional
        Rigid-body physics material (friction/restitution) spawned and bound to the asset's
        collider(s) at spawn time, so PhysX applies it when the body is first initialized.
        Defaults to None (the asset keeps its authored material).
    validate_composition : bool, optional
        After spawn, assert the reference composed renderable geometry and has no dangling material
        bindings, raising loud instead of spawning a silently-invisible asset. Defaults to True; set
        False for a legitimately geometry-less asset.
    """

    func: Callable = from_files.spawn_from_usd
    source_path: str | None = None
    disable_instanceable: bool = False
    physics_material: RigidBodyMaterialCfg | None = None
    validate_composition: bool = True
