"""Preserve Holosoma's asset contracts with URDF importer 3."""

from __future__ import annotations

from dataclasses import fields, replace
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence
import xml.etree.ElementTree as ET


def merge_fixed_joints(
    urdf_path: str,
    output_path: str,
    preserve_bodies: Sequence[str] = (),
    ros_package_paths: list[dict[str, str]] | None = None,
) -> None:
    """Use the native mass/transform merger, retaining named and ``dont_collapse`` joints.

    Only the intermediate XML hides protected joints from the native merger. Restore
    their fixed type before import, so no extra DOFs or artificial inertia are added.
    """
    from isaaclab.utils.version import has_kit

    if has_kit():
        from isaaclab.sim.utils import enable_extension

        enable_extension("isaacsim.asset.importer.urdf")
    from isaacsim.asset.importer.urdf.impl.urdf_utils import (
        _rewrite_relative_mesh_paths_to_absolute,
        merge_fixed_joints as native_merge,
    )

    source = Path(urdf_path).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    protected = set()
    for joint in root.findall("joint"):
        child = joint.find("child")
        if joint.get("type") == "fixed" and (
            joint.get("dont_collapse", "").lower() == "true"
            or (child is not None and child.get("link") in preserve_bodies)
        ):
            protected.add(joint.get("name"))
            joint.set("type", "holosoma_preserved_fixed")
    # The intermediate file lives outside the asset tree. Reuse the pinned
    # importer's resolver for local, file:// and package:// mesh paths.
    for element in root.iter("texture"):
        filename = element.get("filename")
        if filename and "://" not in filename and not Path(filename).is_absolute():
            element.set("filename", str((source.parent / filename).resolve()))
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    _rewrite_relative_mesh_paths_to_absolute(output_path, str(source.parent), ros_package_paths)
    native_merge(output_path, output_path)
    tree = ET.parse(output_path)
    for joint in tree.getroot().findall("joint"):
        if joint.get("name") in protected:
            joint.set("type", "fixed")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def replace_collision_cylinders(stage: Any) -> None:
    """Keep the 5.1 capsule radius and cylindrical height; preserve visuals and mass."""
    from pxr import Usd, UsdGeom, UsdPhysics

    paths = [
        prim.GetPath()
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if prim.IsA(UsdGeom.Cylinder) and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsInstanceProxy():
            owner = prim
            while owner.IsInstanceProxy():
                owner = owner.GetParent()
            owner.SetInstanceable(False)
            prim = stage.GetPrimAtPath(path)
        prim.SetTypeName("Capsule")
        capsule = UsdGeom.Capsule(prim)
        capsule.GetExtentAttr().Set(UsdGeom.Boundable.ComputeExtentFromPlugins(capsule, Usd.TimeCode.Default()))


def convert_urdf(cfg: Any) -> str:
    from isaaclab.sim.converters import UrdfConverter
    from pxr import Usd

    # The upstream flag only emits a warning now. Apply its old physical meaning
    # to the generated USD before any engine imports that USD into its model.
    with TemporaryDirectory(prefix="holosoma-urdf-") as scratch:
        native_cfg = replace(cfg, replace_cylinders_with_capsules=False)
        if cfg.merge_fixed_joints:
            merged = str(Path(scratch) / Path(cfg.asset_path).name)
            merge_fixed_joints(cfg.asset_path, merged, getattr(cfg, "preserve_fixed_bodies", ()), cfg.ros_package_paths)
            # Conversion output must outlive the temporary input.
            native_cfg = replace(native_cfg, asset_path=merged, merge_fixed_joints=False)
        converter = UrdfConverter(native_cfg)
        path: str = converter.usd_path
    if cfg.replace_cylinders_with_capsules:
        stage = Usd.Stage.Open(path)
        replace_collision_cylinders(stage)
        stage.GetRootLayer().Save()
    return path


def _spawn_from_urdf(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Any:
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd_file

    return spawn_from_usd_file(prim_path, convert_urdf(cfg), cfg, translation, orientation)


@lru_cache(maxsize=1)
def _native_urdf_spawner() -> Any:
    from isaaclab.sim.utils import clone

    return clone(_spawn_from_urdf)


@lru_cache(maxsize=1)
def _native_urdf_config_type() -> Any:
    from isaaclab.sim import UrdfFileCfg
    from isaaclab.utils import configclass

    @configclass
    class HolosomaUrdfFileCfg(UrdfFileCfg):
        preserve_fixed_bodies: tuple[str, ...] = ()

    return HolosomaUrdfFileCfg


def urdf_spawn_config(cfg: Any, *, preserve_fixed_bodies: Sequence[str] = ()) -> Any:
    """Keep configuration construction pure; convert only when the asset spawns."""
    values = {field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init}
    values.update(func=_native_urdf_spawner(), preserve_fixed_bodies=tuple(preserve_fixed_bodies))
    return _native_urdf_config_type()(**values)
