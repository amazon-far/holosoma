"""Read and write generic kinematic trees using standard OpenUSD schemas."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from holosoma_retargeting.kinematics import (
    KinematicTree,
    MeshAttachment,
    PointSetAttachment,
    RigidBodyDefinition,
    SphericalJointDefinition,
    Transform,
    ValidationReport,
    validate_kinematic_tree,
)
from holosoma_retargeting.kinematics.model import MetadataValue


def _pxr():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "OpenUSD support requires the optional 'usd' dependencies. "
            "Install holosoma-retargeting[usd]."
        ) from exc
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt


def _usd_shade():
    try:
        from pxr import UsdShade  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "OpenUSD material support requires the optional 'usd' dependencies. "
            "Install holosoma-retargeting[usd]."
        ) from exc
    return UsdShade


def _identifier(name: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not result or result[0].isdigit():
        result = f"_{result}"
    return result


def _path(value: str):
    _, Sdf, _, _, _, _ = _pxr()
    return Sdf.Path(value)


def create_usd_stage(
    path: str | Path,
    *,
    meters_per_unit: float = 1.0,
    up_axis: str = "Z",
):
    """Create a new USD stage with explicit metric and axis conventions."""

    _, _, Usd, UsdGeom, _, _ = _pxr()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    if stage is None:
        raise RuntimeError(f"Could not create USD stage: {path}")
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    axis_token = UsdGeom.Tokens.z if up_axis.upper() == "Z" else UsdGeom.Tokens.y
    if up_axis.upper() not in {"Y", "Z"}:
        raise ValueError("OpenUSD up_axis must be 'Y' or 'Z'")
    UsdGeom.SetStageUpAxis(stage, axis_token)
    return stage


def open_usd_stage(path: str | Path, *, read_only: bool = True):
    """Open an existing USD stage; writes occur only if callers explicitly save it."""

    del read_only
    _, _, Usd, _, _, _ = _pxr()
    stage = Usd.Stage.Open(str(Path(path)))
    if stage is None:
        raise FileNotFoundError(f"Could not open USD stage: {path}")
    return stage


def _gf_vec3(value: np.ndarray):
    Gf, _, _, _, _, _ = _pxr()
    value = np.asarray(value, dtype=float)
    return Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))


def _gf_quat(value: np.ndarray):
    Gf, _, _, _, _, _ = _pxr()
    value = np.asarray(value, dtype=float)
    return Gf.Quatf(float(value[0]), Gf.Vec3f(float(value[1]), float(value[2]), float(value[3])))


def _numpy_quat(value: Any) -> np.ndarray:
    imaginary = value.GetImaginary()
    return np.array([value.GetReal(), imaginary[0], imaginary[1], imaginary[2]], dtype=float)


def _set_transform(xformable: Any, transform: Transform) -> None:
    _, _, _, UsdGeom, _, _ = _pxr()
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        tuple(float(value) for value in transform.translation_m)
    )
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(_gf_quat(transform.rotation_wxyz))


def _get_transform(prim: Any) -> Transform:
    _, _, _, UsdGeom, _, _ = _pxr()
    matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
    translation = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    return Transform(
        translation_m=np.array(translation, dtype=float),
        rotation_wxyz=_numpy_quat(quaternion),
    )


def _metadata_type(value: MetadataValue):
    _, Sdf, _, _, _, _ = _pxr()
    if isinstance(value, bool):
        return Sdf.ValueTypeNames.Bool
    if isinstance(value, int):
        return Sdf.ValueTypeNames.Int64
    if isinstance(value, float):
        return Sdf.ValueTypeNames.Double
    if isinstance(value, str):
        return Sdf.ValueTypeNames.String
    if isinstance(value, tuple):
        if all(isinstance(item, str) for item in value):
            return Sdf.ValueTypeNames.StringArray
        if all(isinstance(item, bool) for item in value):
            return Sdf.ValueTypeNames.BoolArray
        if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            return Sdf.ValueTypeNames.Int64Array
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return Sdf.ValueTypeNames.DoubleArray
    raise TypeError(f"Unsupported USD metadata value: {value!r}")


def _write_metadata(prim: Any, metadata: Mapping[str, MetadataValue]) -> None:
    for name, value in metadata.items():
        attribute = prim.CreateAttribute(name, _metadata_type(value), custom=True)
        attribute.Set(list(value) if isinstance(value, tuple) else value)


def _read_metadata(prim: Any) -> dict[str, MetadataValue]:
    result: dict[str, MetadataValue] = {}
    for attribute in prim.GetAttributes():
        if not attribute.IsCustom():
            continue
        value = attribute.Get()
        if value is None:
            continue
        if isinstance(value, (list, tuple)) or value.__class__.__module__.startswith("pxr.Vt"):
            result[attribute.GetName()] = tuple(value)
        else:
            result[attribute.GetName()] = value
    return result


def _write_point_set(body_prim: Any, point_set: PointSetAttachment) -> None:
    Gf, Sdf, _, UsdGeom, _, Vt = _pxr()
    path = body_prim.GetPath().AppendChild(_identifier(point_set.name))
    points = UsdGeom.Points.Define(body_prim.GetStage(), path)
    points.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, point)) for point in point_set.points_m]))
    points.CreateWidthsAttr().Set([float(point_set.width_m)] * len(point_set.points_m))
    points.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    prim = points.GetPrim()
    prim.CreateAttribute("model:attachmentName", Sdf.ValueTypeNames.String, custom=True).Set(point_set.name)
    prim.CreateAttribute("model:pointNames", Sdf.ValueTypeNames.StringArray, custom=True).Set(
        list(point_set.point_names)
    )
    _write_metadata(prim, point_set.metadata)


def _material_key(attachment: MeshAttachment) -> tuple[str, tuple[int, int, int]]:
    return attachment.category, attachment.color_rgb


def _define_preview_material(stage: Any, looks_path: Any, attachment: MeshAttachment) -> Any:
    """Create one reusable standard UsdPreviewSurface material."""

    Gf, Sdf, _, _, _, _ = _pxr()
    UsdShade = _usd_shade()
    category, color_rgb = _material_key(attachment)
    material_name = _identifier(f"{category}_{color_rgb[0]}_{color_rgb[1]}_{color_rgb[2]}")
    material = UsdShade.Material.Define(stage, looks_path.AppendChild(material_name))
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    color = Gf.Vec3f(*(float(channel) / 255.0 for channel in color_rgb))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    material.GetPrim().CreateAttribute("model:meshCategory", Sdf.ValueTypeNames.String, custom=True).Set(category)
    return material


def _write_mesh(body_prim: Any, attachment: MeshAttachment, material: Any) -> None:
    Gf, Sdf, _, UsdGeom, _, Vt = _pxr()
    path = body_prim.GetPath().AppendChild(_identifier(attachment.name))
    mesh = UsdGeom.Mesh.Define(body_prim.GetStage(), path)
    mesh.CreatePointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*map(float, vertex)) for vertex in attachment.vertices_m])
    )
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(attachment.faces))
    mesh.CreateFaceVertexIndicesAttr().Set(np.asarray(attachment.faces, dtype=np.int32).reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.render)
    color = [float(channel) / 255.0 for channel in attachment.color_rgb]
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    prim = mesh.GetPrim()
    prim.CreateAttribute("model:attachmentName", Sdf.ValueTypeNames.String, custom=True).Set(attachment.name)
    prim.CreateAttribute("model:meshCategory", Sdf.ValueTypeNames.String, custom=True).Set(attachment.category)
    _write_metadata(prim, attachment.metadata)
    _usd_shade().MaterialBindingAPI.Apply(prim).Bind(material)


def write_kinematic_tree_to_stage(
    stage: Any,
    model: KinematicTree,
    *,
    root_path: str = "/XsensAvatar",
    replace_existing: bool = False,
) -> None:
    """Write a generic tree without modifying unrelated stage content."""

    _, Sdf, _, UsdGeom, UsdPhysics, _ = _pxr()
    validation = validate_kinematic_tree(model)
    validation.raise_if_invalid()
    root_sdf_path = Sdf.Path(root_path)
    existing = stage.GetPrimAtPath(root_sdf_path)
    if existing.IsValid():
        if not replace_existing:
            raise ValueError(f"USD root already exists: {root_path}")
        stage.RemovePrim(root_sdf_path)

    root = UsdGeom.Xform.Define(stage, root_sdf_path)
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().CreateAttribute("model:name", Sdf.ValueTypeNames.String, custom=True).Set(model.name)
    root.GetPrim().CreateAttribute("model:rootBody", Sdf.ValueTypeNames.String, custom=True).Set(model.root_body)
    _write_metadata(root.GetPrim(), model.metadata)
    bodies_scope = UsdGeom.Scope.Define(stage, root_sdf_path.AppendChild("Bodies"))
    UsdGeom.Scope.Define(stage, root_sdf_path.AppendChild("Joints"))
    has_meshes = any(body.meshes for body in model.bodies)
    looks_path = root_sdf_path.AppendChild("Looks")
    if has_meshes:
        UsdGeom.Scope.Define(stage, looks_path)
    materials: dict[tuple[str, tuple[int, int, int]], Any] = {}

    body_paths: dict[str, Any] = {}
    for body in model.bodies:
        body_path = bodies_scope.GetPath().AppendChild(_identifier(body.name))
        xform = UsdGeom.Xform.Define(stage, body_path)
        _set_transform(xform, body.reference_pose)
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        xform.GetPrim().CreateAttribute("model:name", Sdf.ValueTypeNames.String, custom=True).Set(body.name)
        _write_metadata(xform.GetPrim(), body.metadata)
        body_paths[body.name] = body_path
        for point_set in body.point_sets:
            _write_point_set(xform.GetPrim(), point_set)
        for mesh in body.meshes:
            key = _material_key(mesh)
            material = materials.get(key)
            if material is None:
                material = _define_preview_material(stage, looks_path, mesh)
                materials[key] = material
            _write_mesh(xform.GetPrim(), mesh, material)

    joints_path = root_sdf_path.AppendChild("Joints")
    for definition in model.joints:
        joint = UsdPhysics.SphericalJoint.Define(stage, joints_path.AppendChild(_identifier(definition.name)))
        joint.CreateBody0Rel().SetTargets([body_paths[definition.parent_body]])
        joint.CreateBody1Rel().SetTargets([body_paths[definition.child_body]])
        joint.CreateLocalPos0Attr().Set(_gf_vec3(definition.parent_frame.translation_m))
        joint.CreateLocalRot0Attr().Set(_gf_quat(definition.parent_frame.rotation_wxyz))
        joint.CreateLocalPos1Attr().Set(_gf_vec3(definition.child_frame.translation_m))
        joint.CreateLocalRot1Attr().Set(_gf_quat(definition.child_frame.rotation_wxyz))
        prim = joint.GetPrim()
        prim.CreateAttribute("model:name", Sdf.ValueTypeNames.String, custom=True).Set(definition.name)
        _write_metadata(prim, definition.metadata)


def _relationship_body_name(stage: Any, relationship: Any) -> str:
    targets = relationship.GetTargets()
    if len(targets) != 1:
        raise ValueError(f"Expected one body target for {relationship.GetPath()}, got {len(targets)}")
    prim = stage.GetPrimAtPath(targets[0])
    if not prim.IsValid():
        raise ValueError(f"Joint relationship targets missing body: {targets[0]}")
    attribute = prim.GetAttribute("model:name")
    return str(attribute.Get()) if attribute and attribute.HasAuthoredValueOpinion() else prim.GetName()


def _read_point_set(prim: Any) -> PointSetAttachment:
    _, _, _, UsdGeom, _, _ = _pxr()
    points = UsdGeom.Points(prim)
    positions = np.asarray(points.GetPointsAttr().Get(), dtype=float)
    names_attr = prim.GetAttribute("model:pointNames")
    point_names = (
        tuple(str(name) for name in names_attr.Get())
        if names_attr
        else tuple(str(i) for i in range(len(positions)))
    )
    widths = points.GetWidthsAttr().Get()
    width = float(widths[0]) if widths else 0.008
    name_attr = prim.GetAttribute("model:attachmentName")
    name = str(name_attr.Get()) if name_attr else prim.GetName()
    metadata = _read_metadata(prim)
    metadata.pop("model:attachmentName", None)
    metadata.pop("model:pointNames", None)
    return PointSetAttachment(name, positions, point_names, width, metadata)


def _read_mesh(prim: Any) -> MeshAttachment:
    _, _, _, UsdGeom, _, _ = _pxr()
    mesh = UsdGeom.Mesh(prim)
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    if not np.all(counts == 3):
        raise ValueError(f"Only triangular meshes are supported: {prim.GetPath()}")
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    display_color = mesh.GetDisplayColorPrimvar().Get()
    color = (180, 180, 180) if not display_color else tuple(round(float(c) * 255.0) for c in display_color[0])
    name_attr = prim.GetAttribute("model:attachmentName")
    category_attr = prim.GetAttribute("model:meshCategory")
    name = str(name_attr.Get()) if name_attr else prim.GetName()
    category = str(category_attr.Get()) if category_attr else "render"
    metadata = _read_metadata(prim)
    metadata.pop("model:attachmentName", None)
    metadata.pop("model:meshCategory", None)
    return MeshAttachment(name, vertices, faces, color, category, metadata)


def read_kinematic_tree_from_stage(
    stage: Any,
    *,
    root_path: str = "/XsensAvatar",
) -> KinematicTree:
    """Reconstruct the generic model from standard USD physics and geometry."""

    _, _, _, UsdGeom, UsdPhysics, _ = _pxr()
    root = stage.GetPrimAtPath(_path(root_path))
    if not root.IsValid():
        raise KeyError(f"USD kinematic root does not exist: {root_path}")
    bodies_prim = stage.GetPrimAtPath(root.GetPath().AppendChild("Bodies"))
    joints_prim = stage.GetPrimAtPath(root.GetPath().AppendChild("Joints"))
    if not bodies_prim.IsValid() or not joints_prim.IsValid():
        raise ValueError(f"USD root '{root_path}' must contain Bodies and Joints scopes")

    bodies: list[RigidBodyDefinition] = []
    for prim in bodies_prim.GetChildren():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name_attr = prim.GetAttribute("model:name")
        name = str(name_attr.Get()) if name_attr else prim.GetName()
        point_sets: list[PointSetAttachment] = []
        meshes: list[MeshAttachment] = []
        for child in prim.GetChildren():
            if child.IsA(UsdGeom.Points):
                point_sets.append(_read_point_set(child))
            elif child.IsA(UsdGeom.Mesh):
                meshes.append(_read_mesh(child))
        metadata = _read_metadata(prim)
        metadata.pop("model:name", None)
        bodies.append(RigidBodyDefinition(name, _get_transform(prim), tuple(point_sets), tuple(meshes), metadata))

    joints: list[SphericalJointDefinition] = []
    for prim in joints_prim.GetChildren():
        if not prim.IsA(UsdPhysics.SphericalJoint):
            continue
        joint = UsdPhysics.SphericalJoint(prim)
        name_attr = prim.GetAttribute("model:name")
        name = str(name_attr.Get()) if name_attr else prim.GetName()
        metadata = _read_metadata(prim)
        metadata.pop("model:name", None)
        joints.append(
            SphericalJointDefinition(
                name=name,
                parent_body=_relationship_body_name(stage, joint.GetBody0Rel()),
                child_body=_relationship_body_name(stage, joint.GetBody1Rel()),
                parent_frame=Transform(
                    np.asarray(joint.GetLocalPos0Attr().Get(), dtype=float),
                    _numpy_quat(joint.GetLocalRot0Attr().Get()),
                ),
                child_frame=Transform(
                    np.asarray(joint.GetLocalPos1Attr().Get(), dtype=float),
                    _numpy_quat(joint.GetLocalRot1Attr().Get()),
                ),
                metadata=metadata,
            )
        )

    name_attr = root.GetAttribute("model:name")
    root_body_attr = root.GetAttribute("model:rootBody")
    metadata = _read_metadata(root)
    metadata.pop("model:name", None)
    metadata.pop("model:rootBody", None)
    return KinematicTree(
        name=str(name_attr.Get()) if name_attr else root.GetName(),
        root_body=str(root_body_attr.Get()) if root_body_attr else bodies[0].name,
        bodies=tuple(bodies),
        joints=tuple(joints),
        metadata=metadata,
    )


def validate_usd_kinematic_tree(
    stage: Any,
    *,
    root_path: str = "/XsensAvatar",
) -> ValidationReport:
    """Validate a serialized model through the same generic invariants."""

    try:
        return validate_kinematic_tree(read_kinematic_tree_from_stage(stage, root_path=root_path))
    except (KeyError, TypeError, ValueError) as exc:
        return ValidationReport(errors=(str(exc),))
