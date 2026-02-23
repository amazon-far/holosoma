#!/usr/bin/env python3
"""Add a mesh to an existing motion NPZ file.

The mesh is stored as `mesh_vertices` and `mesh_faces` arrays inside the NPZ,
which the terrain system auto-loads when `mesh_type=load_obj` and
`tile_mesh=False`.

Usage:
    # Add mesh to motion file (overwrites in-place)
    python scripts/add_mesh_to_npz.py motion.npz mesh.obj

    # FBX files are auto-converted via Blender
    python scripts/add_mesh_to_npz.py motion.npz scene.fbx

    # Save to a new file instead of overwriting
    python scripts/add_mesh_to_npz.py motion.npz mesh.obj -o output.npz

    # Replace existing mesh data
    python scripts/add_mesh_to_npz.py motion.npz new_mesh.obj --replace

Supported mesh formats:
    - Direct (trimesh): OBJ, STL, PLY, GLB, GLTF, OFF, DAE
    - Via Blender:       FBX, BLEND (requires Blender installed)
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import trimesh

# Formats that need Blender conversion
BLENDER_FORMATS = {".fbx", ".blend"}


def _find_blender() -> str | None:
    """Find Blender binary on the system."""
    path = shutil.which("blender")
    if path:
        return path
    # Common install locations
    for candidate in [
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/snap/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]:
        if Path(candidate).is_file():
            return candidate
    return None


def _convert_with_blender(mesh_path: Path, output_obj: Path) -> None:
    """Convert a mesh file to OBJ using Blender's Python API."""
    blender = _find_blender()
    if blender is None:
        raise RuntimeError(
            f"Blender is required to load {mesh_path.suffix} files but was not found.\n"
            "Install Blender: https://www.blender.org/download/\n"
            "Or convert manually: blender --background --python-expr "
            "\"import bpy; bpy.ops.import_scene.fbx(filepath='input.fbx'); "
            "bpy.ops.wm.obj_export(filepath='output.obj')\""
        )

    suffix = mesh_path.suffix.lower()
    if suffix == ".fbx":
        import_cmd = f"bpy.ops.import_scene.fbx(filepath=r'{mesh_path}')"
    elif suffix == ".blend":
        import_cmd = f"bpy.ops.wm.open_mainfile(filepath=r'{mesh_path}')"
    else:
        raise ValueError(f"Unsupported Blender format: {suffix}")

    script = textwrap.dedent(f"""\
        import bpy
        # Clear default objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        # Import
        {import_cmd}
        # Export as OBJ
        bpy.ops.wm.obj_export(filepath=r'{output_obj}')
    """)

    print(f"Converting {mesh_path.name} -> OBJ via Blender...")
    result = subprocess.run(
        [blender, "--background", "--python-expr", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not output_obj.exists():
        raise RuntimeError(
            f"Blender conversion failed (exit code {result.returncode}).\n"
            f"stderr: {result.stderr[-500:] if result.stderr else '(empty)'}"
        )
    print(f"Blender conversion done: {output_obj}")


def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    """Load a mesh file, using Blender for FBX/BLEND formats."""
    mesh_file = Path(mesh_path)
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

    if mesh_file.suffix.lower() in BLENDER_FORMATS:
        # Convert via Blender to a temp OBJ, then load with trimesh
        with tempfile.TemporaryDirectory() as tmp:
            tmp_obj = Path(tmp) / f"{mesh_file.stem}.obj"
            _convert_with_blender(mesh_file, tmp_obj)
            return _load_trimesh(tmp_obj)
    else:
        return _load_trimesh(mesh_file)


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh file with trimesh."""
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load as Trimesh: {type(mesh)}")
    return mesh


def add_mesh_to_npz(
    npz_path: str,
    mesh_path: str,
    output_path: str | None = None,
    replace: bool = False,
) -> str:
    npz_file = Path(npz_path)
    if not npz_file.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")

    # Load existing NPZ
    data = dict(np.load(str(npz_file), allow_pickle=True))
    print(f"Loaded NPZ with keys: {sorted(data.keys())}")

    if "mesh_vertices" in data and not replace:
        print(
            f"NPZ already contains mesh data "
            f"({data['mesh_vertices'].shape[0]} vertices, {data['mesh_faces'].shape[0]} faces). "
            f"Use --replace to overwrite."
        )
        sys.exit(1)

    # Load mesh (auto-converts FBX via Blender if needed)
    mesh = load_mesh(mesh_path)

    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"  Bounds min: {mesh.vertices.min(axis=0)}")
    print(f"  Bounds max: {mesh.vertices.max(axis=0)}")

    # Add mesh data
    data["mesh_vertices"] = mesh.vertices.astype(np.float32)
    data["mesh_faces"] = mesh.faces.astype(np.int32)

    # Save
    out = output_path or str(npz_file)
    np.savez(out, **data)
    print(f"Saved to: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Add mesh to motion NPZ file. FBX files auto-convert via Blender."
    )
    parser.add_argument("npz", help="Path to motion NPZ file")
    parser.add_argument("mesh", help="Path to mesh file (OBJ, STL, FBX, etc.)")
    parser.add_argument("-o", "--output", help="Output path (default: overwrite input)")
    parser.add_argument(
        "--replace", action="store_true", help="Replace existing mesh data"
    )
    args = parser.parse_args()
    add_mesh_to_npz(args.npz, args.mesh, args.output, args.replace)


if __name__ == "__main__":
    main()
