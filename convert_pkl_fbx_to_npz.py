#!/usr/bin/env python3
"""
Convert a PKL motion file + FBX scene file into a single NPZ file
with scene mesh included (in FK/MuJoCo coordinate space).

Usage:
    python convert_pkl_fbx_to_npz.py motion.pkl scene.fbx output.npz [--output-fps 50]
    $PYTHON convert_pkl_fbx_to_npz.py huddle_walk.pkl huddle.fbx output/huddle_walk.npz                                                                                                                                                                                                                       
    $PYTHON convert_pkl_fbx_to_npz.py stair_walk.pkl stair.fbx output/stair_walk.npz   
    $PYTHON convert_pkl_fbx_to_npz.py stair_walk_60depth.pkl stair_60depth.fbx output/stair_walk_60depth.npz    

Steps performed:
    1. Convert motion PKL -> NPZ via convert_motion_to_mj_npz.py
    2. Extract scene mesh from FBX via Blender (headless)
    3. Transform mesh vertices from Blender coords to FK/MuJoCo coords
       using the inverse of fk_to_blender_3x3 from the PKL
    4. Merge mesh into the NPZ and save
"""

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


BLENDER_EXTRACT_SCRIPT = '''
import bpy
import numpy as np
import sys

fbx_path = sys.argv[-2]
out_path = sys.argv[-1]

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.import_scene.fbx(filepath=fbx_path)

all_verts = []
all_faces = []
vert_offset = 0

for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        mesh = obj.data
        mat = obj.matrix_world
        verts = []
        for v in mesh.vertices:
            co = mat @ v.co
            verts.append([co.x, co.y, co.z])
        faces = []
        for poly in mesh.polygons:
            if len(poly.vertices) == 3:
                faces.append([poly.vertices[0] + vert_offset, poly.vertices[1] + vert_offset, poly.vertices[2] + vert_offset])
            elif len(poly.vertices) == 4:
                faces.append([poly.vertices[0] + vert_offset, poly.vertices[1] + vert_offset, poly.vertices[2] + vert_offset])
                faces.append([poly.vertices[0] + vert_offset, poly.vertices[2] + vert_offset, poly.vertices[3] + vert_offset])
        all_verts.extend(verts)
        all_faces.extend(faces)
        vert_offset += len(verts)

vertices = np.array(all_verts, dtype=np.float32)
faces = np.array(all_faces, dtype=np.int32)
np.savez(out_path, mesh_vertices=vertices, mesh_faces=faces)
print(f"Extracted mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
'''


def extract_fbx_mesh(fbx_path: str, output_npz_path: str):
    """Extract mesh from FBX using Blender in headless mode."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(BLENDER_EXTRACT_SCRIPT)
        script_path = f.name

    try:
        result = subprocess.run(
            ['blender', '--background', '--python', script_path, '--', fbx_path, output_npz_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"Blender stderr:\n{result.stderr[-500:]}", file=sys.stderr)
            raise RuntimeError(f"Blender mesh extraction failed (exit code {result.returncode})")
        for line in result.stdout.splitlines():
            if 'Extracted mesh' in line:
                print(f"  {line.strip()}")
    finally:
        os.unlink(script_path)


def get_blender_to_fk_matrix(pkl_path: str) -> np.ndarray:
    """Get the blender-to-FK coordinate transform from the PKL file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if 'fk_to_blender_3x3' in data and data['fk_to_blender_3x3'] is not None:
        fk_to_blender = np.array(data['fk_to_blender_3x3'], dtype=np.float32)
        blender_to_fk = np.linalg.inv(fk_to_blender)
        return blender_to_fk
    else:
        print("  Warning: no fk_to_blender_3x3 in PKL, assuming identity")
        return np.eye(3, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PKL motion + FBX scene -> NPZ with scene mesh"
    )
    parser.add_argument("pkl_path", type=str, help="Input motion PKL file")
    parser.add_argument("fbx_path", type=str, help="Input scene FBX file")
    parser.add_argument("output_path", type=str, help="Output NPZ path")
    parser.add_argument("--output-fps", type=int, default=50, help="Output FPS (default: 50)")
    parser.add_argument("--input-fps", type=int, default=30, help="Input FPS (default: 30)")
    parser.add_argument("--robot-xml", type=str, default=None, help="MuJoCo robot XML path")
    args = parser.parse_args()

    pkl_path = os.path.abspath(args.pkl_path)
    fbx_path = os.path.abspath(args.fbx_path)
    output_path = os.path.abspath(args.output_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Convert motion PKL -> NPZ
        print(f"[1/4] Converting motion: {pkl_path}")
        tmp_motion_npz = os.path.join(tmpdir, "motion.npz")
        convert_cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "convert_motion_to_mj_npz.py"),
            pkl_path, tmp_motion_npz,
            "--input-fps", str(args.input_fps),
            "--output-fps", str(args.output_fps),
        ]
        if args.robot_xml:
            convert_cmd += ["--robot-xml", args.robot_xml]
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError("Motion conversion failed")
        print(f"  Done. Output FPS: {args.output_fps}")

        # Step 2: Extract scene mesh from FBX
        print(f"[2/4] Extracting scene mesh: {fbx_path}")
        tmp_mesh_npz = os.path.join(tmpdir, "mesh.npz")
        extract_fbx_mesh(fbx_path, tmp_mesh_npz)

        # Step 3: Transform mesh to FK/MuJoCo coordinates
        print(f"[3/4] Transforming mesh coordinates (Blender -> FK/MuJoCo)")
        blender_to_fk = get_blender_to_fk_matrix(pkl_path)
        mesh_data = np.load(tmp_mesh_npz)
        mesh_verts = mesh_data['mesh_vertices'].astype(np.float32)
        mesh_faces = mesh_data['mesh_faces']
        mesh_verts_fk = (blender_to_fk @ mesh_verts.T).T.astype(np.float32)
        print(f"  Vertices: {mesh_verts_fk.shape[0]}, Faces: {mesh_faces.shape[0]}")

        # Step 4: Add ground plane
        print(f"[4/5] Adding ground plane")
        motion_data = dict(np.load(tmp_motion_npz, allow_pickle=True))
        jp = motion_data['joint_pos']

        # Ground plane covering robot path + scene mesh + 2m margin
        all_xy = np.concatenate([jp[:, :2], mesh_verts_fk[:, :2]], axis=0)
        margin = 2.0
        xy_min = all_xy.min(axis=0) - margin
        xy_max = all_xy.max(axis=0) + margin

        ground_verts = np.array([
            [xy_min[0], xy_min[1], 0.0],
            [xy_max[0], xy_min[1], 0.0],
            [xy_max[0], xy_max[1], 0.0],
            [xy_min[0], xy_max[1], 0.0],
        ], dtype=np.float32)
        ground_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        # Prepend ground plane, shift existing face indices by 4
        all_verts = np.concatenate([ground_verts, mesh_verts_fk], axis=0)
        all_faces = np.concatenate([ground_faces, mesh_faces + 4], axis=0)
        print(f"  Ground: [{xy_min[0]:.1f},{xy_min[1]:.1f}] to [{xy_max[0]:.1f},{xy_max[1]:.1f}]")

        # Step 5: Save final NPZ
        print(f"[5/5] Saving: {output_path}")
        motion_data['mesh_vertices'] = all_verts
        motion_data['mesh_faces'] = all_faces

        # Ensure string arrays use str dtype (not object) for allow_pickle=False compatibility
        for key in ['joint_names', 'body_names']:
            if key in motion_data and hasattr(motion_data[key], 'dtype') and motion_data[key].dtype == object:
                motion_data[key] = np.array(motion_data[key].tolist(), dtype=str)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        np.savez(output_path, **motion_data)

        print(f"\nDone!")
        print(f"  Frames: {jp.shape[0]}, FPS: {args.output_fps}")
        print(f"  Robot range:  x=[{jp[:,0].min():.2f},{jp[:,0].max():.2f}] y=[{jp[:,1].min():.2f},{jp[:,1].max():.2f}] z=[{jp[:,2].min():.2f},{jp[:,2].max():.2f}]")
        print(f"  Mesh range:   x=[{all_verts[:,0].min():.2f},{all_verts[:,0].max():.2f}] y=[{all_verts[:,1].min():.2f},{all_verts[:,1].max():.2f}] z=[{all_verts[:,2].min():.2f},{all_verts[:,2].max():.2f}]")
        print(f"  Mesh:         {all_verts.shape[0]} vertices, {all_faces.shape[0]} faces (incl. ground)")


if __name__ == "__main__":
    main()
