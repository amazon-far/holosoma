"""Convert motion .npz files to .npy in the visualize.py format.

Each motion .npz has body_pos_w / body_quat_w (39 bodies) and joint_pos (36
columns: first 7 are pelvis pose, last 29 are joint angles in motion-data
order). visualize.py expects:
    {
        "root_trans": (T, 3),     # pelvis world pos
        "root_ori":   (T, 4),     # pelvis quat xyzw
        "dof_pos":    (T, num_dofs),  # joint angles
        "fps":        int,
    }

So we extract pelvis (body 1) and the post-7 joint columns. The result is
the canonical kinematic motion data, with no physics rollout — useful as a
sanity-check ghost when verifying that ``teacher_<motion>.npy`` is actually
teacher's physics rollout (which should differ from this) and not a copy of
the kinematic reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    motion_dir = Path(args.motion_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in sorted(motion_dir.glob("*.npz")):
        d = np.load(npz_path, allow_pickle=True)
        pel_pos = d["body_pos_w"][:, 1, :].astype(np.float32)  # (T, 3)
        pel_quat = d["body_quat_w"][:, 1, :].astype(np.float32)  # (T, 4) xyzw
        joint_full = d["joint_pos"]  # (T, 36)
        joint_pos = joint_full[:, 7:].astype(np.float32)  # (T, 29)

        out = {
            "root_trans": pel_pos,
            "root_ori": pel_quat,
            "dof_pos": joint_pos,
            "fps": int(d["fps"][0]) if d["fps"].size else 50,
        }
        out_path = out_dir / f"motion_{npz_path.stem}.npy"
        np.save(out_path, out)
        print(f"{npz_path.name} → {out_path.name} ({pel_pos.shape[0]} frames)")


if __name__ == "__main__":
    main()
