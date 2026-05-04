"""Relabel rollouts by "did the student stay upright and reach the goal x?".

Drops the motion-tracking criterion entirely. For every (motion, rollout)
pair we compute:

  * ``progress``: (final_student_x − start_x) / (goal_x − start_x).
    1.0 means the student got all the way to the motion's last pelvis x.
  * ``z_ok``: ``|final_student_z − final_motion_z| < z_tolerance`` — the
    student's pelvis at the end of recording is within ``z_tolerance`` of
    where the motion expected it to be (so it didn't fall through the
    floor or rocket above the stair top).
  * ``not_fell``: rollout was recorded for at least ``min_progress_frames``
    fraction of the motion's full length (rollout_single_policy clips at
    catastrophic z, so a short rollout is the "robot fell" signal).

Success = all three ≥ thresholds.

Outputs ``{motion_basename: {label, progress, z_ok, fell, ...}}`` JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def label_one(rollout_path: Path, motion_path: Path,
              progress_thresh: float, z_tol: float, min_frames_frac: float):
    rollout = np.load(rollout_path, allow_pickle=True).item()
    motion = np.load(motion_path, allow_pickle=True)
    target = motion["body_pos_w"][:, 1, :]  # (T_m, 3) tile-local
    student = rollout["root_trans"]  # (T_s, 3) tile-local

    start_x = float(target[0, 0])
    goal_x = float(target[-1, 0])
    final_z_target = float(target[-1, 2])
    final_x_student = float(student[-1, 0])
    final_z_student = float(student[-1, 2])

    span = goal_x - start_x
    if abs(span) < 1e-6:
        progress = 0.0
    else:
        progress = (final_x_student - start_x) / span

    z_ok = abs(final_z_student - final_z_target) < z_tol
    fell = student.shape[0] < int(target.shape[0] * min_frames_frac)
    reached = progress >= progress_thresh

    label = "success" if (reached and z_ok and not fell) else "fail"
    return {
        "label": label,
        "progress": progress,
        "z_diff": final_z_student - final_z_target,
        "z_ok": z_ok,
        "fell": fell,
        "n_frames": int(student.shape[0]),
        "motion_n_frames": int(target.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_dir", required=True)
    parser.add_argument("--motion_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress_threshold", type=float, default=0.9,
                        help="Fraction of (goal_x − start_x) the student must cover (default 0.9).")
    parser.add_argument("--z_tolerance", type=float, default=1.0,
                        help="Max |final_student_z − final_motion_z| in metres (default 1.0).")
    parser.add_argument("--min_frames_fraction", type=float, default=0.5,
                        help="Rollouts shorter than this fraction of motion length are 'fell' (default 0.5).")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    motion_dir = Path(args.motion_dir)
    motions = {p.stem: p for p in motion_dir.glob("*.npz")}

    labels = {}
    for npy in sorted(rollout_dir.glob("policy_*.npy")):
        motion_key = npy.stem[len("policy_"):]
        if motion_key not in motions:
            print(f"[skip] no motion .npz for {npy.name}")
            continue
        labels[motion_key] = label_one(
            npy, motions[motion_key],
            args.progress_threshold, args.z_tolerance, args.min_frames_fraction,
        )

    n_success = sum(1 for v in labels.values() if v["label"] == "success")
    print(f"Success: {n_success}/{len(labels)}")
    with open(args.out, "w") as f:
        json.dump(labels, f, indent=2, sort_keys=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
