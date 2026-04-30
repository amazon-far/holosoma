"""Decide whether each fresh student rollout actually succeeded.

Criterion (intuitive, world-frame, applied to the trimmed rollout):
  - max pelvis xy deviation from the motion target during the rollout
  - whether the rollout reached the motion's intended end (last-frame pelvis
    within ``end_threshold`` metres of the motion's last pelvis target).

A motion is labelled ``ok`` only if BOTH:
  - max pelvis deviation across all rendered frames < ``max_threshold``
  - last-frame pelvis is within ``end_threshold`` of the motion's last pelvis target

Otherwise ``fail``. Output is a JSON ``{motion_basename: label}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def label_one(student_path: Path, motion_path: Path, max_thresh: float, end_thresh: float) -> str:
    student = np.load(student_path, allow_pickle=True).item()
    motion = np.load(motion_path, allow_pickle=True)
    target_pelvis = motion["body_pos_w"][:, 1, :]
    rollout_pelvis = student["root_trans"]

    n = min(len(target_pelvis), len(rollout_pelvis))
    if n == 0:
        return "fail"
    err = np.linalg.norm(target_pelvis[:n] - rollout_pelvis[:n], axis=1)

    # Last-frame distance to motion's intended end position.
    end_err = float(np.linalg.norm(target_pelvis[-1] - rollout_pelvis[-1]))

    # If the rollout was trimmed early (shorter than the motion), the robot
    # didn't make it to the end. Treat as fail.
    if len(rollout_pelvis) < len(target_pelvis) - 5:
        return "fail"

    if err.max() > max_thresh:
        return "fail"
    if end_err > end_thresh:
        return "fail"
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_dir", required=True)
    parser.add_argument("--motion_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_threshold", type=float, default=0.5,
                        help="Max pelvis xy/z deviation allowed during rollout (m).")
    parser.add_argument("--end_threshold", type=float, default=0.5,
                        help="Max distance allowed at last frame from motion end (m).")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    motion_dir = Path(args.motion_dir)
    motions = {p.stem: p for p in motion_dir.glob("*.npz")}

    labels: dict[str, str] = {}
    for npy in sorted(rollout_dir.glob("student_*.npy")):
        motion_key = npy.stem[len("student_"):]
        if motion_key not in motions:
            continue
        labels[motion_key] = label_one(
            npy, motions[motion_key], args.max_threshold, args.end_threshold
        )
        print(f"{motion_key}: {labels[motion_key]}")

    with open(args.out, "w") as f:
        json.dump(labels, f, indent=2, sort_keys=True)
    n_ok = sum(1 for v in labels.values() if v == "ok")
    print(f"Wrote {args.out}: {n_ok}/{len(labels)} ok")


if __name__ == "__main__":
    main()
