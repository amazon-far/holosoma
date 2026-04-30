"""Post-process paired rollout npys.

For each motion we have a student and a teacher rollout; this script:

  1. **trims** each at the first frame whose pelvis position deviates more than
     ``--threshold`` metres from the motion target. (Without this the saved
     trajectory keeps going after the robot falls because eval-mode envs don't
     auto-reset, ending with garbage `z = -1000m` plummet frames.)

  2. **pads** the shorter of (student, teacher) to the longer one by repeating
     the last valid frame. This is what makes divergence visible: the failed
     policy freezes at its last good pose while the surviving policy keeps
     walking, so the video shows them clearly separating instead of cutting
     off the moment one of them dies.

Both rollouts are also clipped to the motion's actual length.

Usage:
    python scripts/eval/unreal_engine/trim_rollouts_at_failure.py \
        --rollout_dir /tmp/dagger_sweep/paired_renders/rollouts \
        --motion_dir /tmp/dagger_sweep/filtered_motions \
        --threshold 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _failure_index(rollout: dict, motion: np.lib.npyio.NpzFile, threshold: float) -> int:
    """First frame at which the robot has *catastrophically* left the
    intended scene — defined as pelvis z dropping more than ``threshold``
    metres below the lowest motion-target pelvis z (free-falling into the
    void), or rising more than ``threshold`` metres above the highest
    motion-target pelvis z (rocketing into the sky). The official
    success-rate eval uses a stricter local-frame body-pose criterion which
    is too tight for visualisation: it would clip OK-labeled motions where
    the robot stayed on the stair but drifted slightly in xy."""

    rollout_pelvis = rollout["root_trans"]
    target_z = motion["body_pos_w"][:, 1, 2]
    z_min = float(target_z.min()) - threshold
    z_max = float(target_z.max()) + threshold
    z = rollout_pelvis[:, 2]
    bad = (z < z_min) | (z > z_max)
    if not bad.any():
        return len(rollout_pelvis)
    return int(max(np.argmax(bad), 1))


def _trim_in_place(rollout: dict, end: int) -> dict:
    return {
        "root_trans": rollout["root_trans"][:end],
        "root_ori": rollout["root_ori"][:end],
        "dof_pos": rollout["dof_pos"][:end],
        "fps": rollout.get("fps", 50),
    }


def _pad(rollout: dict, target_len: int) -> dict:
    cur_len = len(rollout["root_trans"])
    if cur_len >= target_len:
        return rollout
    pad_n = target_len - cur_len
    return {
        "root_trans": np.concatenate(
            [rollout["root_trans"], np.repeat(rollout["root_trans"][-1:], pad_n, axis=0)], axis=0
        ),
        "root_ori": np.concatenate(
            [rollout["root_ori"], np.repeat(rollout["root_ori"][-1:], pad_n, axis=0)], axis=0
        ),
        "dof_pos": np.concatenate(
            [rollout["dof_pos"], np.repeat(rollout["dof_pos"][-1:], pad_n, axis=0)], axis=0
        ),
        "fps": rollout.get("fps", 50),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_dir", required=True)
    parser.add_argument("--motion_dir", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Catastrophic z-bound margin (m) past the motion's pelvis z range. "
        "Default 2m: a robot inside [z_min−2, z_max+2] is considered still on "
        "the scene; outside that, it has rocketed/plummeted and gets clipped.",
    )
    parser.add_argument(
        "--max_pad",
        type=int,
        default=None,
        help="Cap the padded length at this many frames (default: motion length).",
    )
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    motion_dir = Path(args.motion_dir)

    motions = {p.stem: p for p in motion_dir.glob("*.npz")}

    # First pass: discover (student, teacher) pairs grouped by motion key.
    pairs: dict[str, dict[str, Path]] = {}
    for npy in sorted(rollout_dir.glob("*.npy")):
        stem = npy.stem
        if stem.startswith("student_"):
            label, motion_key = "student", stem[len("student_"):]
        elif stem.startswith("teacher_"):
            label, motion_key = "teacher", stem[len("teacher_"):]
        else:
            continue
        pairs.setdefault(motion_key, {})[label] = npy

    for motion_key, members in sorted(pairs.items()):
        if motion_key not in motions:
            print(f"[skip] no motion .npz for {motion_key}")
            continue
        if "student" not in members or "teacher" not in members:
            print(f"[skip] {motion_key}: missing one of (student, teacher)")
            continue

        student = np.load(members["student"], allow_pickle=True).item()
        teacher = np.load(members["teacher"], allow_pickle=True).item()
        motion = np.load(motions[motion_key], allow_pickle=True)
        motion_len = motion["body_pos_w"].shape[0]

        s_end = _failure_index(student, motion, args.threshold)
        t_end = _failure_index(teacher, motion, args.threshold)
        # Always cap by motion length; never let either trajectory extend past
        # what the motion itself defines.
        s_end = min(s_end, motion_len, len(student["root_trans"]))
        t_end = min(t_end, motion_len, len(teacher["root_trans"]))

        student_trim = _trim_in_place(student, s_end)
        teacher_trim = _trim_in_place(teacher, t_end)

        target = max(s_end, t_end)
        if args.max_pad is not None:
            target = min(target, args.max_pad)

        student_padded = _pad(student_trim, target)
        teacher_padded = _pad(teacher_trim, target)

        np.save(members["student"], student_padded)
        np.save(members["teacher"], teacher_padded)
        print(
            f"{motion_key}: trimmed s={s_end} t={t_end}, padded both to {target} "
            f"(motion_len={motion_len})"
        )


if __name__ == "__main__":
    main()
