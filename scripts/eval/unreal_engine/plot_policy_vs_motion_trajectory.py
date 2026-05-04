"""Plot policy rollout root trajectory vs the commanded motion pelvis trajectory.

Reads the per-motion rollout produced by ``rollout_single_policy.py`` and
the corresponding raw motion ``.npz`` (the file the script feeds to the
policy as the goal trajectory), and draws side-by-side top-down (xy) and
side-view (xz) plots so you can see whether the policy reached the goal x
and stayed at the right z over the synthetic stair tile.

Usage::

    python scripts/eval/unreal_engine/plot_policy_vs_motion_trajectory.py \
        --rollout_dir /tmp/dagger_sweep/synth_..._model_33000/rollouts \
        --motion_dir  src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/synth_stairs_h010-030_w030-080 \
        --labels_json /tmp/dagger_sweep/synth_..._model_33000/reach_labels.json \
        --out_dir     /tmp/dagger_sweep/synth_..._model_33000/trajectory_plots \
        --n_success 4 --n_fail 4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_pair(rollout_npy: Path, motion_npz: Path):
    rollout = np.load(rollout_npy, allow_pickle=True).item()
    motion = np.load(motion_npz, allow_pickle=True)

    # body_pos_w[:, 1, :] is the pelvis trajectory in tile-local coords.
    # rollout["root_trans"] is also in tile-local coords (rollout_single_policy
    # subtracts env_origins each frame).
    motion_pelvis = np.asarray(motion["body_pos_w"][:, 1, :], dtype=np.float64)
    student_root = np.asarray(rollout["root_trans"], dtype=np.float64)
    return motion_pelvis, student_root


def _plot_one(ax_xy, ax_xz, name, label, motion_xyz, student_xyz):
    color = "tab:green" if label == "success" else "tab:red"

    ax_xy.plot(motion_xyz[:, 0], motion_xyz[:, 1], "k--", lw=1.4,
               label="motion command (pelvis)", alpha=0.9)
    ax_xy.plot(student_xyz[:, 0], student_xyz[:, 1], color=color, lw=1.6,
               label="policy rollout (root)", alpha=0.95)
    ax_xy.scatter(motion_xyz[0, 0], motion_xyz[0, 1], c="k", marker="o", s=22, zorder=5)
    ax_xy.scatter(motion_xyz[-1, 0], motion_xyz[-1, 1], c="k", marker="*", s=40, zorder=5,
                  label="motion goal")
    ax_xy.scatter(student_xyz[-1, 0], student_xyz[-1, 1], c=color, marker="X", s=40,
                  zorder=5, label="policy final")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title(f"[{label.upper()}] {name}\ntop-down (xy)", fontsize=9)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best", fontsize=7)

    ax_xz.plot(motion_xyz[:, 0], motion_xyz[:, 2], "k--", lw=1.4, alpha=0.9,
               label="motion command (pelvis)")
    ax_xz.plot(student_xyz[:, 0], student_xyz[:, 2], color=color, lw=1.6, alpha=0.95,
               label="policy rollout (root)")
    ax_xz.scatter(motion_xyz[0, 0], motion_xyz[0, 2], c="k", marker="o", s=22, zorder=5)
    ax_xz.scatter(motion_xyz[-1, 0], motion_xyz[-1, 2], c="k", marker="*", s=40, zorder=5)
    ax_xz.scatter(student_xyz[-1, 0], student_xyz[-1, 2], c=color, marker="X", s=40, zorder=5)
    ax_xz.set_xlabel("x [m]")
    ax_xz.set_ylabel("z [m]")
    ax_xz.set_title("side-view (xz)", fontsize=9)
    ax_xz.grid(True, alpha=0.3)
    ax_xz.legend(loc="best", fontsize=7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_dir", required=True)
    parser.add_argument("--motion_dir", required=True)
    parser.add_argument("--labels_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_success", type=int, default=4)
    parser.add_argument("--n_fail", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    motion_dir = Path(args.motion_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = json.load(open(args.labels_json))
    motions = {p.stem: p for p in motion_dir.glob("*.npz")}

    success_keys = sorted([k for k, v in labels.items() if v["label"] == "success"])
    fail_keys = sorted([k for k, v in labels.items() if v["label"] == "fail"])
    rng = random.Random(args.seed)
    rng.shuffle(success_keys)
    rng.shuffle(fail_keys)

    picks = ([("success", k) for k in success_keys[:args.n_success]]
             + [("fail", k) for k in fail_keys[:args.n_fail]])
    if not picks:
        print("No motions to plot.")
        return

    n = len(picks)
    cols = 2
    rows = n
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.0 * rows))
    if rows == 1:
        axes = np.array([axes])

    for i, (label, key) in enumerate(picks):
        rollout_npy = rollout_dir / f"policy_{key}.npy"
        motion_npz = motions.get(key)
        if not rollout_npy.exists() or motion_npz is None:
            print(f"[skip] missing inputs for {key}")
            continue
        motion_xyz, student_xyz = _load_pair(rollout_npy, motion_npz)
        _plot_one(axes[i, 0], axes[i, 1], key, label, motion_xyz, student_xyz)

        # Per-motion single-figure version too.
        fig_one, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(11, 4.0))
        _plot_one(ax_xy, ax_xz, key, label, motion_xyz, student_xyz)
        fig_one.tight_layout()
        fig_one.savefig(out_dir / f"{label}_{key}.png", dpi=120)
        plt.close(fig_one)

    fig.tight_layout()
    summary_path = out_dir / "summary.png"
    fig.savefig(summary_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {summary_path} and {len(picks)} per-motion plots in {out_dir}")


if __name__ == "__main__":
    main()
