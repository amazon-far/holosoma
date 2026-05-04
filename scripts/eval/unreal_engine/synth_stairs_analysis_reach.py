"""Plot reach-target success rate against synthetic stair parameters.

Consumes ``relabel_reach_target.py`` output (a JSON of
``{motion: {"label": "success"|"fail", "progress": float, ...}}``) plus the
``synth_h<HH>_w<WW>_n<NN>_<dir>_i<idx>.npz`` filename schema, and emits:

  * ``reach_success_vs_height.png``
  * ``reach_success_vs_width.png``
  * ``reach_success_heatmap.png``
  * ``progress_histogram.png`` — distribution of how far the student got
  * ``per_motion_results.csv``
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_NAME_RE = re.compile(
    r"^synth_h(?P<h>\d+)_w(?P<w>\d+)_n(?P<n>\d+)_(?P<dir>up|down)_i\d+$"
)


def _parse_filename(base: str) -> dict | None:
    m = _NAME_RE.match(base)
    if not m:
        return None
    return {
        "h": int(m.group("h")) / 100.0,
        "w": int(m.group("w")) / 100.0,
        "n_steps": int(m.group("n")),
        "ascending": m.group("dir") == "up",
    }


def _binned_sr(values_x: np.ndarray, ok_mask: np.ndarray, edges: np.ndarray):
    centers, sr, count = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values_x >= lo) & (values_x < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        centers.append((lo + hi) / 2)
        sr.append(float(ok_mask[mask].mean()))
        count.append(n)
    return np.array(centers), np.array(sr), np.array(count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="labels.json from relabel_reach_target.py")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.labels) as f:
        labels = json.load(f)
    print(f"Loaded {len(labels)} labelled motions")

    rows = []
    for motion, info in labels.items():
        meta = _parse_filename(motion)
        if meta is None:
            continue
        rows.append({
            "motion": motion,
            "h": meta["h"],
            "w": meta["w"],
            "n_steps": meta["n_steps"],
            "ascending": meta["ascending"],
            "label": info["label"],
            "progress": float(info.get("progress", 0.0)),
            "z_diff": float(info.get("z_diff", 0.0)),
            "z_ok": bool(info.get("z_ok", False)),
            "fell": bool(info.get("fell", False)),
        })
    print(f"Joined {len(rows)} rows with parameters")

    if not rows:
        print("No rows to plot.")
        return

    csv_path = out_dir / "per_motion_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    h = np.array([r["h"] for r in rows])
    w = np.array([r["w"] for r in rows])
    progress = np.array([r["progress"] for r in rows])
    success = np.array([r["label"] == "success" for r in rows])
    fell = np.array([r["fell"] for r in rows])
    z_ok = np.array([r["z_ok"] for r in rows])
    asc = np.array([r["ascending"] for r in rows])

    print(
        f"Overall: {success.mean():.3f} success "
        f"({success.sum()}/{len(success)}); "
        f"fell={fell.mean():.3f}, z_ok={z_ok.mean():.3f}, "
        f"mean progress={progress.mean():.3f}"
    )
    if asc.any():
        print(
            f"  Ascending: success={success[asc].mean():.3f} "
            f"({success[asc].sum()}/{asc.sum()}), mean progress={progress[asc].mean():.3f}"
        )
    if (~asc).any():
        print(
            f"  Descending: success={success[~asc].mean():.3f} "
            f"({success[~asc].sum()}/{(~asc).sum()}), mean progress={progress[~asc].mean():.3f}"
        )

    # Plot 1: SR vs height (binned)
    h_edges = np.linspace(0.10, 0.30, 6)
    h_centers, sr_h, n_h = _binned_sr(h, success, h_edges)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h_centers, sr_h, marker="o", label="reach success")
    for c, s, n in zip(h_centers, sr_h, n_h):
        ax.annotate(f"n={n}", (c, s), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color="gray")
    ax.set_xlabel("step height h [m]")
    ax.set_ylabel("success rate (didn't fall AND reached goal_x)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Synthetic stairs: reach-target success vs step height")
    fig.tight_layout()
    fig.savefig(out_dir / "reach_success_vs_height.png", dpi=140)
    plt.close(fig)
    print("Wrote reach_success_vs_height.png")

    # Plot 2: SR vs width
    w_edges = np.linspace(0.30, 0.80, 6)
    w_centers, sr_w, n_w = _binned_sr(w, success, w_edges)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(w_centers, sr_w, marker="o", label="reach success")
    for c, s, n in zip(w_centers, sr_w, n_w):
        ax.annotate(f"n={n}", (c, s), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color="gray")
    ax.set_xlabel("step width w [m]")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Synthetic stairs: reach-target success vs step width")
    fig.tight_layout()
    fig.savefig(out_dir / "reach_success_vs_width.png", dpi=140)
    plt.close(fig)
    print("Wrote reach_success_vs_width.png")

    # Plot 3: 2D heatmap (h × w)
    h_bins = np.linspace(0.10, 0.30, 5)
    w_bins = np.linspace(0.30, 0.80, 5)
    grid_sr = np.full((len(h_bins) - 1, len(w_bins) - 1), np.nan)
    grid_n = np.zeros((len(h_bins) - 1, len(w_bins) - 1), dtype=int)
    for i in range(len(h_bins) - 1):
        for j in range(len(w_bins) - 1):
            mask = (h >= h_bins[i]) & (h < h_bins[i + 1]) & \
                   (w >= w_bins[j]) & (w < w_bins[j + 1])
            n = int(mask.sum())
            grid_n[i, j] = n
            if n > 0:
                grid_sr[i, j] = success[mask].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid_sr, origin="lower", aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1,
                   extent=[w_bins[0], w_bins[-1], h_bins[0], h_bins[-1]])
    for i in range(grid_sr.shape[0]):
        for j in range(grid_sr.shape[1]):
            cx = (w_bins[j] + w_bins[j + 1]) / 2
            cy = (h_bins[i] + h_bins[i + 1]) / 2
            n = grid_n[i, j]
            txt = "—" if n == 0 else f"{grid_sr[i, j]:.2f}\n(n={n})"
            ax.text(cx, cy, txt, ha="center", va="center", fontsize=8)
    ax.set_xlabel("step width w [m]")
    ax.set_ylabel("step height h [m]")
    ax.set_title("Reach-target success rate (h × w)")
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(out_dir / "reach_success_heatmap.png", dpi=140)
    plt.close(fig)
    print("Wrote reach_success_heatmap.png")

    # Plot 4: progress histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(progress, bins=20, range=(-0.5, 1.5), color="tab:blue", alpha=0.7,
            edgecolor="white")
    ax.axvline(1.0, color="green", linestyle="--", alpha=0.6, label="reached goal")
    ax.axvline(0.9, color="orange", linestyle="--", alpha=0.6, label="success threshold (0.9)")
    ax.set_xlabel("progress  (final_x − start_x) / (goal_x − start_x)")
    ax.set_ylabel("count")
    ax.set_title("Synthetic stairs: distribution of student progress along the stair")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "progress_histogram.png", dpi=140)
    plt.close(fig)
    print("Wrote progress_histogram.png")


if __name__ == "__main__":
    main()
