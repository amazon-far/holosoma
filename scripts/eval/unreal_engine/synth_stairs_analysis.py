"""Plot per-motion success rate against synthetic stair parameters.

Reads the per-motion success_rate report (the file
``success_rate_iter<iter>.txt`` written by the SuccessRateCallback) and the
synthetic motion .npz filenames (which encode ``h``, ``w``, ``n_steps``,
direction). Produces:

  * ``success_rate_vs_height.png`` — binned success rate at 0.5m / 0.25m
  * ``success_rate_vs_width.png``
  * ``success_rate_heatmap.png`` — (h, w) success-rate heatmap
  * ``per_motion_results.csv`` — joined table for downstream notebooks
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_NAME_RE = re.compile(
    r"^synth_h(?P<h>\d+)_w(?P<w>\d+)_n(?P<n>\d+)_(?P<dir>up|down)_i\d+$"
)


def _parse_per_motion(report_path: Path) -> dict[str, dict[str, str]]:
    """Return {motion_basename: {"0.5m": "OK"|"FAIL", "0.25m": ...}}."""
    by_thresh: dict[str, dict[str, str]] = {}
    current = None
    with open(report_path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"^--- Per-Motion \[(?P<t>[0-9.]+m)\]", line)
            if m:
                current = m.group("t")
                continue
            if current is None:
                continue
            sm = re.match(r"^\s*(OK|FAIL):\s+(.+)$", line)
            if sm:
                status, path = sm.group(1), sm.group(2)
                base = Path(path).stem
                by_thresh.setdefault(base, {})[current] = status
    return by_thresh


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
    parser.add_argument("--report", required=True, help="success_rate_iter<N>.txt")
    parser.add_argument("--motion_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_thresh = _parse_per_motion(Path(args.report))
    print(f"Parsed {len(by_thresh)} motions from {args.report}")

    rows = []
    for base, status_map in by_thresh.items():
        meta = _parse_filename(base)
        if meta is None:
            continue
        rows.append(
            {
                "motion": base,
                "h": meta["h"],
                "w": meta["w"],
                "n_steps": meta["n_steps"],
                "ascending": meta["ascending"],
                "ok_05m": status_map.get("0.5m") == "OK",
                "ok_025m": status_map.get("0.25m") == "OK",
            }
        )
    print(f"Joined {len(rows)} rows with parameters")

    # CSV dump.
    csv_path = out_dir / "per_motion_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    h = np.array([r["h"] for r in rows])
    w = np.array([r["w"] for r in rows])
    n_steps = np.array([r["n_steps"] for r in rows])
    ok_05 = np.array([r["ok_05m"] for r in rows])
    ok_025 = np.array([r["ok_025m"] for r in rows])
    asc = np.array([r["ascending"] for r in rows])
    print(
        f"Overall: {ok_05.mean():.3f} @ 0.5m, {ok_025.mean():.3f} @ 0.25m"
        f" ({ok_05.sum()}/{len(ok_05)})"
    )
    if asc.any():
        print(f"  Ascending: {ok_05[asc].mean():.3f} ({ok_05[asc].sum()}/{asc.sum()})")
    if (~asc).any():
        print(
            f"  Descending: {ok_05[~asc].mean():.3f} "
            f"({ok_05[~asc].sum()}/{(~asc).sum()})"
        )

    # SR vs height
    h_edges = np.linspace(0.10, 0.30, 6)
    h_centers, sr05_h, n_h = _binned_sr(h, ok_05, h_edges)
    _, sr025_h, _ = _binned_sr(h, ok_025, h_edges)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h_centers, sr05_h, marker="o", label="success @ 0.5m")
    ax.plot(h_centers, sr025_h, marker="x", linestyle="--", alpha=0.6, label="success @ 0.25m")
    for c, s, n in zip(h_centers, sr05_h, n_h):
        ax.annotate(f"n={n}", (c, s), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color="gray")
    ax.set_xlabel("step height h [m]")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Synthetic stairs: success rate vs step height")
    fig.tight_layout()
    fig.savefig(out_dir / "success_rate_vs_height.png", dpi=140)
    print("Wrote success_rate_vs_height.png")
    plt.close(fig)

    # SR vs width
    w_edges = np.linspace(0.30, 0.80, 6)
    w_centers, sr05_w, n_w = _binned_sr(w, ok_05, w_edges)
    _, sr025_w, _ = _binned_sr(w, ok_025, w_edges)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(w_centers, sr05_w, marker="o", label="success @ 0.5m")
    ax.plot(w_centers, sr025_w, marker="x", linestyle="--", alpha=0.6, label="success @ 0.25m")
    for c, s, n in zip(w_centers, sr05_w, n_w):
        ax.annotate(f"n={n}", (c, s), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color="gray")
    ax.set_xlabel("step width w [m]")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Synthetic stairs: success rate vs step width")
    fig.tight_layout()
    fig.savefig(out_dir / "success_rate_vs_width.png", dpi=140)
    print("Wrote success_rate_vs_width.png")
    plt.close(fig)

    # 2D heatmap (h × w)
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
                grid_sr[i, j] = ok_05[mask].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        grid_sr,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        extent=[w_bins[0], w_bins[-1], h_bins[0], h_bins[-1]],
    )
    for i in range(grid_sr.shape[0]):
        for j in range(grid_sr.shape[1]):
            cx = (w_bins[j] + w_bins[j + 1]) / 2
            cy = (h_bins[i] + h_bins[i + 1]) / 2
            n = grid_n[i, j]
            if n == 0:
                txt = "—"
            else:
                txt = f"{grid_sr[i, j]:.2f}\n(n={n})"
            ax.text(cx, cy, txt, ha="center", va="center", fontsize=8)
    ax.set_xlabel("step width w [m]")
    ax.set_ylabel("step height h [m]")
    ax.set_title("Synth stairs success rate @ 0.5m (h × w)")
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(out_dir / "success_rate_heatmap.png", dpi=140)
    print("Wrote success_rate_heatmap.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
