#!/usr/bin/env bash

# --- Isaac Sim 5.1 lane (conda env hssim, Python 3.11) ---
ISAACSIM_VERSION=5.1.0
ISAACLAB_REF=v2.3.2
ISAACSIM_SETUP_ID=isaacsim-${ISAACSIM_VERSION}_isaaclab-${ISAACLAB_REF}

# --- Isaac Sim 6.1 lane (uv venv hssim6, Python 3.12) ---
# Isaac Lab 3.0 is pre-GA and release/3.0.0 moves daily, so pin the commit rather than the
# branch (same idiom as MUJOCO_WARP_COMMIT below). Re-pin deliberately, and re-run
# scripts/isaac_capability_report.py and the isaacsim6 tests when doing so.
ISAACSIM6_VERSION=6.1.0
ISAACLAB3_BRANCH=release/3.0.0
ISAACLAB3_COMMIT=864e7c240a9b127148f5c596caa900b716f8c3a6
ISAACSIM6_SETUP_ID=isaacsim-${ISAACSIM6_VERSION}_isaaclab-${ISAACLAB3_COMMIT}
# Isaac Lab 3.0 requires Python 3.12 (requires-python = ">=3.12,<3.13"). Its own pins -- torch
# 2.12.0 on the cu130 index, warp-lang 1.17.0, numpy>=2 -- are declared in the IsaacLab
# checkout's pyproject.toml and uv.lock, so this lane resolves them from there rather than
# restating them here. Restating would create a second source of truth that silently drifts.
ISAACSIM6_PYTHON_VERSION=3.12
MUJOCO_VERSION=3.10.0
# mujoco_warp has no tags or branches. This commit tracks the 3.10.0 line with the batched-camera
# renderer API (create_render_context/render/get_rgb/get_depth) required by WarpBackend.
MUJOCO_WARP_COMMIT=ecaef88917a3c90cd238bf76681ca770f58033df
WARP_LANG_VERSION=1.15.0
