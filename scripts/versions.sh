#!/usr/bin/env bash

# --- Isaac Sim 5.1 lane (conda env hssim, Python 3.11) ---
ISAACSIM_VERSION=5.1.0
ISAACLAB_REF=v2.3.2
ISAACSIM_SETUP_ID=isaacsim-${ISAACSIM_VERSION}_isaaclab-${ISAACLAB_REF}

# --- Isaac Sim 6.2 / Isaac Lab 3.0 GA (uv venv hssim6, Python 3.12) ---
ISAACSIM6_VERSION=6.2.0
ISAACLAB3_BRANCH=v3.0.0
ISAACLAB3_COMMIT=e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4
ISAACSIM6_SETUP_ID=isaacsim-${ISAACSIM6_VERSION}_isaaclab-${ISAACLAB3_COMMIT}
# Native physics, rendering, Torch and Warp versions come from the release's uv.lock.
ISAACSIM6_PYTHON_VERSION=3.12
MUJOCO_VERSION=3.10.0
# mujoco_warp has no tags or branches. This commit tracks the 3.10.0 line with the batched-camera
# renderer API (create_render_context/render/get_rgb/get_depth) required by WarpBackend.
MUJOCO_WARP_COMMIT=ecaef88917a3c90cd238bf76681ca770f58033df
WARP_LANG_VERSION=1.15.0
