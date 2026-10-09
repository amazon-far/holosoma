#!/usr/bin/env bash

ISAACSIM_VERSION=5.1.0
ISAACLAB_REF=v2.3.2
ISAACSIM_SETUP_ID=isaacsim-${ISAACSIM_VERSION}_isaaclab-${ISAACLAB_REF}
MUJOCO_VERSION=3.10.0
# mujoco_warp has no tags or branches. This commit tracks the 3.10.0 line with the batched-camera
# renderer API (create_render_context/render/get_rgb/get_depth) required by WarpBackend.
MUJOCO_WARP_COMMIT=ecaef88917a3c90cd238bf76681ca770f58033df
WARP_LANG_VERSION=1.15.0
