#!/bin/bash
# CI runs this inside the holosoma-mujoco docker image (hsmujoco conda env,
# MuJoCo + GPU-accelerated MuJoCo-Warp).
set -ex

cd /workspace/holosoma

# Headless GL for the classic MuJoCo camera renderer (mujoco.Renderer). The CI container has no
# X server/DISPLAY, so the default GLX path dies with "gladLoadGL error"; the NVIDIA base image
# (nvcr.io/nvidia/isaac-sim) ships libEGL + the GLVND nvidia vendor, so EGL renders offscreen on
# the GPU. Verified on GPU hardware headless. Set before any mujoco import (pytest subprocesses
# inherit it). mjwarp uses its own GPU renderer and is unaffected.
export MUJOCO_GL=egl

source scripts/source_mujoco_setup.sh
pip install -e 'src/holosoma[unitree,booster]'
pip install -e src/holosoma_inference

# Runs both MuJoCo backends: mujoco_warp cells use the GPU, mujoco_classic cells run on CPU
# within this image. The mujoco_classic/mujoco_warp sub-tags imply the mujoco umbrella (conftest).
marker="mujoco"
if [[ "$HOLOSOMA_MULTIGPU" == "True" ]]; then
   marker="$marker and multi_gpu"
elif [[ "$HOLOSOMA_MULTIGPU" == "False" ]]; then
   marker="$marker and not multi_gpu"
fi

pytest -s --strict-markers --ignore=thirdparty --ignore=src/holosoma_inference -m "$marker"
