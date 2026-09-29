#!/bin/bash
# CI runs this inside the holosoma-mujoco docker image (hsmujoco conda env,
# MuJoCo + GPU-accelerated MuJoCo-Warp).
set -ex

cd /workspace/holosoma

# Headless GL for the classic MuJoCo camera renderer (mujoco.Renderer). The CI container has no
# X server/DISPLAY, so the default GLX path dies with "gladLoadGL error". The image ships libEGL,
# and the NVIDIA runtime mounts the driver vendor libraries for offscreen GPU rendering. Set this
# before any mujoco import (pytest subprocesses inherit it). mjwarp uses its own GPU renderer.
export MUJOCO_GL=egl

source scripts/source_mujoco_setup.sh
pip install -e 'src/holosoma[unitree,booster]'
python -c 'import mujoco, mujoco_warp, warp'

# Runs both MuJoCo backends: mujoco_warp cells use the GPU, mujoco_classic cells run on CPU
# within this image. The mujoco_classic/mujoco_warp sub-tags imply the mujoco umbrella (conftest).
marker="mujoco"
if [[ "$HOLOSOMA_MULTIGPU" == "True" ]]; then
   marker="$marker and multi_gpu"
elif [[ "$HOLOSOMA_MULTIGPU" == "False" ]]; then
   marker="$marker and not multi_gpu"
fi

pytest -s --strict-markers --ignore=thirdparty --ignore=src/holosoma_inference -m "$marker"
