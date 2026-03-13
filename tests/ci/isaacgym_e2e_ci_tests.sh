#!/bin/bash
# CI runs this inside holosoma docker
set -ex

cd /workspace/holosoma

source scripts/source_isaacgym_setup.sh
pip install -e 'src/holosoma[unitree,booster]'
pip install -e src/holosoma_inference

marker="not isaacsim and requires_inference"
if [[ "$HOLOSOMA_MULTIGPU" == "True" ]]; then
   marker="$marker and multi_gpu"
elif [[ "$HOLOSOMA_MULTIGPU" == "False" ]]; then
   marker="$marker and not multi_gpu"
fi

pytest -s --ignore=thirdparty --ignore=src/holosoma_inference -m "$marker"
