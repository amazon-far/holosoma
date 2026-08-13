#!/bin/bash
# CI runs this inside holosoma docker
set -ex


cd /workspace/holosoma

source scripts/source_isaacsim_setup.sh
python -m pip install -e 'src/holosoma[unitree,booster]'
python -m pip install -e src/holosoma_inference

marker="isaacsim"
if [[ "$HOLOSOMA_MULTIGPU" == "True" ]]; then
   marker="$marker and multi_gpu"
elif [[ "$HOLOSOMA_MULTIGPU" == "False" ]]; then
   marker="$marker and not multi_gpu"
fi

python -m pytest -s --strict-markers -m "$marker" --ignore=thirdparty --ignore=src/holosoma_inference
