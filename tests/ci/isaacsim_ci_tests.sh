#!/bin/bash
# CI runs this inside holosoma docker
set -ex


cd /workspace/holosoma

source scripts/source_isaacsim_setup.sh
python -m pip install -e 'src/holosoma[unitree,booster]'

python -m pytest -s --strict-markers -m "isaacsim" --ignore=thirdparty --ignore=src/holosoma_inference
