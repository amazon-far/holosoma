#!/bin/bash
# CI runs this inside holosoma docker
set -ex


cd /workspace/holosoma

source scripts/source_isaacsim_setup.sh
python -m pip install -e 'src/holosoma[unitree,booster]'
python -m pip install -e src/holosoma_inference

python -m pytest -s -m "isaacsim" --ignore=holosoma/holosoma/envs/legged_base_task/tests/ --ignore=thirdparty
