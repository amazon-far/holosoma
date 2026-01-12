#!/bin/bash
# CI runs this inside holosoma docker
set -ex

cd /workspace/holosoma

source scripts/source_isaacgym_setup.sh
pip install -e 'src/holosoma[unitree,booster]'
pip install -e src/holosoma_inference

pytest -s --ignore=thirdparty -m "not isaacsim"
