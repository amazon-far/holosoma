#!/bin/bash
# CI runs this inside holosoma docker on a CPU-only runner.
# Pure tests with no simulator backend (config, utils, distributions, pure unit tests).
# The isaacsim conda env is sourced only for its CPU torch install; no simulator boots here.
set -ex

cd /workspace/holosoma

source scripts/source_isaacsim_setup.sh
python -m pip install -e 'src/holosoma[unitree,booster]'
python -m pip install -e src/holosoma_inference

python -m pytest -s --strict-markers -m "no_sim" --ignore=thirdparty --ignore=src/holosoma_inference
