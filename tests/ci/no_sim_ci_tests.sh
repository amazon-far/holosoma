#!/bin/bash
# CI runs this inside holosoma docker on a CPU-only runner.
# Pure tests with no simulator backend (config, utils, distributions, pure unit tests).
# The isaacsim conda env is sourced only for its CPU torch install; no simulator boots here.
set -ex

cd /workspace/holosoma

case "${HOLOSOMA_ISAACSIM_BACKEND:-isaaclab2}" in
  isaaclab2) source scripts/source_isaacsim_setup.sh ;;
  isaaclab3) source scripts/source_isaacsim3_setup.sh ;;
  *) echo "Unknown Isaac Sim backend: $HOLOSOMA_ISAACSIM_BACKEND" >&2; exit 1 ;;
esac
python -m pip install -e 'src/holosoma[unitree,booster]'
python -m pip install -e src/holosoma_inference

python -m pytest -s --strict-markers -m "no_sim" --ignore=thirdparty --ignore=src/holosoma_inference
