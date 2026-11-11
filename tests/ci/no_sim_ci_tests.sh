#!/bin/bash
# CI runs this inside holosoma docker on a CPU-only runner.
# Pure tests with no simulator backend (config, utils, distributions, pure unit tests).
# The isaacsim conda env is sourced only for its CPU torch install; no simulator boots here.
set -ex

backend=isaacsim5
if [[ $# != 0 ]]; then
  if [[ $# != 2 || $1 != --backend ]]; then
    echo "Usage: $0 [--backend isaacsim5|isaacsim6]" >&2
    exit 2
  fi
  backend=$2
fi
case "$backend" in
  isaacsim5|isaacsim6) ;;
  *) echo "Unknown Isaac Sim backend: $backend" >&2; exit 2 ;;
esac

cd /workspace/holosoma

case "$backend" in
  isaacsim5) source scripts/source_isaacsim_setup.sh ;;
  isaacsim6) source scripts/source_isaacsim6_setup.sh ;;
esac
python -m pip install -e 'src/holosoma[unitree,booster]'
python -m pip install -e src/holosoma_inference

python -m pytest -s --strict-markers --isaacsim-backend "$backend" -m "no_sim" --ignore=thirdparty --ignore=src/holosoma_inference
