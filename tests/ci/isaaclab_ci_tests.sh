#!/bin/bash
# CI runs this inside holosoma docker
set -ex

if [[ $# != 2 || $1 != --backend ]]; then
  echo "Usage: $0 --backend isaaclab2|isaaclab3" >&2
  exit 2
fi
backend=$2
if [[ $backend == isaacsim ]]; then backend=isaaclab2; fi
case "$backend" in
  isaaclab2|isaaclab3) ;;
  *) echo "Unknown Isaac Sim backend: $backend" >&2; exit 2 ;;
esac

cd /workspace/holosoma

case "$backend" in
  isaaclab2) source scripts/source_isaaclab2_setup.sh ;;
  isaaclab3) source scripts/source_isaaclab3_setup.sh ;;
esac
python -m pip install -e 'src/holosoma[unitree,booster]'

preflight_args=(--backend "$backend")
if [[ "$HOLOSOMA_MULTIGPU" == "True" ]]; then
   preflight_args+=(--multigpu)
fi
python scripts/validate_isaaclab_runtime.py "${preflight_args[@]}"

marker="isaacsim"
if [[ "$HOLOSOMA_MULTIGPU" == "True" ]]; then
   marker="$marker and multi_gpu"
elif [[ "$HOLOSOMA_MULTIGPU" == "False" ]]; then
   marker="$marker and not multi_gpu"
fi

python -m pytest -s --strict-markers --isaaclab-backend "$backend" -m "$marker" --ignore=thirdparty --ignore=src/holosoma_inference

# The new runtime also changes Python, NumPy and PyTorch. Exercise the existing
# pure suite in that environment once, alongside the old combined-image job.
if [[ $backend == isaaclab3 && ${HOLOSOMA_MULTIGPU:-False} == False ]]; then
  bash tests/ci/no_sim_ci_tests.sh --backend "$backend"
fi
