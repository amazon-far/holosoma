#!/bin/bash
# CI runs this inside holosoma docker
set -ex

if [[ $# != 2 || $1 != --backend ]]; then
  echo "Usage: $0 --backend isaacsim5|isaacsim6" >&2
  exit 2
fi
backend=$2
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

preflight_args=(--backend "$backend")
if [[ "$HOLOSOMA_MULTIGPU" == "True" ]]; then
   preflight_args+=(--multigpu)
fi
python scripts/validate_isaacsim_runtime.py "${preflight_args[@]}"

marker="isaacsim"
if [[ "$HOLOSOMA_MULTIGPU" == "True" ]]; then
   marker="$marker and multi_gpu"
elif [[ "$HOLOSOMA_MULTIGPU" == "False" ]]; then
   marker="$marker and not multi_gpu"
fi

python -m pytest -s --strict-markers --isaacsim-backend "$backend" -m "$marker" --ignore=thirdparty --ignore=src/holosoma_inference

# The new runtime also changes Python, NumPy and PyTorch. Exercise the existing
# pure suite in that environment once, alongside the old combined-image job.
if [[ $backend == isaacsim6 && ${HOLOSOMA_MULTIGPU:-False} == False ]]; then
  bash tests/ci/no_sim_ci_tests.sh --backend "$backend"
fi
