#!/bin/bash
# CI runs this inside holosoma docker
set -ex


cd /workspace/holosoma

case "${HOLOSOMA_ISAACSIM_BACKEND:-isaaclab2}" in
  isaaclab2) source scripts/source_isaacsim_setup.sh ;;
  isaaclab3) source scripts/source_isaacsim3_setup.sh ;;
  *) echo "Unknown Isaac Sim backend: $HOLOSOMA_ISAACSIM_BACKEND" >&2; exit 1 ;;
esac
python -m pip install -e 'src/holosoma[unitree,booster]'

preflight_args=(--backend "${HOLOSOMA_ISAACSIM_BACKEND:-isaaclab2}")
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

python -m pytest -s --strict-markers -m "$marker" --ignore=thirdparty --ignore=src/holosoma_inference

# The new runtime also changes Python, NumPy and PyTorch. Exercise the existing
# pure suite in that environment once, alongside the old combined-image job.
if [[ ${HOLOSOMA_ISAACSIM_BACKEND:-isaaclab2} == isaaclab3 && ${HOLOSOMA_MULTIGPU:-False} == False ]]; then
  bash tests/ci/no_sim_ci_tests.sh
fi
