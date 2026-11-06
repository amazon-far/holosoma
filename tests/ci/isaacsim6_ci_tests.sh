#!/usr/bin/env bash
set -euo pipefail
cd /workspace/holosoma
source scripts/source_isaacsim6_setup.sh
export DISPLAY= WAYLAND_DISPLAY= HEADLESS=1
marker="(isaacsim or isaacsim6)"
if [[ "${HOLOSOMA_MULTIGPU:-False}" == "True" ]]; then
    marker="$marker and multi_gpu"
else
    marker="$marker and not multi_gpu"
fi
for preset in isaacsim6 isaacsim6-newton isaacsim6-ovphysx; do
    python -m pytest -s --strict-markers --isaac-preset="$preset" -m "$marker" \
        --ignore=thirdparty --ignore=src/holosoma_inference
done
