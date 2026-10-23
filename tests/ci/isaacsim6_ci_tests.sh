#!/bin/bash
# CI runs this inside the holosoma-isaacsim6 docker image.
#
# Mirrors isaacsim_ci_tests.sh, with two differences forced by the lane:
#
#   * `source_isaacsim6_setup.sh`, not `source_isaacsim_setup.sh`. The 6.1 lane is a uv venv under
#     its own HOLOSOMA_DEPS_DIR (Isaac Lab 3.0 rejects conda), so the 5.1 activate script would
#     put the wrong interpreter on PATH.
#   * No `pip install -e`. The image already installed holosoma into the venv with the numpy
#     ceiling overridden for this lane (holosoma pins numpy<2, Isaac Lab 3.0 requires >=2);
#     re-installing here would resolve numpy 1 back in and break every Isaac Lab import.
set -ex

cd /workspace/holosoma

source scripts/source_isaacsim6_setup.sh

# Run only this generation's simulator tests; the 5.1 wrappers require a different Isaac Lab.
# `no_sim` also covers the lane's pure unit tests. Running it here as well as in the combined image
# is deliberate -- it is
# what catches an eager `isaaclab` import at module scope, which makes a no_sim test uncollectable
# and, because a collection error aborts the whole run, takes the entire suite down with it.
marker="isaacsim6 or no_sim"
if [[ "$HOLOSOMA_MULTIGPU" == "True" ]]; then
   marker="($marker) and multi_gpu"
elif [[ "$HOLOSOMA_MULTIGPU" == "False" ]]; then
   marker="($marker) and not multi_gpu"
fi

python -m pytest -s --strict-markers --ignore=thirdparty --ignore=src/holosoma_inference -m "$marker"
