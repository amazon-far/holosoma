#!/usr/bin/env bash
ISAACSIM6_ROOT=${ISAACSIM6_ROOT:-${HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}/isaacsim6}
source "$ISAACSIM6_ROOT/hssim6/bin/activate"
export OMNI_KIT_ACCEPT_EULA=1
# SciPy's threaded OpenBLAS at-fork handler can crash during Kit startup.
# Set this before Python imports NumPy/SciPy; CUDA physics is unaffected.
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
