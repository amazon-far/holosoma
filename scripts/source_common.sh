#!/usr/bin/env bash

# Exported so Python can read them too. The Dockerfiles set HOLOSOMA_DEPS_DIR as an ENV, so
# code that reads it worked in CI but not on a dev machine that had merely sourced an activate
# script -- which is how the USD conversion cache ended up resolving into the source tree.
export HOLOSOMA_DEPS_DIR=${HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}
export WORKSPACE_DIR=$HOLOSOMA_DEPS_DIR
# Left unexported: only the shell scripts use it, and the name is referenced by some conda init
# scripts, so putting it in the environment is avoidable risk.
CONDA_ROOT=$HOLOSOMA_DEPS_DIR/miniconda3
