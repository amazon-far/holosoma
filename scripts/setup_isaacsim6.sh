#!/usr/bin/env bash
# Setup the hssim6 environment -- Isaac Sim 6.1.0 + Isaac Lab 3.0, Python 3.12.
#
# This lane is deliberately NOT a conda environment, unlike hssim. Isaac Lab 3.0 removed
# `isaaclab.sh --conda` (the script is deprecated outright, to be deleted in 3.1) and actively
# rejects conda: a downloaded Isaac Sim package combined with CONDA_PREFIX is a hard error, and
# its venv-on-bundled-python check requires CONDA_PREFIX to be empty. So hssim6 is a uv venv.
#
# Dependency resolution is delegated to the IsaacLab checkout's own pyproject.toml + uv.lock via
# `uv sync`. That matters: in 3.0 the member packages declare `dependencies = []` and every real
# pin (torch 2.12.0 from the cu130 index, warp-lang 1.17.0, numpy>=2, the 12 editable members)
# lives in the root project. Installing members directly would therefore install code with no
# dependencies, and restating the 63 root dependencies here would be a second source of truth.
#
# holosoma is then installed into the same venv with its numpy ceiling overridden. holosoma pins
# numpy>=1.23.5,<2 while Isaac Lab 3.0 requires numpy>=2, and that conflict is resolved for this
# lane only -- holosoma's published metadata is untouched, so the hssim / hsmujoco lanes
# keep numpy 1.
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Separate workspace from the 5.1 lane: its own IsaacLab checkout, venv and caches. The two
# generations cannot share an IsaacLab clone (incompatible refs) and must not share a sentinel.
#
# This MUST precede source_common.sh, which defaults HOLOSOMA_DEPS_DIR to the 5.1 lane's
# ~/.holosoma_deps and exports it. Setting it afterwards is a silent no-op that aims this lane at
# the other lane's workspace. An explicit caller value (as the Dockerfiles set) still wins.
export HOLOSOMA_DEPS_DIR=${HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps_sim6}

source "${SCRIPT_DIR}/source_common.sh"
source "${SCRIPT_DIR}/versions.sh"

WORKSPACE_DIR=$HOLOSOMA_DEPS_DIR
VENV_NAME=${VENV_NAME:-hssim6}
VENV_ROOT=$WORKSPACE_DIR/$VENV_NAME
ISAACLAB_DIR=$WORKSPACE_DIR/IsaacLab
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_${VENV_NAME}_${ISAACSIM6_SETUP_ID}

if [[ -f $SENTINEL_FILE ]]; then
  echo "hssim6 already set up ($SENTINEL_FILE); nothing to do."
  exit 0
fi

if ! command -v uv >/dev/null; then
  echo "uv is required for the Isaac Sim 6.1 lane but was not found on PATH."
  echo "Install it from https://docs.astral.sh/uv/ and re-run this script."
  exit 1
fi

mkdir -p "$WORKSPACE_DIR"

# Pin the exact commit, not the branch. release/3.0.0 is pre-GA and advances daily, so a
# branch-tracking clone would silently change the environment between two runs of this script.
# Compare against the commit rather than `rev-parse $branch`, which drifts with the remote.
if [[ ! -d $ISAACLAB_DIR ]]; then
  git clone --branch "$ISAACLAB3_BRANCH" https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
  git -C "$ISAACLAB_DIR" checkout --detach "$ISAACLAB3_COMMIT"
elif [[ $(git -C "$ISAACLAB_DIR" rev-parse HEAD) != "$ISAACLAB3_COMMIT" ]]; then
  echo "IsaacLab at $ISAACLAB_DIR is not at $ISAACLAB3_COMMIT."
  echo "Run scripts/reset_isaacsim6.sh before rebuilding this environment."
  exit 1
fi

# Place the venv under the workspace rather than IsaacLab/.venv (uv's default) so it is
# lane-named and survives re-cloning IsaacLab.
export UV_PROJECT_ENVIRONMENT=$VENV_ROOT

# `--extra isaacsim` adds Isaac Sim itself: in 3.0 `isaacsim` is an optional extra, since the
# base install targets Newton. holosoma needs the Kit runtime, so the extra is required.
cd "$ISAACLAB_DIR"
uv sync --extra isaacsim --python "$ISAACSIM6_PYTHON_VERSION"

# Override rather than --no-deps: this resolves holosoma's full dependency set while forcing the
# one genuinely incompatible pin, so the lane cannot silently drift from pyproject.toml the way a
# hand-maintained dependency list would.
NUMPY_OVERRIDE=$WORKSPACE_DIR/holosoma-overrides.txt
cat > "$NUMPY_OVERRIDE" <<'EOF'
# Isaac Lab 3.0 requires numpy>=2; holosoma's own pin is numpy>=1.23.5,<2. Overridden for the
# Isaac Sim 6.1 lane only. See scripts/setup_isaacsim6.sh.
numpy>=2
EOF

uv pip install --python "$VENV_ROOT/bin/python" --overrides "$NUMPY_OVERRIDE" \
  -e "$ROOT_DIR/src/holosoma[unitree,booster]"

touch "$SENTINEL_FILE"

set +x
echo
echo "hssim6 ready. Activate it with:"
echo "  source scripts/source_isaacsim6_setup.sh"
