#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsgym}
ISAACGYM_RELEASE=preview4
ISAACGYM_URL=https://developer.nvidia.com/isaac-gym-preview-4
PYTHON_VERSION=3.8
CUDNN_WHEEL_URL=https://files.pythonhosted.org/packages/9f/fd/713452cd72343f682b1c7b9321e23829f00b842ceaedcda96e742ea0b0b3/nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl
CUDNN_WHEEL_SHA256=165764f44ef8c61fcdfdfdbe769d687e06374059fbb388b6c89ecb0e28793a6f

source "${SCRIPT_DIR}/source_common.sh"

ENV_ROOT=$CONDA_ROOT/envs/$CONDA_ENV_NAME
ISAACGYM_DIR=$WORKSPACE_DIR/isaacgym
ISAACGYM_ARCHIVE_PATH=${ISAACGYM_ARCHIVE_PATH:-$WORKSPACE_DIR/IsaacGym_Preview_4_Package.tar.gz}
RUNTIME_CONSTRAINTS=$SCRIPT_DIR/isaacgym_constraints.txt
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_${CONDA_ENV_NAME}_isaacgym-${ISAACGYM_RELEASE}_python-${PYTHON_VERSION}

echo "conda environment name is set to: $CONDA_ENV_NAME"

mkdir -p "$WORKSPACE_DIR"

if [[ -f $SENTINEL_FILE ]]; then
  exit 0
fi

if [[ ! -d $CONDA_ROOT ]]; then
  mkdir -p "$CONDA_ROOT"
  curl -fL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o "$CONDA_ROOT/miniconda.sh"
  bash "$CONDA_ROOT/miniconda.sh" -b -u -p "$CONDA_ROOT"
  rm "$CONDA_ROOT/miniconda.sh"
fi

if [[ ! -d $ENV_ROOT ]]; then
  "$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  "$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  # Solve with libmamba, conda's own default solver, instead of installing mamba into base:
  # that install pulls conda-forge's conda into base and breaks the bundled ToS plugin.
  "$CONDA_ROOT/bin/conda" create -y --solver libmamba -n "$CONDA_ENV_NAME" \
    "python=$PYTHON_VERSION" -c conda-forge --override-channels
fi

ACTUAL_PYTHON_VERSION=$("$ENV_ROOT/bin/python" -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $ACTUAL_PYTHON_VERSION != "$PYTHON_VERSION" ]]; then
  echo "Isaac Gym Preview 4 requires Python $PYTHON_VERSION; $CONDA_ENV_NAME uses $ACTUAL_PYTHON_VERSION."
  echo "Remove $ENV_ROOT and rerun this script."
  exit 1
fi

source "$CONDA_ROOT/bin/activate" "$CONDA_ENV_NAME"
export LD_LIBRARY_PATH="${ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

conda install -c conda-forge -y libstdcxx-ng ffmpeg libiconv

python -m pip install --upgrade pip
# PyTorch's cu121 index no longer lists this transitive dependency, but the
# exact wheel required by Torch 2.4.1 remains available from PyPI.
python -m pip install --no-deps "${CUDNN_WHEEL_URL}#sha256=${CUDNN_WHEEL_SHA256}"
python -m pip install \
  "torch==2.4.1" "torchvision==0.19.1" \
  --index-url https://download.pytorch.org/whl/cu121

if [[ ! -d $ISAACGYM_DIR ]]; then
  if [[ -f $ISAACGYM_ARCHIVE_PATH ]] && ! tar -tzf "$ISAACGYM_ARCHIVE_PATH" >/dev/null; then
    echo "Removing incomplete Isaac Gym archive: $ISAACGYM_ARCHIVE_PATH"
    rm -f "$ISAACGYM_ARCHIVE_PATH"
  fi
  if [[ ! -f $ISAACGYM_ARCHIVE_PATH ]]; then
    mkdir -p "$(dirname "$ISAACGYM_ARCHIVE_PATH")"
    download_tmp=$(mktemp "${ISAACGYM_ARCHIVE_PATH}.tmp.XXXXXX")
    trap 'rm -f "$download_tmp"' EXIT
    curl -fL --retry 3 "$ISAACGYM_URL" -o "$download_tmp"
    tar -tzf "$download_tmp" >/dev/null
    mv "$download_tmp" "$ISAACGYM_ARCHIVE_PATH"
    trap - EXIT
  fi
  tar -xzf "$ISAACGYM_ARCHIVE_PATH" -C "$WORKSPACE_DIR"
fi

python -m pip install -c "$RUNTIME_CONSTRAINTS" -e "$ISAACGYM_DIR/python"
python -m pip install -c "$RUNTIME_CONSTRAINTS" -e "$ROOT_DIR/src/holosoma[unitree,booster]"

python -c 'from isaacgym import gymapi; print(f"Isaac Gym binding: {gymapi.__file__}")'
touch "$SENTINEL_FILE"
