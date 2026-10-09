#!/usr/bin/env bash
set -e

ISAACSIM_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ISAACSIM_ROOT_DIR=$(dirname "$ISAACSIM_SCRIPT_DIR")
source "${ISAACSIM_SCRIPT_DIR}/source_common.sh"
source "${ISAACSIM_SCRIPT_DIR}/versions.sh"

CONDA_ENV_NAME=${CONDA_ENV_NAME:-hssim}
ISAACSIM_ENV_ROOT=$CONDA_ROOT/envs/$CONDA_ENV_NAME
ISAACLAB_DIR=$WORKSPACE_DIR/IsaacLab
ISAACSIM_RUNTIME_CONSTRAINTS=$ISAACSIM_SCRIPT_DIR/isaacsim_constraints.txt
ISAACSIM_SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_${CONDA_ENV_NAME}_${ISAACSIM_SETUP_ID}

isaacsim_setup_complete() {
  [[ -f $ISAACSIM_SENTINEL_FILE ]]
}

isaacsim_ensure_miniconda() {
  mkdir -p "$WORKSPACE_DIR"

  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p "$CONDA_ROOT"
    curl -fL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -o "$CONDA_ROOT/miniconda.sh"
    bash "$CONDA_ROOT/miniconda.sh" -b -u -p "$CONDA_ROOT"
    rm "$CONDA_ROOT/miniconda.sh"
  fi

  "$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  "$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
}

isaacsim_ensure_standalone_conda_environment() {
  if [[ ! -d $ISAACSIM_ENV_ROOT ]]; then
    # Solve with libmamba, conda's own default solver, instead of installing mamba into base:
    # that install pulls conda-forge's conda into base and breaks the bundled ToS plugin.
    "$CONDA_ROOT/bin/conda" create -y --solver libmamba \
      -n "$CONDA_ENV_NAME" python=3.11 -c conda-forge --override-channels
  fi

  actual_python_version=$("$ISAACSIM_ENV_ROOT/bin/python" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  if [[ $actual_python_version != "3.11" ]]; then
    echo "$CONDA_ENV_NAME requires Python 3.11; the existing environment uses $actual_python_version."
    echo "Remove $ISAACSIM_ENV_ROOT and rerun this script."
    exit 1
  fi
}

isaacsim_validate_image_runtime() {
  local runtime_root=$1
  local runtime_version

  if [[ ! -f $runtime_root/setup_conda_env.sh || ! -f $runtime_root/VERSION ]]; then
    echo "Isaac Sim runtime not found at $runtime_root."
    exit 1
  fi

  runtime_version=$(head -n 1 "$runtime_root/VERSION")
  if [[ $runtime_version != "$ISAACSIM_VERSION"* ]]; then
    echo "Expected Isaac Sim $ISAACSIM_VERSION at $runtime_root; found $runtime_version."
    exit 1
  fi
}

isaacsim_prepare_isaaclab() {
  if [[ ! -d $ISAACLAB_DIR ]]; then
    git clone https://github.com/isaac-sim/IsaacLab.git --branch "$ISAACLAB_REF" "$ISAACLAB_DIR"
  elif [[ $(git -C "$ISAACLAB_DIR" rev-parse HEAD) != $(git -C "$ISAACLAB_DIR" rev-parse "$ISAACLAB_REF") ]]; then
    echo "IsaacLab is not at $ISAACLAB_REF. Run scripts/reset_isaacsim.sh before rebuilding this environment."
    exit 1
  fi

  if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
    ln -sfn "$ISAACSIM_ROOT_PATH" "$ISAACLAB_DIR/_isaac_sim"
  fi
}

isaacsim_ensure_image_conda_environment() {
  # Isaac Lab owns the activation hooks that expose its bundled Isaac Sim
  # runtime inside a regular Python 3.11 Conda environment.
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  "$ISAACLAB_DIR/isaaclab.sh" --conda "$CONDA_ENV_NAME"
}

isaacsim_activate_environment() {
  source "$CONDA_ROOT/bin/activate" "$CONDA_ENV_NAME"
}

isaacsim_install_environment_dependencies() {
  "$CONDA_ROOT/bin/conda" install -n "$CONDA_ENV_NAME" -c conda-forge -y \
    ffmpeg libiconv libglu
}

isaacsim_install_delta() {
  if isaacsim_setup_complete; then
    return
  fi

  if ! python -c 'import isaacsim, torch' >/dev/null; then
    echo "Isaac Sim and PyTorch must be available before installing the Holosoma delta."
    exit 1
  fi

  if ! command -v cmake >/dev/null || ! command -v make >/dev/null || ! command -v gcc >/dev/null; then
    if command -v sudo >/dev/null && sudo -n true >/dev/null 2>&1; then
      sudo apt install -y cmake build-essential
    else
      conda install -c conda-forge -y cmake compilers
    fi
  fi

  cd "$ISAACLAB_DIR"
  python -m pip install 'setuptools<81'
  echo 'setuptools<81' > build-constraints.txt
  export PIP_BUILD_CONSTRAINT
  PIP_BUILD_CONSTRAINT=$(realpath build-constraints.txt)
  sed -i 's/flatdict==4.0.1/flatdict==4.1.0/' source/isaaclab/setup.py

  # Holosoma needs Isaac Lab's core and asset definitions, not its optional
  # task and reinforcement-learning stacks.
  python -m pip install -c "$ISAACSIM_RUNTIME_CONSTRAINTS" -e "$ISAACLAB_DIR/source/isaaclab"
  python -m pip install -c "$ISAACSIM_RUNTIME_CONSTRAINTS" -e "$ISAACLAB_DIR/source/isaaclab_assets"
  python -m pip install h5py
  unset PIP_BUILD_CONSTRAINT

  python -m pip install --upgrade pip
  python -m pip install -c "$ISAACSIM_RUNTIME_CONSTRAINTS" \
    -e "$ISAACSIM_ROOT_DIR/src/holosoma[unitree,booster]"

  touch "$ISAACSIM_SENTINEL_FILE"
}

isaacsim_delta_main() {
  if [[ -z ${ISAACSIM_ROOT_PATH:-} ]]; then
    echo "ISAACSIM_ROOT_PATH must point to an existing Isaac Sim image runtime."
    echo "Use setup_isaacsim.sh to install Isaac Sim into a new environment."
    exit 1
  fi

  if isaacsim_setup_complete; then
    return
  fi

  isaacsim_validate_image_runtime "$ISAACSIM_ROOT_PATH"
  isaacsim_ensure_miniconda
  isaacsim_prepare_isaaclab
  isaacsim_ensure_image_conda_environment
  isaacsim_install_environment_dependencies
  isaacsim_activate_environment
  isaacsim_install_delta
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  isaacsim_delta_main "$@"
fi
