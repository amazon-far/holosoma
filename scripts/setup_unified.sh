#!/bin/bash
# setup_unified.sh
# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

echo "Setting up UNIFIED Holosoma environment (Inference + Retargeting)"

# 1. OS & Architecture Detection
OS=$(uname -s)
ARCH=$(uname -m)

case $ARCH in
  "aarch64"|"arm64") ARCH="aarch64" ;;
  "x86_64") ARCH="x86_64" ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

case $OS in
  "Linux")
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh"
    PACKAGE_MANAGER="apt-get"
    INSTALL_CMD="sudo apt-get install -y"
    ;;
  "Darwin")
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    PACKAGE_MANAGER="brew"
    INSTALL_CMD="brew install"
    ;;
  *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

# 2. Workspace Setup
source ${SCRIPT_DIR}/source_common.sh
# 통합 환경 이름: holosoma_unified
ENV_NAME="hunified"
ENV_ROOT=$CONDA_ROOT/envs/$ENV_NAME
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_unified

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  
  # ---------------------------------------------------------
  # System Dependency Installation (from Inference script)
  # ---------------------------------------------------------
  if [[ $OS == "Linux" ]]; then
    $INSTALL_CMD swig
  elif [[ $OS == "Darwin" ]]; then
    # Install brew if needed
    if ! command -v brew &> /dev/null; then
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      echo >> $HOME/.zprofile
      echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> $HOME/.zprofile
      eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    $INSTALL_CMD swig
  fi

  # ---------------------------------------------------------
  # Conda Installation
  # ---------------------------------------------------------
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl $MINICONDA_URL -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # ---------------------------------------------------------
  # Create Conda Environment
  # Note: Choosing Python 3.10 for better hardware compatibility
  # ---------------------------------------------------------
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    
    # Python 3.10 사용 (Inference 기준)
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n $ENV_NAME python=3.10 -c conda-forge --override-channels
  fi

  source $CONDA_ROOT/bin/activate $ENV_NAME

  # ---------------------------------------------------------
  # Install Inference Dependencies
  # ---------------------------------------------------------
  # Install libstdcxx-ng (Linux fix)
  if [[ $OS == "Linux" ]]; then
      conda install -c conda-forge -y libstdcxx-ng
  fi

  # Unitree & Pinocchio Logic (Platform specific)
  if [[ $OS == "Linux" && $ARCH == "aarch64" ]]; then
    # Jetson / ARM Linux case
    sudo nvpmodel -m 0 2>/dev/null || true
    pip install pin>=3.8.0
  else
    # x86 Linux or macOS
    if [[ ! -d $WORKSPACE_DIR/unitree_sdk2_python ]]; then
      git clone https://github.com/unitreerobotics/unitree_sdk2_python.git $WORKSPACE_DIR/unitree_sdk2_python
    fi
    pip install -e $WORKSPACE_DIR/unitree_sdk2_python/
    # Pinocchio via Conda usually safer for non-Jetson
    conda install pinocchio -y -c conda-forge --override-channels
  fi

  # Install holosoma_inference
  pip install -e $ROOT_DIR/src/holosoma_inference[unitree,booster]

  # ---------------------------------------------------------
  # Install Retargeting Dependencies
  # ---------------------------------------------------------
  echo "Installing Retargeting modules..."
  # pip upgrade for safety
  pip install -U pip
  
  # Install holosoma_retargeting
  # 만약 의존성 충돌이 발생하면 pip가 에러를 뱉거나 경고를 줄 것입니다.
  pip install -e $ROOT_DIR/src/holosoma_retargeting

  cd $ROOT_DIR
  touch $SENTINEL_FILE
  
  echo "Setup Finished! Environment '$ENV_NAME' is ready."
fi