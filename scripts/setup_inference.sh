# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

echo "Setting up inference environment"

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

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/hsinference

SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_inference

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install swig based on OS
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

  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl $MINICONDA_URL -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n hsinference python=3.10 -c conda-forge --override-channels
  fi

  source $CONDA_ROOT/bin/activate hsinference

  # Install libstdcxx-ng to fix the error: `version `GLIBCXX_3.4.32' not found` on Ubuntu 24.04
  # Only needed on Linux (not macOS)
  if [[ $OS == "Linux" ]]; then
    conda install -c conda-forge -y libstdcxx-ng
  fi

  # Install holosoma_inference
  # Note: On macOS, only Unitree SDK is supported (Booster SDK is Linux-only)
  if [[ $OS == "Darwin" ]]; then
    echo "Note: Installing Unitree SDK only (Booster SDK is not supported on macOS)"
    pip install -e $ROOT_DIR/src/holosoma_inference[unitree]
  else
    pip install -e $ROOT_DIR/src/holosoma_inference[unitree,booster]
  fi
  # Setup a few things for ARM64 Linux (G1 Jetson)
  # Otherwise we get this error:
  # /opt/rh/gcc-toolset-14/root/usr/include/c++/14/bits/stl_vector.h:1130: ...
  if [[ $OS == "Linux" && $ARCH == "aarch64" ]]; then
    sudo nvpmodel -m 0 2>/dev/null || true
    pip install pin>=3.8.0
  else
    if [[ ! -d $WORKSPACE_DIR/unitree_sdk2_python ]]; then
      git clone https://github.com/unitreerobotics/unitree_sdk2_python.git $WORKSPACE_DIR/unitree_sdk2_python
    fi
    pip install -e $WORKSPACE_DIR/unitree_sdk2_python/
    $CONDA_ROOT/bin/conda install pinocchio -y -c conda-forge --override-channels
  fi

  cd $ROOT_DIR
  touch $SENTINEL_FILE
fi
