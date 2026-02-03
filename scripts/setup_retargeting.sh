# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/hsretargeting
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_retargeting

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT

    # Detect OS and arch
    OS_NAME="$(uname -s)"
    ARCH_NAME="$(uname -m)"

    # Decide installer name based on OS/arch
    if [[ "$OS_NAME" == "Linux" ]]; then
      MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$OS_NAME" == "Darwin" ]]; then
      if [[ "$ARCH_NAME" == "arm64" ]]; then
        # Apple Silicon
        MINICONDA_INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
      else
        # Intel Mac
        MINICONDA_INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
      fi
    else
      echo "Unsupported OS: $OS_NAME"
      exit 1
    fi

    curl "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" -o "$CONDA_ROOT/miniconda.sh"
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n hsretargeting python=3.11 -c conda-forge --override-channels
  fi

  source $CONDA_ROOT/bin/activate hsretargeting

  # Install holosoma_retargeting
  pip install -U pip
  pip install -e $ROOT_DIR/src/holosoma_retargeting
  touch $SENTINEL_FILE
fi
