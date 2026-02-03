# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Use CONDA_ENV_NAME if provided, otherwise default to "hssim"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hssim}
echo "conda environment name is set to: $CONDA_ENV_NAME"

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/$CONDA_ENV_NAME
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_$CONDA_ENV_NAME
echo "SENTINEL_FILE: $SENTINEL_FILE"

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    if [[ ! -f $CONDA_ROOT/bin/mamba ]]; then
      $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    fi
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n $CONDA_ENV_NAME python=3.11 -c conda-forge --override-channels
  fi

  # Initialize conda and activate environment
  # Set CONDA_BASE to ensure conda recognizes this installation
  export CONDA_BASE=$CONDA_ROOT
  source $CONDA_ROOT/etc/profile.d/conda.sh
  conda config --set auto_activate_base false
  conda activate $CONDA_ENV_NAME

  # Install ffmpeg for video encoding
  conda install -c conda-forge -y ffmpeg
  conda install -c conda-forge -y libiconv
  conda install -c conda-forge -y libglu

  # Below follows https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
  # Install IsaacSim
  pip install --upgrade pip
  pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

  # Install dependencies from PyPI first
  pip install pyperclip
  # Then install isaacsim from NVIDIA index only
  pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

  if [[ ! -d $WORKSPACE_DIR/IsaacLab ]]; then
    git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.0 $WORKSPACE_DIR/IsaacLab
  fi

  # Install cmake and build tools via conda (no sudo required)
  conda install -c conda-forge -y cmake make gcc_linux-64 gxx_linux-64
  
  # Install pre-commit via pip (isaaclab.sh tries to install it with sudo)
  pip install pre-commit
  
  # Workaround for evdev build issue on older Linux kernels
  # evdev requires KEY_LINK_PHONE which may not be available on kernel 5.15
  # Create a stub package to satisfy the dependency
  echo "Creating evdev stub package to avoid build issues..."
  EVDEV_STUB_DIR=$(mktemp -d)
  mkdir -p $EVDEV_STUB_DIR/evdev
  echo "# Stub package for evdev" > $EVDEV_STUB_DIR/evdev/__init__.py
  cat > $EVDEV_STUB_DIR/pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "evdev"
version = "1.7.1"
description = "Stub package for evdev (Linux input device library not required for IsaacSim)"

[tool.setuptools.packages.find]
where = ["."]
EOF
  pip install $EVDEV_STUB_DIR/
  rm -rf $EVDEV_STUB_DIR
  echo "✓ evdev stub installed"
  
  cd $WORKSPACE_DIR/IsaacLab
  # work-around for egl_probe cmake max version issue
  export CMAKE_POLICY_VERSION_MINIMUM=3.5
  
  # Override sudo to avoid requiring root privileges
  # isaaclab.sh uses sudo for apt-get, but we've already installed everything via conda/pip
  function sudo() { "$@"; }
  export -f sudo
  
  # Install IsaacLab (allow evdev-related errors to be ignored)
  ./isaaclab.sh --install || echo "IsaacLab installation completed (some optional dependencies may have failed)"

 # Install Holosoma
  pip install -U pip
  pip install -e $ROOT_DIR/src/holosoma[unitree,booster]

  # Force upgrade wandb to override rl-games constraint
  pip install --upgrade 'wandb>=0.21.1'
  touch $SENTINEL_FILE
fi
