# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/hsgym
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_isaacgym

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
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n hsgym python=3.8 -c conda-forge --override-channels
  fi

  source $CONDA_ROOT/bin/activate hsgym

  # Install libstdcxx-ng to fix the error: `version `GLIBCXX_3.4.32' not found` on Ubuntu 24.04
  conda install -c conda-forge -y libstdcxx-ng

  # Install ffmpeg for video encoding
  conda install -c conda-forge -y ffmpeg
  conda install -c conda-forge -y libiconv

  # Install Isaac Gym
  if [[ ! -d $WORKSPACE_DIR/isaacgym ]]; then
    wget https://developer.nvidia.com/isaac-gym-preview-4 -O $WORKSPACE_DIR/IsaacGym_Preview_4_Package.tar.gz
    tar -xzf $WORKSPACE_DIR/IsaacGym_Preview_4_Package.tar.gz -C $WORKSPACE_DIR
  fi
  cd $WORKSPACE_DIR/isaacgym/python
  $ENV_ROOT/bin/pip install -e .

  # Install Holosoma
  pip install -U pip
  pip install -e $ROOT_DIR/src/holosoma[unitree,booster]
  touch $SENTINEL_FILE
fi
