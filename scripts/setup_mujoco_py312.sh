#!/bin/bash
# Setup hsmujoco_py312 conda env — Python 3.12 for ROS2 Jazzy compatibility.
# Mirrors setup_mujoco.sh but uses Python 3.12 and installs holosoma with --no-deps
# (because holosoma pins numpy==1.23.5 which has no cp312 wheel).
set -e

# Resolve SCRIPT_DIR through symlinks
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
ROOT_DIR=$(dirname "$SCRIPT_DIR")
PROJECTS_DIR=$(dirname "$ROOT_DIR")

source ${SCRIPT_DIR}/source_common.sh

CONDA_ENV_NAME=hsmujoco_py312
ENV_ROOT=$CONDA_ROOT/envs/$CONDA_ENV_NAME
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_${CONDA_ENV_NAME}

mkdir -p $WORKSPACE_DIR

if [[ -f $SENTINEL_FILE ]]; then
  echo "✅ $CONDA_ENV_NAME already set up. Remove $SENTINEL_FILE to force re-setup."
  exit 0
fi

# --- Create conda env ---
if [[ ! -d $ENV_ROOT ]]; then
  echo "Creating conda env $CONDA_ENV_NAME with Python 3.12..."
  MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n $CONDA_ENV_NAME python=3.12 -c conda-forge --override-channels
fi

source $CONDA_ROOT/bin/activate $CONDA_ENV_NAME

# --- Conda deps ---
conda install -c conda-forge -y libstdcxx-ng ffmpeg

# --- Core Python deps ---
pip install --upgrade pip
pip install "mujoco>=3.0.0" mujoco-python-viewer
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# --- Holosoma runtime deps (compatible with Python 3.12) ---
pip install astor easydict ipdb joblib loguru lxml matplotlib meshcat omegaconf \
    opencv-python plotly pygame pynput rich scipy tensorboard tensordict \
    termcolor tqdm trimesh "yourdfpy>=0.0.58" zmq shapely click \
    "warp-lang>=1.10" pydantic "tyro>=1.0.0" "numpy<2"

# --- Install holosoma + extensions (--no-deps to skip numpy==1.23.5 pin) ---
pip install --no-deps -e $ROOT_DIR/src/holosoma

# Extensions (if present)
EXT_DIR=$PROJECTS_DIR/FAR-HolosomaExtension/src/extensions
for pkg in common x1_humanoid x1_humanoid_inference xdof_inference; do
  PKG_DIR=$EXT_DIR/$pkg
  if [[ -d $PKG_DIR && -f $PKG_DIR/pyproject.toml ]]; then
    echo "Installing extension: $pkg"
    pip install --no-deps -e $PKG_DIR
  fi
done

# Also install holosoma_inference if present
if [[ -f $ROOT_DIR/src/holosoma_inference/pyproject.toml ]]; then
  pip install --no-deps -e $ROOT_DIR/src/holosoma_inference
fi

touch $SENTINEL_FILE
echo ""
echo "✅ $CONDA_ENV_NAME setup complete."
echo ""
echo "To activate:"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  source $CONDA_ROOT/bin/activate $CONDA_ENV_NAME"
echo "  export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${ENV_ROOT}/lib"
