# MuJoCo-only image
# Runs setup_mujoco.sh to create the hsmujoco conda environment (Python 3.10)
# with MuJoCo >= 3.0.0 and holosoma installed.
#
# GPU/Warp acceleration is off by default. To build the GPU-accelerated image:
#   docker build --build-arg WARP=true -f docker/mujoco.Dockerfile ...
# No GPU is needed at build time; the WARP build runs setup_mujoco.sh with
# --skip-driver-check. An NVIDIA GPU + driver >= 555.58.02 is required at run time.
FROM nvcr.io/nvidia/isaac-sim:5.1.0

USER root

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE_DIR=/workspace
ENV CONDA_ROOT=/root/.holosoma_deps/miniconda3
ENV PATH=$CONDA_ROOT/bin:$PATH

RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    swig \
    curl \
    wget \
    unzip \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /miniconda.sh && \
    bash /miniconda.sh -b -u -p $CONDA_ROOT && \
    rm /miniconda.sh

# Configure conda for non-interactive use
RUN echo ". $CONDA_ROOT/etc/profile.d/conda.sh" >> ~/.bashrc && \
    conda config --set always_yes true

RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

COPY . ./holosoma

ARG WARP=false
# --skip-driver-check: no GPU is visible during `docker build`; the GPU is
# required when the image runs, not at install time.
RUN . $CONDA_ROOT/etc/profile.d/conda.sh && \
    cd /workspace/holosoma/scripts && \
    chmod +x setup_mujoco.sh && \
    if [ "$WARP" = "true" ]; then \
        ./setup_mujoco.sh --skip-driver-check; \
    else \
        ./setup_mujoco.sh --no-warp; \
    fi


WORKDIR /workspace/holosoma

ENTRYPOINT []
CMD ["/bin/bash"]
