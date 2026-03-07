# IsaacGym-only image
# Runs setup_isaacgym.sh to create the hsgym conda environment (Python 3.8)
# with IsaacGym Preview 4 and holosoma installed.
#
# NOTE: setup_isaacgym.sh downloads IsaacGym from
#   https://developer.nvidia.com/isaac-gym-preview-4
# which requires an NVIDIA developer account. If the download fails during
# build, pre-download the package and place it at:
#   /workspace/IsaacGym_Preview_4_Package.tar.gz
# before running the setup script (it will skip the download if the file
# already exists).
FROM nvcr.io/nvidia/isaac-sim:5.1.0

USER root

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE_DIR=/workspace
ENV CONDA_ROOT=/root/.holosoma_deps/miniconda3
ENV PATH=$CONDA_ROOT/bin:$PATH

RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update && apt-get install -y --no-install-recommends \
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

RUN . $CONDA_ROOT/etc/profile.d/conda.sh && \
    cd /workspace/holosoma/scripts && \
    chmod +x setup_isaacgym.sh && \
    ./setup_isaacgym.sh


WORKDIR /workspace/holosoma

ENTRYPOINT []
CMD ["/bin/bash"]
