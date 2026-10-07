# syntax=docker/dockerfile:1.7
# Isaac Gym Preview 4 only ships native bindings through Python 3.8. The dependencies target is
# source-free so downstream images can reuse it across Holosoma Python-only releases. Its cu121
# PyTorch stack only needs the matching CUDA runtime, not the full Isaac Sim distribution.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS dependencies

USER root

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV HOLOSOMA_DEPS_DIR=/root/.holosoma_deps
ENV CONDA_ROOT=/root/.holosoma_deps/miniconda3
ENV PATH=$CONDA_ROOT/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    sudo \
    swig \
    unzip \
    wget \
    ca-certificates \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libvulkan1 \
    libx11-6 \
    libxcursor1 \
    libxext6 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxrender1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /miniconda.sh && \
    bash /miniconda.sh -b -u -p $CONDA_ROOT && \
    rm /miniconda.sh

RUN echo ". $CONDA_ROOT/etc/profile.d/conda.sh" >> ~/.bashrc && \
    conda config --set always_yes true

# Install editable package metadata at the final source path. The runtime
# stage supplies the implementation without rerunning package installation.
WORKDIR /workspace/holosoma
COPY scripts/setup_isaacgym.sh scripts/source_common.sh scripts/isaacgym_constraints.txt ./scripts/
COPY src/holosoma/pyproject.toml src/holosoma/README.md ./src/holosoma/
RUN mkdir -p src/holosoma/holosoma && \
    touch src/holosoma/holosoma/__init__.py

# Isaac Gym requires its bundled NVIDIA ICD manifests and must not discover Mesa EGL.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/holosoma \
    . $CONDA_ROOT/etc/profile.d/conda.sh && \
    chmod +x scripts/setup_isaacgym.sh && \
    ISAACGYM_ARCHIVE_PATH=/var/cache/holosoma/IsaacGym_Preview_4_Package.tar.gz \
      scripts/setup_isaacgym.sh && \
    install -Dm644 "$HOLOSOMA_DEPS_DIR/isaacgym/docker/nvidia_icd.json" \
      /usr/share/vulkan/icd.d/nvidia_icd.json && \
    install -Dm644 "$HOLOSOMA_DEPS_DIR/isaacgym/docker/10_nvidia.json" \
      /usr/share/glvnd/egl_vendor.d/10_nvidia.json && \
    conda clean -a -y

COPY docker/apt/zzzz-holosoma-security-updates /etc/apt/apt.conf.d/
RUN apt-get update \
    && apt-get install -y --no-install-recommends unattended-upgrades \
    && unattended-upgrade --verbose \
    && apt-get purge -y unattended-upgrades \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 \
      /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 \
      /usr/share/glvnd/egl_vendor.d/50_mesa.json

LABEL org.holosoma.layer="dependencies"

# The final image differs from dependencies only by the cheap source layer.
FROM dependencies AS runtime

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

ARG HOLOSOMA_COMMIT=unknown
ARG HOLOSOMA_SOURCE_TREE=unknown
LABEL org.holosoma.commit="${HOLOSOMA_COMMIT}" \
      org.holosoma.source-tree="${HOLOSOMA_SOURCE_TREE}" \
      org.holosoma.layer="runtime"

ENTRYPOINT []
CMD ["/bin/bash"]

COPY --link . /workspace/holosoma
