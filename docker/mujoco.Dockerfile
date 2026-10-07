# syntax=docker/dockerfile:1.7
# MuJoCo-only image. The dependencies target is intentionally source-free so
# downstream images can reuse it across Holosoma Python-only releases.
# GPU/Warp acceleration is off by default. To build the GPU-accelerated image:
#   docker build --build-arg WARP=true -f docker/mujoco.Dockerfile ...
# No GPU is needed at build time; an NVIDIA GPU + driver >= 555.58.02 is required at run time.
FROM ubuntu:24.04 AS dependencies

USER root

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV HOLOSOMA_DEPS_DIR=/root/.holosoma_deps
ENV CONDA_ROOT=/root/.holosoma_deps/miniconda3
ENV PATH=$CONDA_ROOT/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    swig \
    curl \
    wget \
    unzip \
    git \
    sudo \
    ca-certificates \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /miniconda.sh && \
    bash /miniconda.sh -b -u -p $CONDA_ROOT && \
    rm /miniconda.sh

RUN echo ". $CONDA_ROOT/etc/profile.d/conda.sh" >> ~/.bashrc && \
    conda config --set always_yes true

# Install editable package metadata at the final source path. The runtime
# stage supplies the implementation without rerunning package installation.
WORKDIR /workspace/holosoma
COPY scripts/setup_mujoco.sh scripts/source_common.sh scripts/versions.sh ./scripts/
COPY src/holosoma/pyproject.toml src/holosoma/README.md ./src/holosoma/
RUN mkdir -p src/holosoma/holosoma && \
    touch src/holosoma/holosoma/__init__.py

ARG WARP=false
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    . $CONDA_ROOT/etc/profile.d/conda.sh && \
    chmod +x scripts/setup_mujoco.sh && \
    if [ "$WARP" = "true" ]; then \
        scripts/setup_mujoco.sh --skip-driver-check; \
    else \
        scripts/setup_mujoco.sh --no-warp; \
    fi && \
    conda clean -a -y

COPY docker/apt/zzzz-holosoma-security-updates /etc/apt/apt.conf.d/
RUN apt-get update \
    && apt-get install -y --no-install-recommends unattended-upgrades \
    && unattended-upgrade --verbose \
    && apt-get purge -y unattended-upgrades \
    && rm -rf /var/lib/apt/lists/*

LABEL org.holosoma.layer="dependencies"

# The final image differs from dependencies only by the cheap source layer.
FROM dependencies AS runtime

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV MUJOCO_GL=egl

ARG HOLOSOMA_COMMIT=unknown
ARG HOLOSOMA_SOURCE_TREE=unknown
LABEL org.holosoma.commit="${HOLOSOMA_COMMIT}" \
      org.holosoma.source-tree="${HOLOSOMA_SOURCE_TREE}" \
      org.holosoma.layer="runtime"

ENTRYPOINT []
CMD ["/bin/bash"]

COPY --link . /workspace/holosoma
