# syntax=docker/dockerfile:1.7
# Combined development image with every Holosoma environment. Dependencies are
# installed without implementation source so Python-only changes reuse this layer.
FROM nvcr.io/nvidia/isaac-sim:5.1.0 AS dependencies

USER root

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV HOLOSOMA_DEPS_DIR=/root/.holosoma_deps
ENV CONDA_ROOT=/root/.holosoma_deps/miniconda3
ENV PATH=$CONDA_ROOT/bin:$PATH
ENV ISAACSIM_ROOT_PATH=/isaac-sim

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    swig \
    curl \
    wget \
    unzip \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /miniconda.sh && \
    bash /miniconda.sh -b -u -p $CONDA_ROOT && \
    rm /miniconda.sh

RUN echo ". $CONDA_ROOT/etc/profile.d/conda.sh" >> ~/.bashrc && \
    conda config --set always_yes true

# Install editable package metadata at the final source paths. The runtime
# stage supplies the implementation without rerunning package installation.
WORKDIR /workspace/holosoma
COPY scripts/setup_all.sh scripts/setup_inference.sh scripts/setup_isaacgym.sh \
    scripts/setup_isaacsim.sh scripts/setup_isaacsim_delta.sh scripts/setup_mujoco.sh \
    scripts/setup_retargeting.sh scripts/source_common.sh scripts/versions.sh \
    scripts/isaacgym_constraints.txt scripts/isaacsim_constraints.txt ./scripts/
COPY src/holosoma/pyproject.toml src/holosoma/README.md ./src/holosoma/
COPY src/holosoma_inference/pyproject.toml src/holosoma_inference/README.md \
    ./src/holosoma_inference/
COPY src/holosoma_retargeting/pyproject.toml src/holosoma_retargeting/README.md \
    ./src/holosoma_retargeting/
RUN mkdir -p \
      src/holosoma/holosoma \
      src/holosoma_inference/holosoma_inference \
      src/holosoma_retargeting/holosoma_retargeting && \
    touch \
      src/holosoma/holosoma/__init__.py \
      src/holosoma_inference/holosoma_inference/__init__.py \
      src/holosoma_retargeting/holosoma_retargeting/__init__.py

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/var/cache/holosoma \
    PIP_NO_CACHE_DIR=1 \
    ISAACGYM_ARCHIVE_PATH=/var/cache/holosoma/IsaacGym_Preview_4_Package.tar.gz \
      bash scripts/setup_all.sh && \
    conda clean -a -y

COPY docker/apt/zzzz-holosoma-security-updates /etc/apt/apt.conf.d/
RUN apt-get update \
    && apt-get install -y --no-install-recommends unattended-upgrades \
    && unattended-upgrade --verbose \
    && apt-get purge -y unattended-upgrades \
    && rm -rf /var/lib/apt/lists/*

LABEL org.holosoma.layer="dependencies"

FROM dependencies AS runtime

ARG HOLOSOMA_COMMIT=unknown
ARG HOLOSOMA_SOURCE_TREE=unknown
LABEL org.holosoma.commit="${HOLOSOMA_COMMIT}" \
      org.holosoma.source-tree="${HOLOSOMA_SOURCE_TREE}" \
      org.holosoma.layer="runtime"

ENTRYPOINT []
CMD ["/bin/bash"]

COPY --link . /workspace/holosoma
