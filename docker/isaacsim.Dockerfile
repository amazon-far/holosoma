# syntax=docker/dockerfile:1.7
# IsaacSim 5.1.0 / IsaacLab 2.3.2 image. The dependencies target is intentionally source-free so
# downstream images can reuse it across Holosoma Python-only releases.
FROM nvcr.io/nvidia/isaac-sim:5.1.0 AS dependencies

USER root

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV HOLOSOMA_DEPS_DIR=/root/.holosoma_deps
ENV CONDA_ROOT=/root/.holosoma_deps/miniconda3
ENV PATH=$CONDA_ROOT/bin:$PATH
ENV OMNI_KIT_ACCEPT_EULA=1
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

# Install editable package metadata at the final source path. The runtime
# stage supplies the implementation without rerunning package installation.
WORKDIR /workspace/holosoma
COPY scripts/setup_isaacsim_delta.sh scripts/source_common.sh scripts/versions.sh \
    scripts/isaacsim_constraints.txt ./scripts/
COPY src/holosoma/pyproject.toml src/holosoma/README.md ./src/holosoma/
RUN mkdir -p src/holosoma/holosoma && \
    touch src/holosoma/holosoma/__init__.py

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    . $CONDA_ROOT/etc/profile.d/conda.sh && \
    chmod +x scripts/setup_isaacsim_delta.sh && \
    scripts/setup_isaacsim_delta.sh && \
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

ARG HOLOSOMA_COMMIT=unknown
ARG HOLOSOMA_SOURCE_TREE=unknown
LABEL org.holosoma.commit="${HOLOSOMA_COMMIT}" \
      org.holosoma.source-tree="${HOLOSOMA_SOURCE_TREE}" \
      org.holosoma.layer="runtime"

ENTRYPOINT []
CMD ["/bin/bash"]

COPY --link . /workspace/holosoma
