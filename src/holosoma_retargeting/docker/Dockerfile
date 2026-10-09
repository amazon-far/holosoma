# syntax=docker/dockerfile:1.7
# Retargeting-only image with Python 3.11. Dependency installation uses package
# metadata at the final source path so Python-only changes add one linked layer.
FROM nvcr.io/nvidia/isaac-sim:5.1.0 AS dependencies

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
    && rm -rf /var/lib/apt/lists/*

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /miniconda.sh && \
    bash /miniconda.sh -b -u -p $CONDA_ROOT && \
    rm /miniconda.sh

RUN echo ". $CONDA_ROOT/etc/profile.d/conda.sh" >> ~/.bashrc && \
    conda config --set always_yes true

WORKDIR /workspace/holosoma
COPY scripts/setup_retargeting.sh scripts/source_common.sh ./scripts/
COPY src/holosoma_retargeting/pyproject.toml src/holosoma_retargeting/README.md \
    ./src/holosoma_retargeting/
RUN mkdir -p src/holosoma_retargeting/holosoma_retargeting && \
    touch src/holosoma_retargeting/holosoma_retargeting/__init__.py

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    . $CONDA_ROOT/etc/profile.d/conda.sh && \
    chmod +x scripts/setup_retargeting.sh && \
    scripts/setup_retargeting.sh && \
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
