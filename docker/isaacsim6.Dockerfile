# syntax=docker/dockerfile:1.7
# Isaac Sim 6.2.0 / Isaac Lab 3.0 image. Mirrors isaacsim.Dockerfile's two-stage shape -- a
# source-free `dependencies` target that downstream images reuse across Holosoma Python-only
# releases, plus a cheap `runtime` source layer -- so both lanes are built and cached the same way.
#
# Three deliberate differences from the 5.1 image, each forced by Isaac Lab 3.0:
#
#   * uv, not conda. Isaac Lab 3.0 removed `isaaclab.sh --conda` and actively *rejects* conda: a
#     downloaded Isaac Sim package combined with a non-empty CONDA_PREFIX is a hard error, and its
#     venv check requires CONDA_PREFIX to be empty. So there is no miniconda install here.
#   * A separate HOLOSOMA_DEPS_DIR. The two generations cannot share an IsaacLab checkout
#     (incompatible refs) or a setup sentinel, and `scripts/setup_isaacsim6.sh` will `rm -rf` the
#     checkout under whatever this points at -- aiming it at the 5.1 workspace would delete the
#     other lane's IsaacLab. It is set as an ENV (not left to the script's default) so the value
#     is visible to anything that runs in the container, not just to setup.
#   * A git clone during build. Isaac Lab 3.0 GA ships a workspace lockfile, so the lane
#     resolves dependencies from the pinned checkout's own pyproject.toml + uv.lock. That means
#     the build needs network access to github.com, which the 5.1 build does not.
FROM nvcr.io/nvidia/isaac-sim:6.2.0 AS dependencies

USER root

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
# Must differ from the 5.1 lane's /root/.holosoma_deps -- see the header.
ENV HOLOSOMA_DEPS_DIR=/root/.holosoma_deps_sim6
ENV OMNI_KIT_ACCEPT_EULA=1
ENV ISAACSIM_ROOT_PATH=/isaac-sim
# uv installs here and the venv lives under HOLOSOMA_DEPS_DIR; both go on PATH so `python` and
# `uv` resolve without sourcing anything, matching how the conda lane puts its bin dir on PATH.
ENV UV_INSTALL_DIR=/usr/local/bin
ENV PATH=$HOLOSOMA_DEPS_DIR/hssim6/bin:$UV_INSTALL_DIR:$PATH

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

# Pinned rather than `latest`: uv is the dependency resolver for this whole lane, so an
# unpinned installer would make the image's contents depend on its build date.
ARG UV_VERSION=0.12.9
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && uv --version

# Install editable package metadata at the final source path. The runtime stage supplies the
# implementation without rerunning package installation.
WORKDIR /workspace/holosoma
COPY scripts/setup_isaacsim6.sh scripts/source_common.sh scripts/versions.sh ./scripts/
COPY src/holosoma/pyproject.toml src/holosoma/README.md ./src/holosoma/
RUN mkdir -p src/holosoma/holosoma && \
    touch src/holosoma/holosoma/__init__.py

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    chmod +x scripts/setup_isaacsim6.sh && \
    scripts/setup_isaacsim6.sh

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
