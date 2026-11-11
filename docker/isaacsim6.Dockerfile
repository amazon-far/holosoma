# syntax=docker/dockerfile:1.7
# Isaac Sim 6.2 / Isaac Lab 3.0, alongside the existing 5.1 / 2.3 image.
FROM ghcr.io/astral-sh/uv:0.12.9@sha256:8b940d3a9d65bed080436972241af2e21c84b5e8c9193f7014ed71479ee795ff AS uv
FROM nvcr.io/nvidia/isaac-sim:6.2.0 AS dependencies

USER root

SHELL ["/bin/bash", "-c"]

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV HOLOSOMA_DEPS_DIR=/root/.holosoma_deps
ENV ISAACSIM_ROOT_PATH=/isaac-sim
ENV ISAACSIM6_ENV_ROOT=/root/.holosoma_deps/envs/hssim6
ENV PATH=$ISAACSIM6_ENV_ROOT/bin:$PATH
ENV OMNI_KIT_ACCEPT_EULA=1
ENV MUJOCO_GL=disable
ENV HUB__ARGS__DETECT_ONLY=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential swig curl wget unzip git sudo ffmpeg libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /workspace/holosoma
COPY scripts/setup_isaacsim6.sh scripts/source_isaacsim6_setup.sh \
    scripts/source_common.sh scripts/versions.sh scripts/isaacsim6_constraints.txt ./scripts/
COPY src/holosoma/pyproject.toml src/holosoma/README.md ./src/holosoma/
RUN mkdir -p src/holosoma/holosoma && touch src/holosoma/holosoma/__init__.py

# Install Lab's Carbonite environment shim before Kit Python imports can start
# worker threads. /etc/ld.so.preload survives python.sh overwriting LD_PRELOAD.
# Keep this system-wide installation out of the standalone setup script.
RUN --mount=type=cache,target=/root/.cache/uv \
    source scripts/setup_isaacsim6.sh \
    && isaacsim6_validate_runtime \
    && mkdir -p "$WORKSPACE_DIR" \
    && isaacsim6_prepare_isaaclab \
    && bash "$ISAACSIM6_ISAACLAB_DIR/docker/scripts/install_carb_env_shim.sh" "$ISAACSIM_ROOT_PATH" \
    && isaacsim6_setup_main

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
