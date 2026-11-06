# syntax=docker/dockerfile:1.7
# Lab 3 owns the matched Sim / Newton / OvPhysX / renderer stack via its uv.lock.
FROM ubuntu:24.04 AS dependencies
USER root
ENV LANG=C.UTF-8 DEBIAN_FRONTEND=noninteractive
ENV HOLOSOMA_DEPS_DIR=/root/.holosoma_deps
ENV ISAACSIM6_ROOT=/root/.holosoma_deps/isaacsim6
ENV OMNI_KIT_ACCEPT_EULA=1
ENV OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake swig git curl ca-certificates libegl1 libgl1 libglib2.0-0 \
    libgomp1 libx11-6 libxext6 libxi6 libxrender1 libxt6 libsm6 libxrandr2 libxinerama1 \
    libxcursor1 libasound2t64 libvulkan1 unzip \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv
WORKDIR /workspace/holosoma
COPY scripts/setup_isaacsim6.sh scripts/versions.sh ./scripts/
COPY src/holosoma/pyproject.toml src/holosoma/README.md ./src/holosoma/
RUN mkdir -p src/holosoma/holosoma && touch src/holosoma/holosoma/__init__.py
RUN --mount=type=cache,target=/root/.cache/uv bash scripts/setup_isaacsim6.sh
LABEL org.holosoma.layer="dependencies"

FROM dependencies AS runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PATH=/root/.holosoma_deps/isaacsim6/hssim6/bin:$PATH
ARG HOLOSOMA_COMMIT=unknown
ARG HOLOSOMA_SOURCE_TREE=unknown
LABEL org.holosoma.commit="${HOLOSOMA_COMMIT}" \
      org.holosoma.source-tree="${HOLOSOMA_SOURCE_TREE}" \
      org.holosoma.layer="runtime"
ENTRYPOINT []
CMD ["/bin/bash"]
COPY --link . /workspace/holosoma
