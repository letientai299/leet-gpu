FROM nvidia/cuda:13.3.1-devel-rockylinux8

ARG CLANGD_VERSION=23.1.0
ARG CLANGD_SHA256=e53b1a96196095faedb7642cf64964f7fb9ad4a0c1f00dd2c172a3d9dcbafdfd
ARG CCACHE_VERSION=4.14.1
ARG CCACHE_SHA256=8f684cf82a7acd2f1daddbefcda505c499afe8245cfbfe78f751ce389cd6df52

# The mise installer fails under QEMU emulation.
RUN yum install -y\
    yum-utils\
    epel-release\
    && yum-config-manager --set-enabled powertools \
    && yum install -y \
    gdb\
    git\
    jq\
    libstdc++-static\
    python3.12-pip\
    unzip\
    curl\
    ca-certificates\
    && yum-config-manager --add-repo https://mise.jdx.dev/rpm/mise.repo \
    && yum install -y mise \
    && yum clean all \
    && rm -rf /var/cache/yum

RUN curl -fsSLo /tmp/clangd.zip\
      "https://github.com/clangd/clangd/releases/download/${CLANGD_VERSION}/clangd-linux-${CLANGD_VERSION}.zip"\
    && echo "${CLANGD_SHA256}  /tmp/clangd.zip" | sha256sum -c - \
    && unzip -q /tmp/clangd.zip -d /opt \
    && rm /tmp/clangd.zip

# EPEL ccache 3.7 rejects `-x cu`, so no CUDA compile hits the cache.
# Existing build trees cache /usr/bin/ccache.
RUN curl -fsSLo /tmp/ccache.tar.gz\
      "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64-musl-static.tar.gz"\
    && echo "${CCACHE_SHA256}  /tmp/ccache.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/ccache.tar.gz -C /usr/local/bin --strip-components=1\
      "ccache-${CCACHE_VERSION}-linux-x86_64-musl-static/ccache"\
    && ln -sf /usr/local/bin/ccache /usr/bin/ccache \
    && rm /tmp/ccache.tar.gz

# Rocky 8 ships CMake 3.26, below the 3.30.4 this project requires, so pip
# provides the only cmake. CLion's Docker toolchain probes /usr/bin instead of
# resolving PATH, and would otherwise report the older one.
# Rocky 8's Ninja 1.8 misses edits saved while their compile runs; 1.12+
# rebuilds them (https://github.com/ninja-build/ninja/pull/1943). Existing
# build trees cache /usr/bin/ninja.
RUN python3.12 -m pip install --no-cache-dir --root-user-action=ignore\
    clang-tidy==22.1.8\
    cmake==4.4.4\
    ninja==1.13.2\
    && ln -sf /usr/local/bin/ninja /usr/bin/ninja \
    && ln -sf /usr/local/bin/cmake /usr/bin/cmake \
    && ln -sf /usr/local/bin/ctest /usr/bin/ctest \
    && ln -sf /usr/local/bin/cpack /usr/bin/cpack

ENV MISE_DATA_DIR=/usr/local/share/mise \
    MISE_CACHE_DIR=/usr/local/share/mise/cache \
    PATH=/opt/clangd_${CLANGD_VERSION}/bin:/usr/local/share/mise/shims:/usr/local/bin:$PATH

RUN mise activate bash >>/etc/bash.bashrc
