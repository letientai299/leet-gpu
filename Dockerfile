FROM nvidia/cuda:13.3.1-devel-rockylinux8

# The mise installer fails under QEMU emulation.
RUN yum install -y \
    yum-utils \
    epel-release \
    && yum-config-manager --set-enabled powertools \
    && yum install -y \
    gdb \
    ccache \
    ninja-build \
    cmake \
    clang-tools-extra \
    libstdc++-static \
    curl \
    ca-certificates \
    && yum-config-manager --add-repo https://mise.jdx.dev/rpm/mise.repo \
    && yum install -y mise \
    && yum clean all \
    && rm -rf /var/cache/yum

ENV MISE_DATA_DIR=/usr/local/share/mise \
    MISE_CACHE_DIR=/usr/local/share/mise/cache \
    PATH=/usr/local/share/mise/shims:/usr/local/bin:$PATH

RUN mise activate bash >> /etc/bash.bashrc
