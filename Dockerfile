FROM nvidia/cuda:13.3.1-devel-rockylinux8

ARG CLANGD_VERSION=22.1.0
ARG CLANGD_SHA256=c54e57dbff3ccc9e8352367ddb7030ad3f624073ec58c7477424e7919f578572

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
    libstdc++-static \
    unzip \
    curl \
    ca-certificates \
    && yum-config-manager --add-repo https://mise.jdx.dev/rpm/mise.repo \
    && yum install -y mise \
    && yum clean all \
    && rm -rf /var/cache/yum

RUN curl -fsSLo /tmp/clangd.zip \
      "https://github.com/clangd/clangd/releases/download/${CLANGD_VERSION}/clangd-linux-${CLANGD_VERSION}.zip" \
    && echo "${CLANGD_SHA256}  /tmp/clangd.zip" | sha256sum -c - \
    && unzip -q /tmp/clangd.zip -d /opt \
    && rm /tmp/clangd.zip

ENV MISE_DATA_DIR=/usr/local/share/mise \
    MISE_CACHE_DIR=/usr/local/share/mise/cache \
    PATH=/opt/clangd_${CLANGD_VERSION}/bin:/usr/local/share/mise/shims:/usr/local/bin:$PATH

RUN mise activate bash >> /etc/bash.bashrc
