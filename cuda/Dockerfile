FROM nvidia/cuda:13.3.1-devel-ubuntu26.04

# Install mise from its apt repo instead of the mise.run script: the script
# extracts a tarball with GNU tar, which fails under QEMU emulation
# (`mkdir: Function not implemented`) when building linux/amd64 on arm64.
# dpkg-based install avoids that syscall path.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdb \
    ccache \
    ninja-build \
    cmake \
    clangd \
    curl \
    ca-certificates \
    gpg \
    && install -dm 755 /etc/apt/keyrings \
    && curl -fsSL https://mise.jdx.dev/gpg-key.pub \
       | gpg --dearmor -o /etc/apt/keyrings/mise-archive-keyring.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/mise-archive-keyring.gpg arch=amd64] https://mise.jdx.dev/deb stable main" \
       > /etc/apt/sources.list.d/mise.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends mise \
    && rm -rf /var/lib/apt/lists/*

ENV MISE_DATA_DIR=/usr/local/share/mise \
    MISE_CACHE_DIR=/usr/local/share/mise/cache \
    PATH=/usr/local/share/mise/shims:/usr/local/bin:$PATH

RUN mise activate bash >> /etc/bash.bashrc
