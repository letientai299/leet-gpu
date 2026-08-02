# cuda

CUDA subproject of the [leetgpu][leetgpu] learning monorepo. The Mac host has no
CUDA toolchain, so builds run in a `linux/amd64` container (emulated under QEMU)
and the resulting binary is shipped to an SSH GPU host to run.

## Setup

```sh
cp .env.example .env   # then edit CONTAINER_NAME / SSH_HOST
mise run dc:up         # build the image, start the container
```

`SSH_HOST` must be reachable via `ssh <SSH_HOST>` (define it in
`~/.ssh/config`).

## Workflow

Run `mise tasks` for the live list. The loop:

- `mise run dev` — watch `src` + `CMakeLists.txt`, rebuild in the container, and
  deploy+run on the GPU host only when the binary actually changed.
- `mise run build` — one-off build into `bin/` (forwards into the container).
- `mise run deploy` — rsync `bin/main` to `$SSH_HOST` and run it.

## CLion intellisense (Docker toolchain)

CLion runs on the Mac, which has no `nvcc`, so a local CMake profile cannot
enable the CUDA language. Point CLion at the container instead — it drives CMake
inside the image, so indexing, completion, and `<<<>>>` syntax resolve against
the real CUDA headers. No GPU is required for indexing.

1. Build the image once: `mise run dc:up`.
2. **Settings → Build, Execution, Deployment → Toolchains → + → Docker**, and
   select the `leet-gpu-cuda` image (or whatever `CONTAINER_NAME` you set).
   CLion auto-detects `cmake`, `ninja`, `gcc/g++`, and `gdb`.
3. **Settings → … → CMake**: add a profile using that Docker toolchain.

Building/running the GPU binary still goes through the `mise` tasks above; the
Docker toolchain is for editor intellisense and debugging, not deploy. See the
[CLion Docker toolchain docs][clion-docker]; JetBrains also ships a reference
[`Dockerfile.remote-cuda-env`][clion-remote-repo].

[leetgpu]: https://leetgpu.com/resources
[clion-docker]:
  https://www.jetbrains.com/help/clion/clion-toolchains-in-docker.html
[clion-remote-repo]: https://github.com/JetBrains/clion-remote
