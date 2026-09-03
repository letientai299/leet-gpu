# cuda

## About

CUDA subproject of the [leetgpu][leetgpu] learning monorepo.

I use a Mac, which can't run CUDA code. Thankfully, we can build in a
`linux/amd64` container and run the binary on an SSH GPU host to run.

## Setup

```sh
cp .env.example .env   # then edit CONTAINER_NAME / SSH_HOST / CUDA_KERNEL
mise run dc:up         # build the image, start the container
```

`SSH_HOST` must be reachable via `ssh <SSH_HOST>` (define it in
`~/.ssh/config`). `CUDA_KERNEL` selects the kernel run by `mise run deploy` and
`mise run dev`.

## Workflow

Run `mise tasks` for the live list. The loop:

- `mise run dev` — watch sources, build configuration, and `.env`; rebuild in
  the container and deploy the selected kernel when its binary or selection
  changes.
- `mise run build` — one-off build into `bin/` (forwards into the container).
- `mise run deploy` — rsync `bin/main` to `$SSH_HOST` and run `$CUDA_KERNEL`.
- `mise run deploy -- <kernel>` — override `CUDA_KERNEL` for one run.
- `mise run deploy -- --help` — show kernel IDs on the remote host.

Run `bin/main --help` on a CUDA host to show available kernel IDs. Architecture
specific kernels live under their matching `src/sm*/` directory. The checked-in
`.clangd` selects that directory's analysis target automatically.

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
