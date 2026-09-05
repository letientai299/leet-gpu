# Setup

## Prerequisites

[mise](https://mise.jdx.dev) for tools and tasks.

### macOS

No local `nvcc`. Docker builds the `linux/amd64` image (`mise run dc:up`). Set
`SSH_HOST` to a GPU machine to run binaries. Homebrew `rsync` 3.2+ is preferred
over macOS `openrsync` for `deploy`.

### Linux GPU host

NVIDIA driver, CUDA toolkit (`nvcc`), CMake 3.22+, Ninja, and ccache. With
`BUILD_ENV=auto` and empty `SSH_HOST`, build and run locally. Default CUDA
architectures are 86, 90, and 100. Binaries from Docker use CUDA 13.3 and a
static runtime; the host driver must load them.

## Environment

```sh
cp .env.example .env
```

`BUILD_ENV` accepts `auto`, `local`, or `docker`. `auto` builds locally when
`nvcc` exists and uses Docker otherwise. CMake writes to `build/local` or
`build/docker`; binaries still land in `bin/`. Fetched dependency sources live
in `.cache/fetchcontent/`; compiler caches live under the matching build
directory. Deleting `build/` drops compiler caches but keeps downloaded sources.
`SSH_HOST` independently selects the run host; leave it empty for local
execution.

## Workflow

Build every app or one app:

```sh
mise run build
mise run build -- src/pmpp/ch03/grayscale
```

Each source path under `src/` defines its app key and binary path. For example,
`src/pmpp/ch03/grayscale.cu` builds `bin/pmpp/ch03/grayscale`.
The `src/` prefix is optional. Use it in `.env` for editor path completion.

Deploy one app and forward its arguments:

```sh
mise run deploy -- src/pmpp/ch02/hello
mise run deploy -- src/pmpp/ch03/grayscale -i /data/input.png -o /data/output.png
```

Remote argument paths refer to files on `SSH_HOST`. `APP` supplies the default
app for `deploy` and `dev`. `ARGS` supplies trusted shell-word arguments when no
arguments are passed explicitly.

Watch the selected app and shared code, then rebuild and deploy after changes:

```sh
mise run dev -- src/pmpp/ch03/grayscale -i /data/input.png -o /data/output.png
mise run dev
```

`dev` also watches `.env`, so editing `APP` or `ARGS` retargets the running loop.

Every binary supports `--help` without initializing CUDA.

Fix or check C++ and CUDA sources:

```sh
mise run fmt
mise run lint
prek --config .config/prek.toml run --all-files
```

[`clang-tidy`][clang-tidy] runs inside the CUDA container against
`build/docker`. [`prek`][prek] runs `mise run lint` before commits and treats
every diagnostic as an error. `mise install` installs its Git hook after the
tools.

## macOS: CLion Intellisense

CLion runs on the Mac without `nvcc`. Point it at the Docker toolchain so it
indexes against CUDA headers inside the image:

1. Run `mise run dc:up`.
2. Add a Docker toolchain using the configured `CONTAINER_NAME` image.
3. Add a CMake profile using that toolchain.

Builds and deployments still use the mise tasks. See the [CLion Docker toolchain
documentation][clion-docker].

### Format on save (match nvim)

nvim formats `.cu` / `.cuh` / `.cpp` / headers on save with `clang-format` and
this repo's [`.clang-format`](../.clang-format). CLion is configured the same way
via `.idea` (untracked, so redo this on a new machine):

- `.idea/codeStyles/Project.xml`, `.idea/editor.xml` — ClangFormat is the
  formatting engine, so `.clang-format` wins over CLion's own code style.
- `.idea/workspace.xml` — `FormatOnSaveOptions` reformats on save, scoped to the
  `C/C++` and `C/C++ Header` file types. CMake and Markdown stay untouched.

CLion formats with the clang-format built into its bundled clangd, not the
`clang-format` pinned in `.config/mise/config.toml`. For an exact match, set
**Settings | Editor | Code Style | C/C++ | General** to use an external
clang-format at the path from `mise which clang-format`.

CMake is not format-on-save in either editor. `gersemi` reads
`.config/gersemi/config.yaml` and runs via `mise run fmt` and the prek hook.

[clion-docker]: https://www.jetbrains.com/help/clion/clion-toolchains-in-docker.html
[clang-tidy]: https://clang.llvm.org/extra/clang-tidy/
[prek]: https://prek.j178.dev/
