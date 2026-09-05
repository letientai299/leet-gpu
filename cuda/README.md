# cuda

## About

Native CUDA apps for the [leetgpu][leetgpu] learning monorepo.

Each source path under `apps/` defines its app key and binary path. For example,
`apps/pmpp/ch03/grayscale.cu` builds `bin/pmpp/ch03/grayscale`.

## Setup

Copy the environment template when using Docker, SSH, or default app arguments:

```sh
cp .env.example .env
```

`BUILD_ENV` accepts `auto`, `local`, or `docker`. `auto` builds locally when
`nvcc` exists and uses Docker otherwise. `SSH_HOST` independently selects the
run host; leave it empty for local execution.

## Workflow

Build every app or one app:

```sh
mise run build
mise run build -- pmpp/ch03/grayscale
```

Deploy one app and forward its arguments:

```sh
mise run deploy -- pmpp/ch02/hello
mise run deploy -- pmpp/ch03/grayscale -i /data/input.png -o /data/output.png
```

Remote argument paths refer to files on `SSH_HOST`. `APP` supplies the default
app for `deploy` and `dev`. `ARGS` supplies trusted shell-word arguments when no
arguments are passed explicitly.

Watch the selected app and shared code, then rebuild and deploy after changes:

```sh
mise run dev -- pmpp/ch03/grayscale -i /data/input.png -o /data/output.png
mise run dev
```

Every binary supports `--help` without initializing CUDA.

## Apps

- `pmpp/ch02/hello`: basic CUDA launch
- `pmpp/ch03/grayscale`: RGB PNG to grayscale PNG

## macOS: CLion intellisense

CLion runs on the Mac without `nvcc`. Point it at the Docker toolchain so it
indexes against CUDA headers inside the image:

1. Run `mise run dc:up`.
2. Add a Docker toolchain using the configured `CONTAINER_NAME` image.
3. Add a CMake profile using that toolchain.

Builds and deployments still use the mise tasks. See the [CLion Docker
toolchain documentation][clion-docker].

[clion-docker]: https://www.jetbrains.com/help/clion/clion-toolchains-in-docker.html
[leetgpu]: https://leetgpu.com/resources
