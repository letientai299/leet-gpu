#!/usr/bin/env bash
# Reach the GPU target named by SSH_HOST and run gpu.sh there.

target_lib_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

is_local_target() {
  [ "$1" = local ] || [ "$1" = localhost ]
}

# One multiplexed SSH connection serves rsync, runs, and clock changes, so the
# handshake is paid once per hour instead of on every deploy.
ssh_opts=(-o ControlMaster=auto -o ControlPath="/tmp/ssh-${CONTAINER_NAME:-leet-gpu}-%r@%h" -o ControlPersist=1h)

# Prints a shell command that runs gpu.sh with ARGS on a remote host. The
# script travels as base64, which every remote shell passes through unchanged.
remote_gpu_command() {
  local script args
  script=$(base64 <"$target_lib_dir/gpu.sh" | tr -d '\n')
  printf -v args ' %q' "$@"
  # shellcheck disable=SC2016 # The remote shell expands $(...), not this one.
  printf 'LG_GPU_MAIN=1 bash -c "$(printf %%s %s | base64 -d)" gpu.sh%s\n' "$script" "$args"
}

# Runs gpu.sh ARGS on the target. Remote runs get a TTY so a dropped connection
# hangs up the remote process instead of leaving it running.
run_gpu() {
  local target=$1
  shift
  if is_local_target "$target"; then
    LG_GPU_MAIN=1 bash "$target_lib_dir/gpu.sh" "$@"
    return
  fi
  # shellcheck disable=SC2029 # The command is escaped by remote_gpu_command.
  ssh -tt -o LogLevel=ERROR "${ssh_opts[@]}" "$target" "$(remote_gpu_command "$@")"
}

# Names one lock request: target and GPU index.
clock_key() {
  printf '%s %s\n' "$1" "${GPU_INDEX:-0}"
}

clock_unlock() {
  local target gpu
  read -r target gpu <<<"$1"
  run_gpu "$target" unlock "$gpu"
}
