#!/usr/bin/env bash
# GPU helpers that run on the GPU host. Source them, or run
# `LG_GPU_MAIN=1 bash gpu.sh run|unlock ...`; SSH callers pass the script text
# through `bash -c`, so the host needs no copy.

gpu_query() {
  nvidia-smi -i "$1" --query-gpu="$2" --format=csv,noheader,nounits \
    | sed 's/^ *//; s/ *$//'
}

# Prints comma-separated fields with no spaces ("Not Active" -> Not_Active).
gpu_fields() {
  gpu_query "$1" "$2" | sed 's/, */,/g; s/ /_/g'
}

# Non-root users go through sudo first, which saves a failing nvidia-smi call.
gpu_admin() {
  local gpu=$1
  shift
  if [ "$(id -u)" != 0 ] && sudo -n nvidia-smi -i "$gpu" "$@" >/dev/null 2>&1; then
    return
  fi
  nvidia-smi -i "$gpu" "$@" >/dev/null 2>&1
}

# Prints the SM clock in MHz for "auto" (highest supported clock at or below
# fraction% of max_clock) or an explicit MHz value.
gpu_pick_clock() {
  local gpu=$1 clock=$2 fraction=$3 max_clock=$4 target
  if [ "$clock" != auto ]; then
    printf '%s\n' "$clock"
    return
  fi
  if [[ ! "$max_clock" =~ ^[1-9][0-9]{0,5}$ ]]; then
    echo "gpu: maximum SM clock unavailable" >&2
    return 1
  fi
  target=$((max_clock * fraction / 100))
  gpu_clock_at_most "$gpu" "$target"
}

# Prints the highest supported SM clock at or below a MHz limit.
gpu_clock_at_most() {
  local clock
  clock=$(nvidia-smi -i "$1" --query-supported-clocks=gr --format=csv,noheader,nounits \
    | awk -v t="$2" '$1 ~ /^[0-9]+$/ && $1 <= t && $1 > best {best=$1} END {print best}')
  if [ -z "$clock" ]; then
    echo "gpu: supported SM clocks unavailable" >&2
    return 1
  fi
  printf '%s\n' "$clock"
}

# A throttled auto lock lowers this per-GPU ceiling; later auto locks start
# below it. It only moves down, so kernels compared in later runs share a clock.
# Delete the file to reset.
gpu_ceiling_file() {
  printf '%s/lg-gpu-%s.clock\n' "${LG_GPU_LOCK_DIR:-/tmp}" "$1"
}

# gpu_cap_clock UUID CLOCK: prints CLOCK lowered to the stored ceiling.
gpu_cap_clock() {
  local ceiling
  ceiling=$(cat "$(gpu_ceiling_file "$1")" 2>/dev/null || true)
  if [[ "$ceiling" =~ ^[1-9][0-9]{0,5}$ ]] && ((ceiling < $2)); then
    printf '%s\n' "$ceiling"
    return
  fi
  printf '%s\n' "$2"
}

# Refuses to benchmark while the GPU slows itself for thermal or hardware
# reasons; the run would measure the fault, not the kernel.
gpu_check_health() {
  local fields name state index
  local -a names states
  fields=$(gpu_fields "$1" clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown)
  names=(hw_slowdown hw_thermal_slowdown sw_thermal_slowdown hw_power_brake_slowdown)
  IFS=, read -r -a states <<<"$fields"
  for index in "${!names[@]}"; do
    name=${names[$index]}
    state=${states[$index]:-}
    if [ "$state" = Active ]; then
      echo "gpu: $(hostname) GPU $1 unhealthy: $name active; not running" >&2
      return 1
    fi
  done
}

clock_watch_pid=
clock_watch_file=

# Samples the SM clock in the background until gpu_stop_watch. The
# power-cap counter is unreliable: on H100 it grows while the clock holds the
# lock, and on A100X it barely moves while the clock sags.
gpu_watch_clock() {
  clock_watch_file=$(mktemp "${TMPDIR:-/tmp}/lg-clock.XXXXXX")
  (
    while :; do
      gpu_query "$1" clocks.sm >>"$clock_watch_file" 2>/dev/null || true
      sleep "${LG_CLOCK_SAMPLE_SECONDS:-0.5}"
    done
  ) &
  clock_watch_pid=$!
}

gpu_stop_watch() {
  if [ -n "$clock_watch_pid" ]; then
    kill "$clock_watch_pid" 2>/dev/null || true
    wait "$clock_watch_pid" 2>/dev/null || true
    clock_watch_pid=
  fi
  if [ -n "$clock_watch_file" ]; then
    rm -f "$clock_watch_file"
    clock_watch_file=
  fi
}

# gpu_review_clock GPU UUID CLOCK MAX_CLOCK
# Steps the ceiling down one supported clock when over 5% of the SM clock
# samples fell below 99% of the lock, the runner's THROTTLED test. Stops at
# half the max clock: below that the GPU is unhealthy, not merely busy.
gpu_review_clock() {
  local gpu=$1 uuid=$2 clock=$3 max_clock=$4 total slow next
  read -r total slow <<<"$(awk -v floor="$((clock * 99))" \
    '$1 ~ /^[0-9]+$/ {n++; if ($1 * 100 < floor) s++} END {print n + 0, s + 0}' \
    "$clock_watch_file")"
  gpu_stop_watch
  ((total >= 10 && slow * 20 > total)) || return 0
  next=$(gpu_clock_at_most "$gpu" "$((clock - 1))") || return 0
  if ((next * 2 < max_clock)); then
    echo "gpu: $(hostname) GPU $gpu unhealthy: throttled below half the max clock" >&2
    return 0
  fi
  (umask 000 && printf '%s\n' "$next" >"$(gpu_ceiling_file "$uuid")")
  echo "gpu: clock under ${clock}MHz in $((slow * 100 / total))% of samples; next runs lock to ${next}MHz" >&2
}

# NVBench discards samples below this percent of max; keep it just under the
# locked clock.
gpu_throttle_threshold() {
  local threshold=$(($1 * 100 / $2 - 2))
  ((threshold >= 0)) || threshold=0
  printf '%s\n' "$threshold"
}

# One lock file per GPU UUID serializes runs from every tool, worktree, and user.
# Opens it on fd 9; created world-writable so other users can share it.
gpu_open_lock() {
  local file
  file=${LG_GPU_LOCK_DIR:-/tmp}/lg-gpu-$1.lock
  if [ ! -e "$file" ]; then
    (umask 000 && : >>"$file") 2>/dev/null || true
  fi
  exec 9>>"$file"
}

gpu_wait_lock() {
  if ! flock -n 9; then
    echo "gpu: $(hostname) GPU $1 busy; waiting" >&2
    flock 9
  fi
}

run_gpu=
run_locked=0
run_keep=0
run_child=
gpu_run_finish() {
  local status=$?
  trap - EXIT HUP INT TERM
  gpu_stop_watch
  if [ "$run_locked" = 1 ] && [ "$run_keep" = 0 ]; then
    gpu_unlock_now "$run_gpu" || true
  fi
  exit "$status"
}

# A trapped signal does not reach the app, so pass it on as TERM; background
# jobs ignore INT when job control is off.
gpu_run_stop() {
  if [ -n "$run_child" ]; then
    kill -TERM "$run_child" 2>/dev/null || true
    wait "$run_child" 2>/dev/null || true
  fi
  exit "$1"
}

# gpu_run GPU CLOCK FRACTION KEEP -- APP [ARGS...]
# Waits for the GPU, locks its clock unless CLOCK is off, runs APP pinned to the
# GPU, and unlocks afterward unless KEEP is 1 (a dev session keeps the lock).
gpu_run() {
  local clock=$2 fraction=$3 info uuid max_clock persistence has_threshold=0 arg status=0
  run_gpu=$1
  run_keep=$4
  shift 4
  [ "${1:-}" != -- ] || shift
  info=$(gpu_query "$run_gpu" uuid,clocks.max.sm,persistence_mode)
  IFS=', ' read -r uuid max_clock persistence <<<"$info"
  if [[ ! "$uuid" =~ ^GPU-[a-fA-F0-9-]+$ ]]; then
    echo "gpu: GPU $run_gpu unavailable" >&2
    return 1
  fi
  gpu_open_lock "$uuid"
  gpu_wait_lock "$run_gpu"
  trap gpu_run_finish EXIT
  trap 'gpu_run_stop 129' HUP
  trap 'gpu_run_stop 130' INT
  trap 'gpu_run_stop 143' TERM

  local request=$clock
  if [ "$clock" != off ]; then
    gpu_check_health "$run_gpu"
    if [ "$persistence" = Disabled ]; then
      echo "gpu: persistence mode off; the clock lock may not hold" >&2
    fi
    if clock=$(gpu_pick_clock "$run_gpu" "$clock" "$fraction" "$max_clock") \
      && { [ "$request" != auto ] || clock=$(gpu_cap_clock "$uuid" "$clock"); } \
      && gpu_admin "$run_gpu" -lgc "$clock,$clock"; then
      run_locked=1
      # The runner flags runs whose measured clock fell below this.
      export LG_LOCKED_CLOCK_MHZ=$clock
      echo "gpu: $(hostname) GPU $run_gpu locked to ${clock}MHz" >&2
      for arg in "$@"; do
        case "$arg" in
          --throttle-threshold*) has_threshold=1 ;;
        esac
      done
      if [ "$has_threshold" = 0 ] && [[ "$max_clock" =~ ^[1-9][0-9]{0,5}$ ]]; then
        set -- "$@" --throttle-threshold "$(gpu_throttle_threshold "$clock" "$max_clock")"
      fi
    else
      echo "gpu: clock lock failed; running unlocked: $(hostname) GPU $run_gpu" >&2
    fi
  fi
  if [ "$run_locked" = 1 ] && [ "$request" = auto ]; then
    gpu_watch_clock "$run_gpu"
  fi
  CUDA_VISIBLE_DEVICES=$uuid "$@" <&0 &
  run_child=$!
  wait "$run_child" || status=$?
  run_child=
  if [ "$run_locked" = 1 ] && [ "$request" = auto ]; then
    gpu_review_clock "$run_gpu" "$uuid" "$clock" "$max_clock"
  fi
  return "$status"
}

gpu_unlock_now() {
  if ! gpu_admin "$1" -rgc; then
    echo "gpu: clock unlock failed: $(hostname) GPU $1" >&2
    return 1
  fi
  echo "gpu: $(hostname) GPU $1 unlocked" >&2
}

# Waits for any run on the GPU to finish before resetting its clock.
gpu_unlock() {
  local uuid
  uuid=$(gpu_query "$1" uuid)
  gpu_open_lock "$uuid"
  gpu_wait_lock "$1"
  gpu_unlock_now "$1"
}

if [ "${LG_GPU_MAIN:-}" = 1 ]; then
  set -euo pipefail
  action=${1:-}
  shift || true
  case "$action" in
    run) gpu_run "$@" ;;
    unlock) gpu_unlock "$@" ;;
    *)
      echo "usage: gpu.sh run GPU CLOCK FRACTION KEEP -- APP [ARGS...] | unlock GPU" >&2
      exit 2
      ;;
  esac
fi
