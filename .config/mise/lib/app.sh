#!/usr/bin/env bash

normalize_app() {
  local app=$1

  if [ -n "$app" ] && [[ ! "$app" =~ ^[a-z0-9][a-z0-9/-]*[a-z0-9]$ ]]; then
    echo "invalid app: $app" >&2
    return 2
  fi

  app=${app#apps/}
  app=${app#src/}
  printf '%s\n' "$app"
}
