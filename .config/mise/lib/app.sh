#!/usr/bin/env bash

normalize_app() {
  local app=$1

  while [[ $app == ./* ]]; do
    app=${app#./}
  done
  app=${app#apps/}
  app=${app#src/}
  app=${app%.cu}
  app=${app%.cpp}

  if [ -n "$app" ] && [[ ! "$app" =~ ^[a-z0-9][a-z0-9/-]*[a-z0-9]$ ]]; then
    echo "invalid app: $app" >&2
    return 2
  fi

  printf '%s\n' "$app"
}
