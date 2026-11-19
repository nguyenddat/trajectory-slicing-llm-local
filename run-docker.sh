#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$project_dir/outputs"

docker run --rm --init --network host \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$project_dir/outputs,target=/app/outputs" \
  trajectory-slicing-llm-local "$@"
