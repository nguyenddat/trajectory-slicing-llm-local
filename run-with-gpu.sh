#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner_args=("$@")
model=""
for ((i = 0; i < ${#runner_args[@]}; i++)); do
  case "${runner_args[i]}" in
    --model)
      if ((i + 1 >= ${#runner_args[@]})); then
        echo "--model needs an Ollama model tag" >&2
        exit 2
      fi
      model="${runner_args[i + 1]}"
      ;;
    --url)
      echo "This script provides its own Ollama URL; use run-docker.sh for an external Ollama server" >&2
      exit 2
      ;;
  esac
done
if [[ -z "$model" ]]; then
  echo "Usage: bash run-with-gpu.sh --model MODEL [--context-length N]" >&2
  exit 2
fi

mkdir -p "$project_dir/outputs"

for arg in "${runner_args[@]}"; do
  if [[ "$arg" == "--dry-run" ]]; then
    bash "$project_dir/run-docker.sh" "${runner_args[@]}" --url http://localhost:11434
    exit
  fi
done
bash "$project_dir/run-docker.sh" "${runner_args[@]}" --url http://localhost:11434 --dry-run >/dev/null

# Ollama's recommended NVIDIA container check. Fail before starting the job if
# Docker cannot pass the GPU through to a container.
docker run --rm --gpus all ubuntu nvidia-smi >/dev/null

ollama_container="trajectory-slicing-ollama-$$"
cleanup() {
  docker rm -f "$ollama_container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --rm --gpus all \
  --name "$ollama_container" \
  --mount type=volume,source=trajectory-slicing-ollama-models,target=/root/.ollama \
  ollama/ollama >/dev/null

ready=0
for ((attempt = 0; attempt < 60; attempt++)); do
  if docker exec "$ollama_container" ollama list >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if ((ready == 0)); then
  docker logs "$ollama_container" >&2
  echo "Ollama did not start within two minutes" >&2
  exit 1
fi

docker exec "$ollama_container" ollama pull "$model"
docker run --rm --init \
  --network "container:$ollama_container" \
  trajectory-slicing-llm-local "${runner_args[@]}" --url http://localhost:11434 --check-gpu
docker run --rm --init \
  --network "container:$ollama_container" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$project_dir/outputs,target=/app/outputs" \
  trajectory-slicing-llm-local "${runner_args[@]}" --url http://localhost:11434
