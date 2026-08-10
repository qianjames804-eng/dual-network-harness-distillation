#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-preflight}"
IMAGE_NAME="${SPARK_IMAGE_NAME:-dual-nn-harness:spark}"
DATA_ROOT="${SPARK_DATA_ROOT:-${ROOT}/.spark-data}"

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
  echo "error: this launcher must run on the ARM64 DGX Spark host" >&2
  exit 1
fi
command -v docker >/dev/null || { echo "error: docker is required" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "error: nvidia-smi is required" >&2; exit 1; }
docker info >/dev/null
nvidia-smi >/dev/null

mkdir -p "${DATA_ROOT}/huggingface"
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"

if [[ "${SPARK_SKIP_BUILD:-0}" != "1" ]]; then
  docker build -f "${ROOT}/Dockerfile.spark" -t "${IMAGE_NAME}" "${ROOT}"
fi

DOCKER_ARGS=(
  --rm
  --gpus all
  --ipc=host
  --user "$(id -u):$(id -g)"
  --env HOME=/tmp
  --env HF_HOME=/workspace/experiment/.cache/huggingface
  --env TOKENIZERS_PARALLELISM=false
  --volume "${ROOT}:/workspace/experiment"
  --volume "${DATA_ROOT}/huggingface:/workspace/experiment/.cache/huggingface"
  --workdir /workspace/experiment
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  DOCKER_ARGS+=(--env HF_TOKEN)
fi

case "${MODE}" in
  preflight)
    docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" bash scripts/spark/preflight.sh
    ;;
  traces)
    EXTRA_ARGS=()
    if [[ -n "${TRACE_MAX_NEW_RECORDS:-}" ]]; then
      EXTRA_ARGS+=(--max-new-records "${TRACE_MAX_NEW_RECORDS}")
    fi
    docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" \
      python -m src.traces.generate_teacher \
      --config configs/paper_faithful_mvp.yaml "${EXTRA_ARGS[@]}"
    ;;
  mvp)
    docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" bash -lc \
      "bash scripts/spark/preflight.sh && bash run_mvp.sh"
    ;;
  shell)
    docker run -it "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" bash
    ;;
  *)
    echo "usage: $0 {preflight|traces|mvp|shell}" >&2
    exit 2
    ;;
esac
