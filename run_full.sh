#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--dry-run" ]]; then
  "${ROOT}/.venv/bin/python" -m src.matrix_runner --suite full
  exit 0
fi
echo "Refusing to launch full GPU matrix implicitly. Use: python -m src.matrix_runner --suite full --execute --execute-template '<command with {dataset} {seed} {method}>' after formal trace generation. See README." >&2
exit 2
