#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--dry-run" ]]; then
  "${ROOT}/.venv/bin/python" -m src.matrix_runner --suite ablation; exit 0
fi
echo "Full ablations require generated formal traces and the matrix runner; use --dry-run to inspect 24 jobs." >&2; exit 2
