#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
"${PYTHON_BIN}" -m src.traces.generate_teacher \
  --config "${ROOT}/configs/paper_faithful_smoke.yaml"
"${PYTHON_BIN}" -m src.harness \
  --config "${ROOT}/configs/paper_faithful_smoke.yaml"
