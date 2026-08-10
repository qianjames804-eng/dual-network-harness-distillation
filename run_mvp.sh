#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m src.traces.generate_teacher \
  --config "${ROOT}/configs/paper_faithful_mvp.yaml"
"${PYTHON_BIN}" -m src.harness \
  --config "${ROOT}/configs/paper_faithful_mvp.yaml"
