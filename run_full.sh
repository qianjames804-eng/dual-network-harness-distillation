#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m src.validate_config --config "${ROOT}/configs/full.yaml"
echo "blocked by design: Full starts only after paper-faithful B0/B1, NN1 and NN2 gates pass" >&2
exit 2
