#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${ROOT}/.venv/bin/python" -m src.harness --config "${ROOT}/configs/resource_adapted_smoke.yaml"
"${ROOT}/.venv/bin/python" -m src.analysis.audit_results --results "${ROOT}/results.csv" --output "${ROOT}/outputs/metrics/latest_smoke_audit.json"
"${ROOT}/.venv/bin/python" -m src.analysis.plot_smoke --results "${ROOT}/results.csv" --output-dir "${ROOT}/outputs/figures"
