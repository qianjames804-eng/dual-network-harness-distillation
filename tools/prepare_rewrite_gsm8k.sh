#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${ROOT}/configs/gsm8k_seed42_rewrite.yaml"
"${ROOT}/.venv/bin/python" -m src.traces.official_rewrite --source "${ROOT}/external/trace-rewriting/data/gsm8k/optimized" --output "${ROOT}/outputs/traces/gsm8k_seed42_rewrite/official.jsonl" --provenance "${ROOT}/provenance/rewrite_gsm8k_s42.json" --seed 42 --examples 512 --cache-dir "${ROOT}/.cache/huggingface" --revision 740312add88f781978c0658806c59bc2815b9866
