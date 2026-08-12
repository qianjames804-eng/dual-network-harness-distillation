#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${1:?usage: run_antidistill.sh {ads|doge|rewrite} --official-command '...'}"; shift
"${ROOT}/.venv/bin/python" -m src.defenses.generate --method "${METHOD}" "$@"
