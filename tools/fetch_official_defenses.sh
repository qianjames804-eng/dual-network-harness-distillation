#!/usr/bin/env bash
# Fetch source snapshots outside the tracked experiment tree, then record pins.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${DUALGUARD_EXTERNAL_DIR:-${ROOT}/external}"
mkdir -p "${TARGET}"
fetch() { local name="$1" url="$2"; if [[ ! -d "${TARGET}/${name}/.git" ]]; then git clone --depth 1 "${url}" "${TARGET}/${name}"; else git -C "${TARGET}/${name}" fetch origin main --depth 1; git -C "${TARGET}/${name}" reset --hard FETCH_HEAD; fi; }
fetch DOGe https://github.com/UNITES-Lab/DOGe.git
fetch trace-rewriting https://github.com/xhOwenMa/trace-rewriting.git
python3 - "${TARGET}" <<'PY'
import json, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1]); pins={name:subprocess.check_output(["git","-C",str(root/name),"rev-parse","HEAD"],text=True).strip() for name in ("DOGe","trace-rewriting")}
(root/"official_defense_pins.json").write_text(json.dumps(pins,indent=2)+"\n")
print(json.dumps(pins,indent=2))
PY
