#!/usr/bin/env bash
set -euo pipefail

python -m tools.preflight_spark
python -m src.validate_config --config configs/paper_faithful_mvp.yaml
