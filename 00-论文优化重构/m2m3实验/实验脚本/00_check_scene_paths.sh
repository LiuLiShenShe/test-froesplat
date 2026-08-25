#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scene_config.sh"

"${PYTHON_BIN}" "${SCRIPT_DIR}/01_prepare_three_scene_assets.py" \
  --manifest "${SCENE_MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --scenes "${1:-${DEFAULT_SCENES}}" \
  --allow-missing-masks \
  --check-only
