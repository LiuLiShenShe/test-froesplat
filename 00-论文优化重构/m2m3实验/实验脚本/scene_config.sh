#!/usr/bin/env bash
set -euo pipefail

M2M3_REPO="${M2M3_REPO:-/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40}"
EXP_ROOT="${EXP_ROOT:-/data/fj/F2DMAS/00-论文优化重构/m2m3实验}"
SCRIPT_ROOT="${SCRIPT_ROOT:-${EXP_ROOT}/实验脚本}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXP_ROOT}/实验输出}"
SCENE_MANIFEST="${SCENE_MANIFEST:-${SCRIPT_ROOT}/scene_manifest.csv}"

RUNS_ROOT="${RUNS_ROOT:-${OUTPUT_ROOT}/runs}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
REPORT_ROOT="${REPORT_ROOT:-${OUTPUT_ROOT}/reports}"
PREPARED_MASK_ROOT="${PREPARED_MASK_ROOT:-${OUTPUT_ROOT}/prepared_masks}"
PREPARED_WEIGHT_ROOT="${PREPARED_WEIGHT_ROOT:-${OUTPUT_ROOT}/prepared_view_weights}"

PYTHON_BIN="${PYTHON_BIN:-${M2M3_REPO}/venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

DEFAULT_SCENES="${DEFAULT_SCENES:-KongQueZhuYu,ChangShouHua2,CaoMei2}"
DEFAULT_METHODS="${DEFAULT_METHODS:-fg2dgs_baseline,fg2dgs_m2m3,fg2dgs_m2m3_floor40}"
DEFAULT_RESOLUTION="${DEFAULT_RESOLUTION:-4}"
DEFAULT_GPU="${DEFAULT_GPU:-0}"
BASE_PORT="${BASE_PORT:-17100}"

mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}" "${REPORT_ROOT}" "${PREPARED_MASK_ROOT}" "${PREPARED_WEIGHT_ROOT}"

manifest_field() {
  local scene="$1"
  local field="$2"
  awk -F, -v scene="${scene}" -v field="${field}" '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        header[$i] = i
      }
      next
    }
    $1 == scene {
      print $header[field]
      found = 1
      exit
    }
    END {
      if (!found) {
        exit 1
      }
    }
  ' "${SCENE_MANIFEST}"
}

scene_exists() {
  local scene="$1"
  awk -F, -v scene="${scene}" 'NR > 1 && $1 == scene { found = 1 } END { exit(found ? 0 : 1) }' "${SCENE_MANIFEST}"
}

split_csv() {
  local value="$1"
  tr ',' '\n' <<< "${value}" | sed '/^[[:space:]]*$/d'
}
