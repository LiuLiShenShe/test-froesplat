#!/usr/bin/env bash
# Downstream Foreground-Object 2DGS on the complete-chain v3full masks.
#
# Three tags, serial (parallel runs have historically triggered the OOM killer):
#   1 RAP_FULL_v3full          §2.5 ON  (rgb_only)      -> the main result
#   2 RAP_FULL_v3full_M2M3GO   §2.5 ON  + M2M3-GO full  -> M2M3 effect
#   3 RAP_FULL_v3full_novw     §2.5 OFF (none)          -> §2.5 on/off ablation
# The C_R1 baseline (frozen single-prompt masks) already exists and is reused
# for the mask-source comparison.
set -uo pipefail

S="/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
PY="/home/test/.venvs/sam3/bin/python"
MASKS="${S}/runs/KongQueZhuYu_RAP_FULL_210_v3full"
LOG="${S}/logs/downstream_v3full_chain.log"
mkdir -p "${S}/logs"

run_tag() {
  local tag="$1"; shift
  echo "=== [$(date '+%F %T')] START ${tag} ===" | tee -a "${LOG}"
  "${PY}" "${S}/scripts/run_downstream_2dgs.py" --scene KongQueZhuYu \
    --mask-run-dir "${MASKS}" --tag "${tag}" "$@" >>"${LOG}" 2>&1
  echo "=== [$(date '+%F %T')] END ${tag} rc=$? ===" | tee -a "${LOG}"
}

run_tag RAP_FULL_v3full        --iterations 30000 --resolution 2 --view-weight-mode rgb_only  --port 6030
run_tag RAP_FULL_v3full_M2M3GO --iterations 30000 --resolution 2 --view-weight-mode rgb_only  --port 6050 --m2m3-go
run_tag RAP_FULL_v3full_novw   --iterations 30000 --resolution 2 --view-weight-mode none      --port 6060

echo "=== [$(date '+%F %T')] ALL TAGS DONE ===" | tee -a "${LOG}"
