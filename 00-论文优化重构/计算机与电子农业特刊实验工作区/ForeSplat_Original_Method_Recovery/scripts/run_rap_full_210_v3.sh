#!/usr/bin/env bash
# Full 210-frame RAP-FSAM3 regeneration with the Round-3 (v3) fixes.
#
# This is the *post-fix* counterpart of runs/KongQueZhuYu_RAP_FULL_210, which was
# produced on 2026-10-25 with the pre-fix code (P2 = "entire plant excluding pot",
# background-green gate reference, pot band at 0.58*H, no q_pot).  The old run is
# kept untouched; this one writes to runs/KongQueZhuYu_RAP_FULL_210_v3.
#
# Differences vs the pre-fix 参数.json (all are the Round-3 defaults, made explicit):
#   --semantic_gate_target_mode potted_plant   (was implicit / plant-only behaviour)
#   --candidate_area_admissibility gate         (was off)
#   P2 prompt text                             (now "entire potted plant"; see DEFAULT_PROMPT_TEXTS)
#   gate reference q_plant / q_bg / q_pot / pot band (code-level fixes, no flag)
#
# Runtime: the pre-fix run measured 20867 s (~5.8 h) for 210 frames on a contended
# GPU.  Launch in the background and poll logs/KongQueZhuYu_RAP_FULL_210_v3.log.
set -euo pipefail

ROOT="/data/fj/F2DMAS"
S="${ROOT}/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
PROD="${ROOT}/00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py"
DATA="${ROOT}/00-论文优化重构/数据管理/02-位姿COLMAP/03-final_locked/KongQueZhuYu"
OUT="${S}/runs/KongQueZhuYu_RAP_FULL_210_v3"
LOG="${S}/logs/KongQueZhuYu_RAP_FULL_210_v3.log"

mkdir -p "${S}/logs"

PY="/home/test/.venvs/sam3/bin/python"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

"${PY}" "${PROD}" \
  --input_dir "${DATA}/images" \
  --output_dir "${OUT}" \
  --device cuda \
  --candidate_mode per_instance \
  --prompt_list P1,P2,P3,P4,P5 \
  --default_prompt_id P2 \
  --use_prompt_ensemble \
  --prompt_selection_mode score_select \
  --ranking_policy r1 \
  --use_semantic_gate \
  --semantic_gate_mode gated_softmax \
  --semantic_gate_temperature 0.25 \
  --semantic_gate_target_mode potted_plant \
  --candidate_area_admissibility gate \
  --use_spnp_refinement \
  --spnp_prompt_form points \
  --spnp_negative_region_mode manuscript \
  --use_residual_repair \
  --use_corrective_geometry \
  --corrective_geometry_backend colmap_tracks_gaussian \
  --colmap_dir "${DATA}" \
  --strict_colmap_observations \
  --force \
  2>&1 | tee "${LOG}"
