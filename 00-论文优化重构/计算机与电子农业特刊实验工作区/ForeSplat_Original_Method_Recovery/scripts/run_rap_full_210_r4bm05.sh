#!/usr/bin/env bash
# Full 210-frame RAP-FSAM3 production run matching the *measured* feasibility
# winner R4bm05 (dev 16 frames: F1 0.985961, +0.002225 over the single-P2
# baseline, SF-FBPR box prompting).
#
# Why this exists: the existing 210-frame masks do NOT correspond to R4bm05.
#   runs/KongQueZhuYu_RAP_FULL_210_v3    : missing SB-FBPR entirely (backend none)
#   runs/KongQueZhuYu_RAP_FULL_210_v3full: margin 0.0, spnp points/manuscript,
#                                          area_admissibility gate, + repair + GCFC
# R4bm05 (the config with the measured positive gain) is:
#   margin 0.05, spnp boxes/ring, area_admissibility off, NO repair, NO GCFC.
# This run makes "paper method == code == reported result" true.
#
# Candidate pool is reused from the v3 run (the expensive SAM3 multi-prompt
# inference); only the decision chain is re-executed:
#   SG-MPS(gated_softmax, margin 0.05) -> CDR/r1 -> SB-FBPR(boxes, ring)
set -euo pipefail

ROOT="/data/fj/F2DMAS"
S="${ROOT}/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
PROD="${ROOT}/00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py"
DATA="${ROOT}/00-论文优化重构/数据管理/02-位姿COLMAP/03-final_locked/KongQueZhuYu"
SRC="${S}/runs/KongQueZhuYu_RAP_FULL_210_v3"        # candidate pool to reuse
OUT="${S}/runs/KongQueZhuYu_RAP_FULL_210_R4bm05"    # R4bm05 output
LOG="${S}/logs/KongQueZhuYu_RAP_FULL_210_R4bm05.log"

mkdir -p "${S}/logs"
PY="/home/test/.venvs/sam3/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"${PY}" "${PROD}" \
  --input_dir "${DATA}/images" \
  --output_dir "${OUT}" \
  --sam3_repo "${ROOT}/第三方源码/SAM3-latest" \
  --sam3_checkpoint "${ROOT}/sam3/sam3.pt" \
  --device cuda \
  --reuse_existing_candidates \
  --candidate_source_dir "${SRC}" \
  --candidate_mode per_instance \
  --prompt_list P1,P2,P3,P4,P5 \
  --default_prompt_id P2 \
  --use_prompt_ensemble \
  --prompt_selection_mode score_select \
  --ranking_policy r1 \
  --use_semantic_gate \
  --semantic_gate_mode gated_softmax \
  --semantic_gate_temperature 0.25 \
  --semantic_gate_margin 0.05 \
  --semantic_gate_target_mode potted_plant \
  --candidate_area_admissibility off \
  --use_spnp_refinement \
  --spnp_backend sam3_if_supported \
  --spnp_prompt_form boxes \
  --spnp_negative_region_mode ring \
  --force \
  2>&1 | tee "${LOG}"
