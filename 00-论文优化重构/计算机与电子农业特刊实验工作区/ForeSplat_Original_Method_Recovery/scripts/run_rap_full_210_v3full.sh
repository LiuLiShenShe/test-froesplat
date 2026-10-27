#!/usr/bin/env bash
# Full 210-frame RAP-FSAM3 *complete chain* with the Round-3 (v3) fixes.
#
# Why this exists: the first v3 launch (run_rap_full_210_v3.sh) omitted
#   --spnp_backend sam3_if_supported   (default is "none")
#   --thin_repair_mode adjacent        (default is "none")
# so SB-FBPR and residual repair were silently skipped on all 210 frames
# (逐帧报告.csv: sb_fbpr_reason=backend_off, residual_repair_executed=0).
# Those masks are "selection + GCFC only" and must not be reported as the
# complete RAP chain.
#
# This re-run reuses the already-generated candidate pool (the expensive SAM3
# multi-prompt inference) via --reuse_existing_candidates, and re-executes the
# full decision chain: SG-MPS -> CDR/r1 -> SB-FBPR(points) -> residual repair
# (adjacent) -> GCFC.  It writes to a NEW directory so the selection-only v3
# masks are preserved.
set -euo pipefail

ROOT="/data/fj/F2DMAS"
S="${ROOT}/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
PROD="${ROOT}/00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py"
DATA="${ROOT}/00-论文优化重构/数据管理/02-位姿COLMAP/03-final_locked/KongQueZhuYu"
SRC="${S}/runs/KongQueZhuYu_RAP_FULL_210_v3"      # candidate pool to reuse
OUT="${S}/runs/KongQueZhuYu_RAP_FULL_210_v3full"  # complete-chain output
LOG="${S}/logs/KongQueZhuYu_RAP_FULL_210_v3full.log"

mkdir -p "${S}/logs"
PY="/home/test/.venvs/sam3/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"${PY}" "${PROD}" \
  --input_dir "${DATA}/images" \
  --output_dir "${OUT}" \
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
  --semantic_gate_target_mode potted_plant \
  --candidate_area_admissibility gate \
  --use_spnp_refinement \
  --spnp_backend sam3_if_supported \
  --spnp_prompt_form points \
  --spnp_negative_region_mode manuscript \
  --use_residual_repair \
  --thin_repair_mode adjacent \
  --use_corrective_geometry \
  --corrective_geometry_backend colmap_tracks_gaussian \
  --colmap_dir "${DATA}" \
  --strict_colmap_observations \
  --force \
  2>&1 | tee "${LOG}"
