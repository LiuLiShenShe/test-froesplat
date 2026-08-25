#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scene_config.sh"

MODE="smoke"
SCENES="${DEFAULT_SCENES}"
METHODS="${DEFAULT_METHODS}"
DRY_RUN=0
PREPARE_ASSETS=1
RESOLUTION="${DEFAULT_RESOLUTION}"

usage() {
  cat <<'USAGE'
Usage:
  02_run_three_scene_pilot.sh [options]

Options:
  --mode smoke|full          smoke=19000 iters, full=30000 iters. Default: smoke
  --scenes A,B,C             Scene ids from scene_manifest.csv.
  --methods A,B,C            fg2dgs_baseline,fg2dgs_m2m3,fg2dgs_m2m3_floor40.
  --resolution N             Passed to train.py --resolution. Default from scene_config.sh.
  --dry-run                  Print commands without launching training.
  --no-prepare               Do not refresh prepared mask/view-weight assets first.
  -h, --help                 Show this help.

Environment:
  CUDA_VISIBLE_DEVICES       GPU id(s). Default falls back to DEFAULT_GPU in scene_config.sh.
  RUN_TAG                    Optional output run tag. Default: timestamp_mode.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --scenes)
      SCENES="$2"
      shift 2
      ;;
    --methods)
      METHODS="$2"
      shift 2
      ;;
    --resolution)
      RESOLUTION="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-prepare)
      PREPARE_ASSETS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${MODE}" in
  smoke)
    ITERATIONS=19000
    TEST_ITERATIONS=(19000)
    SAVE_ITERATIONS=(19000)
    ;;
  full)
    ITERATIONS=30000
    TEST_ITERATIONS=(-1)
    SAVE_ITERATIONS=(7000 30000)
    ;;
  *)
    echo "--mode must be smoke or full" >&2
    exit 2
    ;;
esac

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_${MODE}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_GPU}}"

if [[ "${PREPARE_ASSETS}" -eq 1 ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/01_prepare_three_scene_assets.py" \
    --manifest "${SCENE_MANIFEST}" \
    --output-root "${OUTPUT_ROOT}" \
    --scenes "${SCENES}" \
    --allow-missing-masks \
    --force
fi

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

method_options() {
  local method="$1"
  case "${method}" in
    fg2dgs_baseline)
      echo "--capacity_control_mode none"
      ;;
    fg2dgs_m2m3)
      echo "--capacity_control_mode m2m3 --m2m3_score_mode topology --m2m3_region_mode foreground --m2m3_max_remove_ratio 0.03 --save_capacity_report --capacity_report_interval 1000"
      ;;
    fg2dgs_m2m3_floor40)
      echo "--capacity_control_mode m2m3_floor --capacity_floor_ratio 0.4 --capacity_floor_reference max_seen --capacity_floor_start_iter 18000 --m2m3_score_mode topology --m2m3_region_mode foreground --m2m3_max_remove_ratio 0.03 --save_capacity_report --capacity_report_interval 1000"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 2
      ;;
  esac
}

scene_index=0
while IFS= read -r scene; do
  scene_index=$((scene_index + 1))
  scene_exists "${scene}" || { echo "Unknown scene: ${scene}" >&2; exit 2; }
  source_dir="$(manifest_field "${scene}" source_dir)"
  prepared_mask_dir="${PREPARED_MASK_ROOT}/${scene}"
  prepared_weight="${PREPARED_WEIGHT_ROOT}/${scene}_view_weights.csv"
  prepared_gate="${OUTPUT_ROOT}/prepared_gate_lists/${scene}_mask_matched.txt"

  if [[ ! -d "${source_dir}" ]]; then
    echo "Missing source_dir for ${scene}: ${source_dir}" >&2
    exit 1
  fi
  if [[ ! -d "${prepared_mask_dir}" ]]; then
    echo "Missing prepared mask dir for ${scene}: ${prepared_mask_dir}" >&2
    echo "Run 01_prepare_three_scene_assets.py first." >&2
    exit 1
  fi
  if [[ ! -s "${prepared_gate}" ]]; then
    echo "Missing prepared gate list for ${scene}: ${prepared_gate}" >&2
    echo "Run 01_prepare_three_scene_assets.py first." >&2
    exit 1
  fi
  image_count="$(find "${source_dir}/images" -maxdepth 1 -type f | wc -l)"
  gate_count="$(grep -cve '^[[:space:]]*$' "${prepared_gate}")"
  init_pcd_mode="${INIT_PCD_MODE:-foreground_track}"
  if [[ "${gate_count}" -lt "${image_count}" && "${init_pcd_mode}" == "foreground_track" ]]; then
    init_pcd_mode="foreground_mask"
  fi

  method_index=0
  while IFS= read -r method; do
    method_index=$((method_index + 1))
    model_dir="${RUNS_ROOT}/${RUN_TAG}/${scene}/${method}"
    log_file="${LOG_ROOT}/${RUN_TAG}_${scene}_${method}.log"
    port=$((BASE_PORT + scene_index * 10 + method_index))
    if [[ "${DRY_RUN}" -eq 0 ]]; then
      mkdir -p "${model_dir}" "$(dirname "${log_file}")"
    fi

    cmd=(
      "${PYTHON_BIN}" train.py
      --source_path "${source_dir}"
      --model_path "${model_dir}"
      --eval
      --resolution "${RESOLUTION}"
      --mask_mode alpha
      --mask_dir "${prepared_mask_dir}"
      --mask_pattern 'mask_{stem}.png'
      --mask_threshold 0.5
      --mask_gate_list "${prepared_gate}"
      --init_pcd_mode "${init_pcd_mode}"
      --init_pcd_min_observations 3
      --init_pcd_foreground_threshold 0.9
      --init_pcd_dilate_mask_px 0
      --use_foreground_rgb_loss
      --use_mask_loss
      --use_bg_opacity_loss
      --lambda_mask 0.08
      --lambda_bg 0.02
      --mask_loss_type l1_dice
      --mask_ignore_boundary_px 2
      --mask_loss_start_iter 500
      --mask_loss_warmup_iters 1500
      --pruning_mode mask
      --pruning_start_iter 18000
      --pruning_interval 3000
      --pruning_opacity_threshold 0.005
      --pruning_brightness_threshold 0.01
      --pruning_mask_threshold 0.45
      --pruning_mask_max_views 12
      --pruning_max_remove_ratio 0.03
      --pruning_mask_score_weight 3.0
      --save_pruning_report
      --ip 127.0.0.1
      --port "${port}"
      --iterations "${ITERATIONS}"
      --test_iterations "${TEST_ITERATIONS[@]}"
      --save_iterations "${SAVE_ITERATIONS[@]}"
    )

    if [[ -s "${prepared_weight}" ]]; then
      cmd+=(--view_weight_mode rgb_only --view_weight_list "${prepared_weight}" --view_weight_min 0.6 --view_weight_max 1.0 --view_weight_default 1.0)
    fi

    read -r -a extra_opts <<< "$(method_options "${method}")"
    cmd+=("${extra_opts[@]}")

    if [[ "${DRY_RUN}" -eq 0 ]]; then
      {
        echo "# run_tag: ${RUN_TAG}"
        echo "# scene: ${scene}"
        echo "# method: ${method}"
        echo "# mode: ${MODE}"
        echo "# image_count: ${image_count}"
        echo "# mask_matched_gate_count: ${gate_count}"
        echo "# init_pcd_mode: ${init_pcd_mode}"
        echo "# started_at: $(date --iso-8601=seconds)"
        echo "# command:"
        quote_cmd "${cmd[@]}"
      } > "${model_dir}/command.txt"
    fi

    echo "[M2M3] ${MODE} ${scene} ${method}"
    echo "  model_dir=${model_dir}"
    echo "  log_file=${log_file}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      quote_cmd "${cmd[@]}"
    else
      (
        cd "${M2M3_REPO}"
        quote_cmd "${cmd[@]}"
        "${cmd[@]}"
      ) 2>&1 | tee "${log_file}"
    fi
  done < <(split_csv "${METHODS}")
done < <(split_csv "${SCENES}")

echo "[M2M3] Done. run_tag=${RUN_TAG}"
