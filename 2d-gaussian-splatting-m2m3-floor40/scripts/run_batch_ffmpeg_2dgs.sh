#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/fj/2d-gaussian-splatting-main}"
INPUT_ROOT="${INPUT_ROOT:-/data/fj/01-FFmepg}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/fj/05-2DGS-ground}"
COLMAP_BIN="${COLMAP_BIN:-/data/fj/tools/colmap-cuda/bin/colmap}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/venv/bin/python}"
GPU_ID="${GPU_ID:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OUTPUT_ROOT}/run_${RUN_ID}"
LOG_DIR="${RUN_ROOT}/logs"
STATUS_CSV="${RUN_ROOT}/status.csv"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

if [ "$#" -gt 0 ]; then
  SCENES=("$@")
else
  mapfile -t SCENES < <(find "${INPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | grep -v '^ffmpeg_bin$')
fi

printf 'scene,status,stage,start_time,end_time,elapsed_sec,detail\n' > "${STATUS_CSV}"

echo "[INFO] preparing filtered input directories under ${OUTPUT_ROOT}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prepare_filtered_inputs.py" \
  --input_root "${INPUT_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  $(printf ' --scene %q' "${SCENES[@]}")

for scene in "${SCENES[@]}"; do
  scene_root="${OUTPUT_ROOT}/${scene}"
  train_root="${scene_root}/train_30000"
  convert_log="${LOG_DIR}/${scene}_convert.log"
  train_log="${LOG_DIR}/${scene}_train.log"
  final_ply="${train_root}/point_cloud/iteration_30000/point_cloud.ply"
  start_iso="$(date --iso-8601=seconds)"
  start_epoch="$(date +%s)"

  if [ -f "${final_ply}" ]; then
    end_iso="$(date --iso-8601=seconds)"
    printf '%s,%s,%s,%s,%s,%s,%s\n' "${scene}" "skipped" "all" "${start_iso}" "${end_iso}" "0" "existing_final_output" >> "${STATUS_CSV}"
    continue
  fi

  if [ ! -f "${scene_root}/sparse/0/images.bin" ]; then
    echo "[INFO] ${scene}: running COLMAP convert"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${PROJECT_ROOT}/convert.py" \
      -s "${scene_root}" \
      --colmap_executable "${COLMAP_BIN}" \
      > "${convert_log}" 2>&1
    rc=$?
    if [ "${rc}" -ne 0 ]; then
      end_iso="$(date --iso-8601=seconds)"
      elapsed="$(( $(date +%s) - start_epoch ))"
      printf '%s,%s,%s,%s,%s,%s,%s\n' "${scene}" "failed" "convert" "${start_iso}" "${end_iso}" "${elapsed}" "see_${scene}_convert.log" >> "${STATUS_CSV}"
      continue
    fi
  fi

  echo "[INFO] ${scene}: running 2DGS training"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${PROJECT_ROOT}/train.py" \
    -s "${scene_root}" \
    -m "${train_root}" \
    > "${train_log}" 2>&1
  rc=$?
  end_iso="$(date --iso-8601=seconds)"
  elapsed="$(( $(date +%s) - start_epoch ))"
  if [ "${rc}" -ne 0 ]; then
    printf '%s,%s,%s,%s,%s,%s,%s\n' "${scene}" "failed" "train" "${start_iso}" "${end_iso}" "${elapsed}" "see_${scene}_train.log" >> "${STATUS_CSV}"
    continue
  fi

  printf '%s,%s,%s,%s,%s,%s,%s\n' "${scene}" "completed" "all" "${start_iso}" "${end_iso}" "${elapsed}" "${final_ply}" >> "${STATUS_CSV}"
done
