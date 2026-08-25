#!/usr/bin/env bash
set -u
set -o pipefail

SOURCE_ROOT="/data/fj/04-COLMAP"
OUTPUT_ROOT="/data/fj/11-COLMAP-fuse"
DEFAULT_COLMAP_BIN="/data/fj/tools/colmap-cuda/bin/colmap"
COLMAP_BIN="${COLMAP_BIN:-${DEFAULT_COLMAP_BIN}}"
THREADS="${THREADS:-8}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-2000}"
SEQUENTIAL_OVERLAP="${SEQUENTIAL_OVERLAP:-15}"
ONLY_PLANT="${ONLY_PLANT:-}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/run_${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
SUMMARY_CSV="${RUN_DIR}/summary.csv"
LATEST_LINK="${OUTPUT_ROOT}/latest"
MASTER_LOG="${RUN_DIR}/run.log"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

log_master() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MASTER_LOG}"
}

# Resolve COLMAP binary.
if [ ! -x "${COLMAP_BIN}" ]; then
  COLMAP_BIN="$(command -v colmap || true)"
fi
if [ -z "${COLMAP_BIN}" ] || [ ! -x "${COLMAP_BIN}" ]; then
  echo "ERROR: colmap binary not found." >&2
  exit 1
fi

# Detect whether GPU1 can be used by COLMAP. Fallback to CPU otherwise.
USE_GPU=0
GPU_MODE="cpu"
GPU_NOTE="CPU fallback"
if "${COLMAP_BIN}" -h 2>&1 | grep -qi 'without CUDA'; then
  USE_GPU=0
  GPU_MODE="cpu"
  GPU_NOTE="COLMAP without CUDA"
elif [ -e /dev/nvidia1 ]; then
  USE_GPU=1
  GPU_MODE="gpu1"
  GPU_NOTE="using CUDA_VISIBLE_DEVICES=1 (/dev/nvidia1)"
  export CUDA_VISIBLE_DEVICES=1
else
  USE_GPU=0
  GPU_MODE="cpu"
  GPU_NOTE="GPU1 device not found"
fi

log_master "Run directory: ${RUN_DIR}"
log_master "COLMAP binary: ${COLMAP_BIN}"
log_master "COLMAP mode: ${GPU_MODE} (${GPU_NOTE})"
log_master "Threads: ${THREADS}, max_image_size: ${MAX_IMAGE_SIZE}, sequential_overlap: ${SEQUENTIAL_OVERLAP}"

printf 'plant,status,image_dir,image_count,compute_mode,total_sec,feature_sec,matcher,match_sec,mapper_sec,undistort_sec,patchmatch_sec,fusion_sec,fuse_ply\n' > "${SUMMARY_CSV}"

run_stage() {
  STAGE_ELAPSED=0
  local plant_log="$1"
  local stage="$2"
  shift 2
  local t0 t1 rc
  t0=$(date +%s)
  echo "[$(date '+%F %T')] START ${stage}: $*" >> "${plant_log}"
  "$@" >> "${plant_log}" 2>&1
  rc=$?
  t1=$(date +%s)
  STAGE_ELAPSED=$((t1 - t0))
  echo "[$(date '+%F %T')] END ${stage}: rc=${rc}, elapsed=${STAGE_ELAPSED}s" >> "${plant_log}"
  return ${rc}
}

choose_best_model() {
  local sparse_dir="$1"
  local best_model=""
  local best_images=-1

  while IFS= read -r model_dir; do
    if [ ! -f "${model_dir}/cameras.bin" ] || [ ! -f "${model_dir}/images.bin" ] || [ ! -f "${model_dir}/points3D.bin" ]; then
      continue
    fi

    local images_count
    images_count=$("${COLMAP_BIN}" model_analyzer --path "${model_dir}" 2>/dev/null | awk '/Registered images/ {print $NF; exit}')
    if [ -z "${images_count}" ]; then
      images_count=0
    fi

    if [ "${images_count}" -gt "${best_images}" ]; then
      best_images="${images_count}"
      best_model="${model_dir}"
    fi
  done < <(find "${sparse_dir}" -mindepth 1 -maxdepth 1 -type d | sort)

  echo "${best_model}"
}

find_image_dir() {
  local plant_dir="$1"
  if [ -d "${plant_dir}/images" ]; then
    echo "${plant_dir}/images"
    return 0
  fi
  if [ -d "${plant_dir}/input" ]; then
    echo "${plant_dir}/input"
    return 0
  fi
  return 1
}

# Discover plant directories
mapfile -t PLANTS < <(find "${SOURCE_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

for plant in "${PLANTS[@]}"; do
  case "${plant}" in
    colmap_bin|__pycache__)
      continue
      ;;
  esac

  if [ -n "${ONLY_PLANT}" ] && [ "${plant}" != "${ONLY_PLANT}" ]; then
    continue
  fi

  TOTAL=$((TOTAL + 1))

  plant_src="${SOURCE_ROOT}/${plant}"
  plant_out="${RUN_DIR}/${plant}"
  plant_log="${LOG_DIR}/${plant}.log"

  mkdir -p "${plant_out}"

  {
    echo "============================================================"
    echo "Plant: ${plant}"
    echo "Start: $(date '+%F %T')"
    echo "Source: ${plant_src}"
  } > "${plant_log}"

  if ! image_dir=$(find_image_dir "${plant_src}"); then
    echo "[$(date '+%F %T')] ERROR: no images/ or input/ directory found, skip." >> "${plant_log}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "${plant}" "skipped_no_images" "" "0" "${GPU_MODE}" "0" "0" "none" "0" "0" "0" "0" "0" "" >> "${SUMMARY_CSV}"
    SKIPPED=$((SKIPPED + 1))
    log_master "[${plant}] skipped: no images dir"
    continue
  fi

  image_count=$(find "${image_dir}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
  echo "[$(date '+%F %T')] Input images: ${image_count} from ${image_dir}" >> "${plant_log}"

  if [ "${image_count}" -lt 2 ]; then
    echo "[$(date '+%F %T')] ERROR: less than 2 images, skip." >> "${plant_log}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "${plant}" "skipped_too_few_images" "${image_dir}" "${image_count}" "${GPU_MODE}" "0" "0" "none" "0" "0" "0" "0" "0" "" >> "${SUMMARY_CSV}"
    SKIPPED=$((SKIPPED + 1))
    log_master "[${plant}] skipped: only ${image_count} images"
    continue
  fi

  total_t0=$(date +%s)

  db_path="${plant_out}/database.db"
  sparse_dir="${plant_out}/sparse"
  dense_dir="${plant_out}/dense"
  fuse_ply="${plant_out}/fuse.ply"
  fused_compat="${plant_out}/fused.ply"

  mkdir -p "${sparse_dir}"

  feature_sec=0
  match_sec=0
  mapper_sec=0
  undistort_sec=0
  patchmatch_sec=0
  fusion_sec=0
  matcher_used="sequential"
  status="success"

  # 1) Feature extraction
  run_stage "${plant_log}" "feature_extractor" \
    "${COLMAP_BIN}" feature_extractor \
      --database_path "${db_path}" \
      --image_path "${image_dir}" \
      --ImageReader.single_camera 1 \
      --SiftExtraction.use_gpu "${USE_GPU}" \
      --SiftExtraction.gpu_index 0 \
      --SiftExtraction.num_threads "${THREADS}" \
      --SiftExtraction.max_num_features 16384
  rc=$?
  feature_sec=${STAGE_ELAPSED}
  if [ "${rc}" -ne 0 ]; then
    status="failed_feature"
  fi

  # 2) Matching + 3) Mapping
  best_model=""
  if [ "${status}" = "success" ]; then
    run_stage "${plant_log}" "sequential_matcher" \
      "${COLMAP_BIN}" sequential_matcher \
        --database_path "${db_path}" \
        --SiftMatching.use_gpu "${USE_GPU}" \
        --SiftMatching.gpu_index 0 \
        --SiftMatching.guided_matching true \
        --SiftMatching.num_threads "${THREADS}" \
        --SequentialMatching.overlap "${SEQUENTIAL_OVERLAP}"
    rc=$?
    match_sec=${STAGE_ELAPSED}

    if [ "${rc}" -eq 0 ]; then
      run_stage "${plant_log}" "mapper_sequential" \
        "${COLMAP_BIN}" mapper \
          --database_path "${db_path}" \
          --image_path "${image_dir}" \
          --output_path "${sparse_dir}" \
          --Mapper.num_threads "${THREADS}"
      rc=$?
      mapper_sec=${STAGE_ELAPSED}
    fi

    if [ "${rc}" -ne 0 ]; then
      status="failed_sparse"
    else
      best_model=$(choose_best_model "${sparse_dir}")
      if [ -z "${best_model}" ]; then
        echo "[$(date '+%F %T')] No valid sparse model from sequential matcher, fallback to exhaustive." >> "${plant_log}"

        matcher_used="exhaustive"
        run_stage "${plant_log}" "exhaustive_matcher" \
          "${COLMAP_BIN}" exhaustive_matcher \
            --database_path "${db_path}" \
            --SiftMatching.use_gpu "${USE_GPU}" \
            --SiftMatching.gpu_index 0 \
            --SiftMatching.guided_matching true \
            --SiftMatching.num_threads "${THREADS}"
        rc=$?
        match_sec=$((match_sec + STAGE_ELAPSED))

        if [ "${rc}" -eq 0 ]; then
          run_stage "${plant_log}" "mapper_exhaustive" \
            "${COLMAP_BIN}" mapper \
              --database_path "${db_path}" \
              --image_path "${image_dir}" \
              --output_path "${sparse_dir}" \
              --Mapper.num_threads "${THREADS}"
          rc=$?
          mapper_sec=$((mapper_sec + STAGE_ELAPSED))
        fi

        if [ "${rc}" -ne 0 ]; then
          status="failed_sparse"
        else
          best_model=$(choose_best_model "${sparse_dir}")
          if [ -z "${best_model}" ]; then
            status="failed_sparse_no_model"
          fi
        fi
      fi
      if [ "${status}" = "success" ]; then
        echo "[$(date '+%F %T')] Selected sparse model: ${best_model}" >> "${plant_log}"
      fi
    fi
  fi

  # 4) Undistort
  if [ "${status}" = "success" ]; then
    run_stage "${plant_log}" "image_undistorter" \
      "${COLMAP_BIN}" image_undistorter \
        --image_path "${image_dir}" \
        --input_path "${best_model}" \
        --output_path "${dense_dir}" \
        --output_type COLMAP \
        --max_image_size "${MAX_IMAGE_SIZE}"
    rc=$?
    undistort_sec=${STAGE_ELAPSED}
    if [ "${rc}" -ne 0 ]; then
      status="failed_undistort"
    fi
  fi

  # 5) PatchMatch Stereo
  if [ "${status}" = "success" ]; then
    run_stage "${plant_log}" "patch_match_stereo" \
      "${COLMAP_BIN}" patch_match_stereo \
        --workspace_path "${dense_dir}" \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true \
        --PatchMatchStereo.gpu_index 0
    rc=$?
    patchmatch_sec=${STAGE_ELAPSED}
    if [ "${rc}" -ne 0 ]; then
      status="failed_patchmatch"
    fi
  fi

  # 6) Stereo fusion -> fuse.ply
  if [ "${status}" = "success" ]; then
    run_stage "${plant_log}" "stereo_fusion" \
      "${COLMAP_BIN}" stereo_fusion \
        --workspace_path "${dense_dir}" \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path "${fuse_ply}"
    rc=$?
    fusion_sec=${STAGE_ELAPSED}
    if [ "${rc}" -ne 0 ]; then
      status="failed_fusion"
    elif [ ! -s "${fuse_ply}" ]; then
      status="failed_no_fuse_ply"
    else
      ln -sfn "fuse.ply" "${fused_compat}"
    fi
  fi

  total_t1=$(date +%s)
  total_sec=$((total_t1 - total_t0))

  if [ "${status}" = "success" ]; then
    SUCCESS=$((SUCCESS + 1))
    echo "[$(date '+%F %T')] DONE success in ${total_sec}s, fuse_ply=${fuse_ply}" >> "${plant_log}"
    log_master "[${plant}] success (${total_sec}s)"
  else
    FAILED=$((FAILED + 1))
    echo "[$(date '+%F %T')] DONE ${status} in ${total_sec}s" >> "${plant_log}"
    log_master "[${plant}] ${status} (${total_sec}s)"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${plant}" "${status}" "${image_dir}" "${image_count}" "${GPU_MODE}" \
    "${total_sec}" "${feature_sec}" "${matcher_used}" "${match_sec}" "${mapper_sec}" \
    "${undistort_sec}" "${patchmatch_sec}" "${fusion_sec}" "${fuse_ply}" >> "${SUMMARY_CSV}"

done

ln -sfn "${RUN_DIR}" "${LATEST_LINK}"

log_master "All done. total=${TOTAL}, success=${SUCCESS}, failed=${FAILED}, skipped=${SKIPPED}"
log_master "Summary CSV: ${SUMMARY_CSV}"

printf '\nRun complete.\nRun dir: %s\nSummary: %s\n' "${RUN_DIR}" "${SUMMARY_CSV}"
