#!/usr/bin/env bash
set -u

ROOT="/data/fj"
SUGAR_ROOT="$ROOT/SuGaR-main"
VENV_PY="$SUGAR_ROOT/venv/bin/python"
TRAIN_PY="$SUGAR_ROOT/gaussian_splatting/train.py"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/07-SuGaR-GS/additional30000_rgba_cuda0_${TS}"
LOG_DIR="$RUN_DIR/logs"
RUN_LOG="$RUN_DIR/run.log"
STATUS_CSV="$RUN_DIR/status.csv"

PLANT="WangWenCao1"
SOURCE="/data/fj/04-COLMAP/WangWenCao1"
MODEL="$ROOT/07-SuGaR-GS/${PLANT}/vanilla_gs_rgba_30000_restart_${TS}"
PLANT_LOG="$LOG_DIR/${PLANT}.log"

mkdir -p "$LOG_DIR" "$MODEL"

log() {
  local msg="$1"
  echo "[$(date '+%F %T')] $msg" | tee -a "$RUN_LOG"
}

check_deps() {
  log "Checking venv dependencies..."
  "$VENV_PY" - <<'PY' >>"$RUN_LOG" 2>&1
import importlib
mods=["torch","torchvision","numpy","PIL","plyfile","cv2","open3d","diff_gaussian_rasterization","simple_knn"]
for m in mods:
    importlib.import_module(m)
print("deps_ok")
PY
  if [ $? -ne 0 ]; then
    log "Dependency check failed."
    return 1
  fi
  log "Dependency check passed."
  return 0
}

check_scene() {
  log "Validating scene and RGBA images..."
  "$VENV_PY" - <<'PY' >>"$RUN_LOG" 2>&1
import os, sys
from PIL import Image
sys.path.append('/data/fj/SuGaR-main/gaussian_splatting')
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary

root='/data/fj/04-COLMAP/WangWenCao1'
sp0=os.path.join(root,'sparse/0')
cams=read_intrinsics_binary(os.path.join(sp0,'cameras.bin'))
models=sorted({c.model for c in cams.values()})
allowed={'PINHOLE','SIMPLE_PINHOLE'}
if not set(models).issubset(allowed):
    raise RuntimeError(f'Unsupported camera models: {models}')
ex=read_extrinsics_binary(os.path.join(sp0,'images.bin'))
names=sorted(os.path.basename(v.name) for v in ex.values())
rgba=os.path.join(root,'images_rgba')
ok=0
for n in names:
    p=os.path.join(rgba,n)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with Image.open(p) as im:
        im.verify()
    ok += 1
print('scene_ok', 'models=', models, 'registered=', len(names), 'rgba_ok=', ok)
PY
  if [ $? -ne 0 ]; then
    log "Scene validation failed."
    return 1
  fi
  log "Scene validation passed."
  return 0
}

main() {
  local start_ts end_ts dur status
  echo "plant,status,duration_sec,source_path,model_path,log_file" >"$STATUS_CSV"
  log "Run dir: $RUN_DIR"

  check_deps || exit 2
  check_scene || exit 3

  start_ts="$(date +%s)"
  log "START ${PLANT} | source=${SOURCE} | model=${MODEL}"

  CUDA_VISIBLE_DEVICES=0 "$VENV_PY" "$TRAIN_PY" \
    -s "$SOURCE" \
    -m "$MODEL" \
    -i images_rgba \
    --iterations 30000 \
    --save_iterations 7000 15000 30000 \
    --checkpoint_iterations 7000 15000 30000 \
    >"$PLANT_LOG" 2>&1
  status=$?

  end_ts="$(date +%s)"
  dur="$((end_ts - start_ts))"
  log "END   ${PLANT} | status=${status} | duration_sec=${dur} | log=${PLANT_LOG}"
  echo "${PLANT},${status},${dur},${SOURCE},${MODEL},${PLANT_LOG}" >>"$STATUS_CSV"

  if [ "$status" -eq 0 ]; then
    log "ALL DONE: success"
  else
    log "ALL DONE: failed"
  fi
  return "$status"
}

main "$@"
