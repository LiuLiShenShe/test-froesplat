#!/usr/bin/env bash
set -u

ROOT="/data/fj"
SUGAR_ROOT="$ROOT/SuGaR-main"
VENV_PY="$SUGAR_ROOT/venv/bin/python"
TRAIN_PY="$SUGAR_ROOT/gaussian_splatting/train.py"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/07-SuGaR-GS/restart30000_rgba_cuda0_${TS}"
LOG_DIR="$RUN_DIR/logs"
RUN_LOG="$RUN_DIR/run.log"
STATUS_CSV="$RUN_DIR/status.csv"

mkdir -p "$LOG_DIR"

log() {
  local msg="$1"
  echo "[$(date '+%F %T')] $msg" | tee -a "$RUN_LOG"
}

check_deps() {
  log "Checking venv dependencies..."
  "$VENV_PY" - <<'PY' >>"$RUN_LOG" 2>&1
import importlib
mods=[
  "torch","torchvision","numpy","PIL","plyfile","cv2","open3d",
  "diff_gaussian_rasterization","simple_knn"
]
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

prepare_replacement_douban() {
  log "Validating replacement scene DouBanLv3 (undistorted PINHOLE + RGBA)..."
  "$VENV_PY" - <<'PY' >>"$RUN_LOG" 2>&1
import os, sys
from PIL import Image
sys.path.append('/data/fj/SuGaR-main/gaussian_splatting')
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary

root = '/data/fj/04-COLMAP/DouBanLv3_undist_pinhole_20260305b'
sp0 = os.path.join(root, 'sparse/0')
if not os.path.isdir(root):
    raise FileNotFoundError(root)
if not os.path.exists(os.path.join(sp0, 'cameras.bin')) or not os.path.exists(os.path.join(sp0, 'images.bin')):
    raise FileNotFoundError(f'Missing sparse model under {sp0}')

cams = read_intrinsics_binary(os.path.join(sp0, 'cameras.bin'))
models = sorted({c.model for c in cams.values()})
allowed = {'PINHOLE', 'SIMPLE_PINHOLE'}
if not set(models).issubset(allowed):
    raise RuntimeError(f'Unsupported camera models: {models}')

ex = read_extrinsics_binary(os.path.join(sp0, 'images.bin'))
names = sorted(os.path.basename(v.name) for v in ex.values())
if len(names) < 100:
    raise RuntimeError(f'Too few registered images: {len(names)}')

rgba_dir = os.path.join(root, 'images_rgba')
ok = 0
for n in names:
    p = os.path.join(rgba_dir, n)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with Image.open(p) as im:
        im.verify()
    ok += 1

print('replacement_ready', root, 'models=', models, 'registered=', len(names), 'rgba_ok=', ok)
PY
  if [ $? -ne 0 ]; then
    log "Failed to validate replacement scene DouBanLv3."
    return 1
  fi
  log "DouBanLv3 replacement scene is ready."
  return 0
}

train_one() {
  local plant="$1"
  local source_path="$2"
  local model_path="$ROOT/07-SuGaR-GS/${plant}/vanilla_gs_rgba_30000_restart_${TS}"
  local log_file="$LOG_DIR/${plant}.log"
  local start_ts end_ts dur status

  mkdir -p "$model_path"
  start_ts="$(date +%s)"
  log "START ${plant} | source=${source_path} | model=${model_path}"

  CUDA_VISIBLE_DEVICES=0 "$VENV_PY" "$TRAIN_PY" \
    -s "$source_path" \
    -m "$model_path" \
    -i images_rgba \
    --iterations 30000 \
    --save_iterations 7000 15000 30000 \
    --checkpoint_iterations 7000 15000 30000 \
    >"$log_file" 2>&1
  status=$?

  end_ts="$(date +%s)"
  dur="$((end_ts - start_ts))"
  log "END   ${plant} | status=${status} | duration_sec=${dur} | log=${log_file}"
  echo "${plant},${status},${dur},${source_path},${model_path},${log_file}" >>"$STATUS_CSV"

  return "$status"
}

main() {
  local overall=0
  echo "plant,status,duration_sec,source_path,model_path,log_file" >"$STATUS_CSV"
  log "Run dir: $RUN_DIR"

  check_deps || exit 2
  prepare_replacement_douban || exit 3

  train_one "CaoMei2" "/data/fj/04-COLMAP/CaoMei2" || overall=1
  train_one "DouBanLv3" "/data/fj/04-COLMAP/DouBanLv3_undist_pinhole_20260305b" || overall=1
  train_one "XianKeLai1" "/data/fj/04-COLMAP/XianKeLai1" || overall=1

  if [ "$overall" -eq 0 ]; then
    log "ALL DONE: success"
  else
    log "ALL DONE: partial failure (see $STATUS_CSV)"
  fi
  return "$overall"
}

main "$@"
