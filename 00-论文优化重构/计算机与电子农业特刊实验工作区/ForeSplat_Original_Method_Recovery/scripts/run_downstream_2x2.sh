#!/usr/bin/env bash
# Serial 2x2 downstream runs for the recovery experiment.
#
#   1 C_R1_baseline    frozen C_R1 masks,  M2M3-GO off   (done by hand, 2026-10-25)
#   2 RAP_FULL         full RAP 210 masks, M2M3-GO off
#   3 RAP_FULL_M2M3GO  full RAP 210 masks, M2M3-GO full (quantile tau_p)
#   4 C_R1_M2M3GO      frozen C_R1 masks,  M2M3-GO full
#
# Serial on purpose: two concurrent 30k trainings on this host trigger the kernel
# OOM killer (see 07_M2M3_GO_INTEGRATION.md section 4).  Each training uses a
# distinct network_gui port so a stale viewer socket cannot block the next run.
#
# Usage:
#   scripts/run_downstream_2x2.sh                       # runs 2,3,4
#   scripts/run_downstream_2x2.sh --tags RAP_FULL_M2M3GO,C_R1_M2M3GO
#   scripts/run_downstream_2x2.sh --wait-masks          # block until 210/210
#   scripts/run_downstream_2x2.sh --after-pid 1234      # wait for another run
#
# NOTE: run_downstream_2dgs.py appends /最终掩膜 itself, so RAP_RUN below is the
# RUN directory, not the mask sub-directory (passing the sub-directory makes all
# 210 masks look missing -- this happened once and is why the two variables are
# separate).
set -euo pipefail

PY=/home/test/.venvs/sam3/bin/python
ROOT=/data/fj/F2DMAS
S="$ROOT/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
RAP_RUN="$S/runs/KongQueZhuYu_RAP_FULL_210"
RAP_MASKS="$RAP_RUN/最终掩膜"
C_R1_RUN="$ROOT/00-论文优化重构/数据管理/03-分割Mask/04-fsam3_masks/KongQueZhuYu"
LOG="$S/logs/downstream_2x2.log"

TAGS="RAP_FULL,RAP_FULL_M2M3GO,C_R1_M2M3GO"
WAIT_MASKS=0
AFTER_PID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tags)       TAGS="$2"; shift 2 ;;
    --wait-masks) WAIT_MASKS=1; shift ;;
    --after-pid)  AFTER_PID="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$AFTER_PID" ]]; then
  echo "[wait] waiting for pid $AFTER_PID to exit" | tee -a "$LOG"
  while kill -0 "$AFTER_PID" 2>/dev/null; do sleep 30; done
  echo "[wait] pid $AFTER_PID gone" | tee -a "$LOG"
fi

if [[ $WAIT_MASKS -eq 1 ]]; then
  echo "[wait] waiting for 210 final masks in $RAP_MASKS" | tee -a "$LOG"
  for _ in $(seq 1 480); do          # 480 * 60 s = 8 h ceiling
    n=$(find "$RAP_MASKS" -name 'mask_*.png' | wc -l)
    if [[ $n -ge 210 ]]; then
      echo "[wait] masks ready: $n" | tee -a "$LOG"
      break
    fi
    if ! pgrep -f '生成RAP-FSAM3掩膜.py' > /dev/null; then
      echo "[wait] ABORT: mask generator is gone but only $n/210 masks exist" | tee -a "$LOG"
      exit 1
    fi
    sleep 60
  done
  n=$(find "$RAP_MASKS" -name 'mask_*.png' | wc -l)
  if [[ $n -lt 210 ]]; then
    echo "[wait] ABORT: timed out at $n/210" | tee -a "$LOG"
    exit 1
  fi
fi

run() {
  local tag="$1" masks="$2" port="$3"; shift 3
  echo "=== $(date -Is) $tag" | tee -a "$LOG"
  "$PY" "$S/scripts/run_downstream_2dgs.py" --scene KongQueZhuYu \
    --mask-run-dir "$masks" --tag "$tag" \
    --iterations 30000 --resolution 2 --port "$port" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date -Is) $tag done" | tee -a "$LOG"
}

IFS=',' read -r -a wanted <<< "$TAGS"
for tag in "${wanted[@]}"; do
  case "$tag" in
    RAP_FULL)        run RAP_FULL        "$RAP_RUN"  6040 ;;
    RAP_FULL_M2M3GO) run RAP_FULL_M2M3GO "$RAP_RUN"  6050 --m2m3-go ;;
    C_R1_M2M3GO)     run C_R1_M2M3GO     "$C_R1_RUN" 6060 --m2m3-go ;;
    *) echo "unknown tag: $tag" >&2; exit 2 ;;
  esac
done

echo "=== $(date -Is) requested tags complete: $TAGS" | tee -a "$LOG"
