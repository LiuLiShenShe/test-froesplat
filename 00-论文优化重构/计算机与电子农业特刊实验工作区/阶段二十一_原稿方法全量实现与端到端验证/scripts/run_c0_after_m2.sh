#!/usr/bin/env bash
# Run the full-scene 2DGS control (C0) once the M2 variant has finished.
#
# C0 is the "no VFM-prior injection" reference for Table 6: identical data,
# resolution, iteration budget and evaluator, but without foreground
# initialisation, foreground RGB supervision, mask/opacity losses or view
# weighting.  It runs serially after M2 because the host cannot hold two 30k
# trainings at once (kernel OOM at 00:45 on 2026-10-25).
#
# Usage: ./run_c0_after_m2.sh [m2_driver_pid]
set -u
W=/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十一_原稿方法全量实现与端到端验证
cd "$W" || exit 1

wait_pid=${1:-}
if [ -n "$wait_pid" ]; then
    echo "[c0-chain] $(date -Is) waiting for M2 driver $wait_pid"
    while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
else
    while pgrep -f "scripts/run_m2m3_ablation.py" > /dev/null 2>&1; do sleep 30; done
fi

echo "[c0-chain] $(date -Is) starting C0 (full-scene 2DGS control)"
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_m2m3_ablation.py \
    --sample WanNianQing1 --group C_R1 --variants C0 \
    --iterations 30000 --resolution 2 --run-name m2m3_ablation \
    --port-base 6100 > "logs/driver_C0_serial.log" 2>&1
code=$?
echo "[c0-chain] $(date -Is) C0 driver exited $code"
exit "$code"
