#!/usr/bin/env bash
# Run the remaining M2M3-GO ablation variants strictly one at a time.
#
# The host has a shared 125 GB RAM budget and a full swap file; two concurrent
# 30k trainings already triggered the kernel OOM killer once (M2 died with
# SIGKILL after 30k iterations).  Serial execution is therefore mandatory.
#
# Usage: ./run_ablation_chain.sh [driver_pid_to_wait_for]
set -u
W=/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十一_原稿方法全量实现与端到端验证
cd "$W" || exit 1

wait_pid=${1:-}
if [ -n "$wait_pid" ]; then
    echo "[chain] waiting for driver $wait_pid"
    while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
fi

for variant in M3 M4; do
    echo "[chain] $(date -Is) starting $variant"
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_m2m3_ablation.py \
        --sample WanNianQing1 --group C_R1 --variants "$variant" \
        --iterations 30000 --resolution 2 --run-name m2m3_ablation \
        --port-base 6080 > "logs/driver_${variant}_serial.log" 2>&1
    code=$?
    echo "[chain] $(date -Is) $variant driver exited $code"
    if [ "$code" -ne 0 ]; then
        echo "[chain] stopping: $variant failed, later variants not started"
        exit "$code"
    fi
done
echo "[chain] $(date -Is) all variants done"
