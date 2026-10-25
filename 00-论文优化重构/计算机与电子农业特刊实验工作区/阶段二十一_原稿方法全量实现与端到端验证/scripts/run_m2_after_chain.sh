#!/usr/bin/env bash
# Wait for the running M3->M4 chain to finish, then run the remaining M2 variant.
#
# Serial by design: the host has a shared 125 GB RAM budget, and two concurrent
# 30k trainings previously triggered the kernel OOM killer.  The chain driver
# (run_ablation_chain.sh) already runs M3 then M4; M2 is appended here so that
# no variant ever overlaps another.
#
# Usage: ./run_m2_after_chain.sh [chain_pid]
set -u
W=/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十一_原稿方法全量实现与端到端验证
cd "$W" || exit 1

chain_pid=${1:-}
if [ -n "$chain_pid" ]; then
    echo "[m2-chain] $(date -Is) waiting for chain $chain_pid"
    while kill -0 "$chain_pid" 2>/dev/null; do sleep 30; done
else
    # No pid given: still refuse to start while another driver is alive.
    while pgrep -f "scripts/run_m2m3_ablation.py" > /dev/null 2>&1; do sleep 30; done
fi

echo "[m2-chain] $(date -Is) starting M2"
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_m2m3_ablation.py \
    --sample WanNianQing1 --group C_R1 --variants M2 \
    --iterations 30000 --resolution 2 --run-name m2m3_ablation \
    --port-base 6090 > "logs/driver_M2_serial.log" 2>&1
code=$?
echo "[m2-chain] $(date -Is) M2 driver exited $code"
exit "$code"
