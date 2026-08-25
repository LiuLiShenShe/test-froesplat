#!/usr/bin/env python3
"""
简化的单植株训练脚本
"""

import subprocess
import sys
from pathlib import Path

INSTANT_NGP_DIR = "/data/fj/instant-ngp"
plant = "BaiZhang"
plant_dir = Path(f"/data/fj/10-Instant-NGP/{plant}")
output_dir = plant_dir / "output"
output_dir.mkdir(exist_ok=True)

TRAIN_STEPS = 30000

cmd = [
    sys.executable,
    f"{INSTANT_NGP_DIR}/scripts/run.py",
    "--scene", str(plant_dir),
    "--n_steps", str(TRAIN_STEPS),
    "--save_snapshot", str(output_dir / f"{plant}_trained.msgpack")
]

print(f"训练命令: {' '.join(cmd)}")
print(f"工作目录: {INSTANT_NGP_DIR}")
print("="*70)

# 直接运行，显示所有输出
result = subprocess.run(cmd, cwd=INSTANT_NGP_DIR)

print("="*70)
print(f"返回码: {result.returncode}")

if result.returncode == 0:
    print("✓ 训练成功！")
else:
    print("✗ 训练失败")
