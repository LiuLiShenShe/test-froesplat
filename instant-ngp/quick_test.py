#!/usr/bin/env python3
"""
快速测试脚本 - 只训练100步
"""

import subprocess
import sys
from pathlib import Path

INSTANT_NGP_DIR = "/data/fj/instant-ngp"
plant = "BaiZhang"
plant_dir = Path(f"/data/fj/10-Instant-NGP/{plant}")
output_dir = plant_dir / "output"
output_dir.mkdir(exist_ok=True)

TRAIN_STEPS = 100  # 快速测试

cmd = [
    sys.executable,
    f"{INSTANT_NGP_DIR}/scripts/run.py",
    "--scene", str(plant_dir),
    "--n_steps", str(TRAIN_STEPS),
    "--save_snapshot", str(output_dir / f"{plant}_test.msgpack")
]

print(f"快速测试: {plant} ({TRAIN_STEPS} 步)")
print("="*70)

# 运行并实时显示关键输出
process = subprocess.Popen(
    cmd,
    cwd=INSTANT_NGP_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

for line in process.stdout:
    # 只显示关键信息
    if any(k in line for k in ['SUCCESS', 'ERROR', 'loss=', 'frame', 'Saving']):
        print(line.rstrip())

process.wait()

print("="*70)
if process.returncode == 0:
    print("✓ 测试成功！可以开始完整训练。")

    # 检查输出
    snapshot = output_dir / f"{plant}_test.msgpack"
    if snapshot.exists():
        print(f"✓ 快照已生成: {snapshot} ({snapshot.stat().st_size / 1024 / 1024:.2f} MB)")
else:
    print(f"✗ 测试失败: 返回码 {process.returncode}")
