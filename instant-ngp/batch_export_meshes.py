#!/usr/bin/env python3
"""
批处理导出：将所有训练好的 NeRF 模型导出为 3D mesh
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 配置
TARGET_DIR = "/data/fj/10-Instant-NGP"
INSTANT_NGP_DIR = "/data/fj/instant-ngp"
EXPORT_SCRIPT = f"{INSTANT_NGP_DIR}/export_mesh.py"

# 植株列表
PLANTS = [
    "BaiZhang", "CaoMei1", "CaoMei2", "ChangShouHua2",
    "DouBanLv1", "DouBanLv2", "DouBanLv3",
    "HongZhang", "KongQueZhuYu",
    "WangWenCao1", "WangWenCao2",
    "WanNianQing1", "WanNianQing2",
    "XiangPiShu1",
    "XianKeLai1", "XianKeLai2", "XianKeLai3"
]

def log(message):
    """打印带时间戳的日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def export_plant_mesh(plant_name, resolution=256, density_thresh=10.0):
    """导出单个植株的 3D mesh"""

    plant_dir = Path(TARGET_DIR) / plant_name
    snapshot_file = plant_dir / "output" / f"{plant_name}_trained.msgpack"
    output_mesh = plant_dir / "output" / f"{plant_name}_3d.obj"

    # 检查快照是否存在
    if not snapshot_file.exists():
        log(f"  ✗ 快照不存在: {snapshot_file}")
        return False

    # 运行导出脚本
    cmd = [
        sys.executable,
        EXPORT_SCRIPT,
        str(snapshot_file),
        "-o", str(output_mesh),
        "-r", str(resolution),
        "-d", str(density_thresh)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )

        if result.returncode == 0 and output_mesh.exists():
            file_size = output_mesh.stat().st_size / 1024 / 1024
            log(f"  ✓ 导出成功: {file_size:.2f} MB")
            return True
        else:
            log(f"  ✗ 导出失败")
            if result.stderr:
                # 只显示关键错误信息
                errors = [line for line in result.stderr.split('\n')
                         if 'error' in line.lower() or 'Error' in line]
                if errors:
                    log(f"     {errors[0][:100]}")
            return False

    except subprocess.TimeoutExpired:
        log(f"  ✗ 导出超时")
        return False
    except Exception as e:
        log(f"  ✗ 异常: {e}")
        return False

def main():
    """主函数"""
    log("=" * 70)
    log("Instant-NGP 批处理导出 - NeRF → 3D Mesh")
    log("=" * 70)
    log(f"目标目录: {TARGET_DIR}")
    log(f"植株数量: {len(PLANTS)}")
    log(f"导出分辨率: 256³")
    log("=" * 70)

    stats = {
        "total": len(PLANTS),
        "exported": 0,
        "failed": []
    }

    start_time = datetime.now()

    for i, plant in enumerate(PLANTS, 1):
        log(f"\n[{i}/{len(PLANTS)}] 导出: {plant}")

        if export_plant_mesh(plant):
            stats["exported"] += 1
        else:
            stats["failed"].append(plant)

    end_time = datetime.now()
    duration = end_time - start_time

    # 打印总结
    log("\n" + "=" * 70)
    log("导出完成！")
    log("=" * 70)
    log(f"总植株数: {stats['total']}")
    log(f"导出成功: {stats['exported']}")
    log(f"导出失败: {len(stats['failed'])}")
    log(f"总耗时: {duration}")

    if stats["failed"]:
        log(f"\n失败的植株:")
        for plant in stats["failed"]:
            log(f"  - {plant}")

    log("=" * 70)
    log("\n所有 3D 模型保存在各自的 output/ 目录下")
    log("文件格式: OBJ（可在 Blender、MeshLab、Unity 等软件中打开）")

if __name__ == "__main__":
    main()
