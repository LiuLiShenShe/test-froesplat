#!/usr/bin/env python3
"""
完整的批处理脚本：处理所有植株数据并训练
"""

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime

# 配置
SOURCE_DIR = "/data/fj/04-COLMAP"
TARGET_DIR = "/data/fj/10-Instant-NGP"
INSTANT_NGP_DIR = "/data/fj/instant-ngp"
TRAIN_STEPS = 30000

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

def copy_plant_data(plant_name):
    """复制单个植株的 RGBA 图像和 COLMAP 数据"""
    log(f"复制 {plant_name} 的数据...")

    source_plant = Path(SOURCE_DIR) / plant_name
    target_plant = Path(TARGET_DIR) / plant_name

    # 创建目标目录
    target_plant.mkdir(parents=True, exist_ok=True)

    # 1. 复制 images_rgba -> images
    source_images = source_plant / "images_rgba"
    target_images = target_plant / "images"

    if not source_images.exists():
        log(f"  ✗ 未找到 images_rgba")
        return False

    if target_images.exists():
        shutil.rmtree(target_images)
    shutil.copytree(source_images, target_images)
    num_images = len(list(target_images.glob('*.png')))
    log(f"  ✓ 复制图像: {num_images} 张")

    # 2. 复制 sparse
    source_sparse = source_plant / "sparse"
    target_sparse = target_plant / "sparse"

    if not source_sparse.exists():
        log(f"  ✗ 未找到 sparse")
        return False

    if target_sparse.exists():
        shutil.rmtree(target_sparse)
    shutil.copytree(source_sparse, target_sparse)
    log(f"  ✓ 复制 COLMAP 数据")

    return True

def generate_transforms(plant_name):
    """生成 transforms.json"""
    log(f"生成 {plant_name} 的 transforms.json...")

    plant_dir = Path(TARGET_DIR) / plant_name
    sparse_dir = plant_dir / "sparse" / "0"
    images_dir = plant_dir / "images"
    output_file = plant_dir / "transforms.json"

    script_path = Path(INSTANT_NGP_DIR) / "pycolmap_to_nerf.py"

    cmd = [
        sys.executable,
        str(script_path),
        str(sparse_dir),
        str(images_dir),
        str(output_file),
        "16"  # aabb_scale
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0 and output_file.exists():
            with open(output_file) as f:
                data = json.load(f)
                num_frames = len(data.get('frames', []))
            log(f"  ✓ 生成成功: {num_frames} 帧")
            return True
        else:
            log(f"  ✗ 生成失败")
            if result.stderr:
                log(f"     {result.stderr[:200]}")
            return False

    except Exception as e:
        log(f"  ✗ 异常: {e}")
        return False

def train_plant(plant_name):
    """训练单个植株"""
    log(f"训练 {plant_name} ({TRAIN_STEPS} 步)...")

    plant_dir = Path(TARGET_DIR) / plant_name
    output_dir = plant_dir / "output"
    output_dir.mkdir(exist_ok=True)

    transforms_file = plant_dir / "transforms.json"
    if not transforms_file.exists():
        log(f"  ✗ 未找到 transforms.json")
        return False

    script_path = Path(INSTANT_NGP_DIR) / "scripts" / "run.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--scene", str(plant_dir),
        "--n_steps", str(TRAIN_STEPS),
        "--save_snapshot", str(output_dir / f"{plant_name}_trained.msgpack")
    ]

    log_file = output_dir / "training.log"

    try:
        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                cwd=INSTANT_NGP_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # 实时写入日志并显示关键信息
            for line in process.stdout:
                log_f.write(line)
                log_f.flush()
                # 只显示关键信息
                if any(k in line for k in ['SUCCESS', 'ERROR', 'loss=', 'Training']):
                    print(f"  {line.rstrip()}")
                    sys.stdout.flush()

            process.wait()

        if process.returncode == 0:
            log(f"  ✓ 训练完成")
            log(f"  日志: {log_file}")
            return True
        else:
            log(f"  ✗ 训练失败: 返回码 {process.returncode}")
            return False

    except Exception as e:
        log(f"  ✗ 异常: {e}")
        return False

def main():
    """主函数"""
    log("=" * 70)
    log("Instant-NGP 批处理训练 - 植株数据")
    log("=" * 70)
    log(f"源目录: {SOURCE_DIR}")
    log(f"目标目录: {TARGET_DIR}")
    log(f"训练步数: {TRAIN_STEPS}")
    log(f"植株数量: {len(PLANTS)}")
    log("=" * 70)

    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)

    stats = {
        "total": len(PLANTS),
        "copied": 0,
        "transformed": 0,
        "trained": 0,
        "failed": []
    }

    start_time = datetime.now()

    for i, plant in enumerate(PLANTS, 1):
        log(f"\n{'='*70}")
        log(f"[{i}/{len(PLANTS)}] 处理植株: {plant}")
        log(f"{'='*70}")

        # 1. 复制数据
        if not copy_plant_data(plant):
            stats["failed"].append((plant, "复制失败"))
            continue
        stats["copied"] += 1

        # 2. 生成 transforms.json
        if not generate_transforms(plant):
            stats["failed"].append((plant, "生成 transforms 失败"))
            continue
        stats["transformed"] += 1

        # 3. 训练
        if not train_plant(plant):
            stats["failed"].append((plant, "训练失败"))
            continue
        stats["trained"] += 1

    end_time = datetime.now()
    duration = end_time - start_time

    # 打印总结
    log("\n" + "=" * 70)
    log("批处理完成！")
    log("=" * 70)
    log(f"总植株数: {stats['total']}")
    log(f"数据复制成功: {stats['copied']}")
    log(f"Transforms 生成成功: {stats['transformed']}")
    log(f"训练成功: {stats['trained']}")
    log(f"总耗时: {duration}")

    if stats["failed"]:
        log(f"\n失败的植株 ({len(stats['failed'])}):")
        for plant, reason in stats["failed"]:
            log(f"  - {plant}: {reason}")

    log("=" * 70)

    # 保存报告
    report_file = Path(TARGET_DIR) / "batch_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            "timestamp": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "stats": stats
        }, f, indent=2)

    log(f"报告已保存: {report_file}")

if __name__ == "__main__":
    main()
