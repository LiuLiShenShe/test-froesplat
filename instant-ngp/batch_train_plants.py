#!/usr/bin/env python3
"""
批处理脚本：复制 RGBA 数据并使用 Instant-NGP 训练所有植株
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

def copy_plant_data(plant_name):
    """复制单个植株的 RGBA 图像和 COLMAP 数据"""
    log(f"开始复制 {plant_name} 的数据...")

    source_plant = Path(SOURCE_DIR) / plant_name
    target_plant = Path(TARGET_DIR) / plant_name

    # 创建目标目录
    target_plant.mkdir(parents=True, exist_ok=True)

    # 1. 复制 images_rgba
    source_images = source_plant / "images_rgba"
    target_images = target_plant / "images"

    if source_images.exists():
        if target_images.exists():
            shutil.rmtree(target_images)
        shutil.copytree(source_images, target_images)
        log(f"  ✓ 复制 images_rgba -> images ({len(list(target_images.glob('*.png')))} 张图像)")
    else:
        log(f"  ✗ 未找到 images_rgba 文件夹")
        return False

    # 2. 复制 sparse 文件夹
    source_sparse = source_plant / "sparse"
    target_sparse = target_plant / "sparse"

    if source_sparse.exists():
        if target_sparse.exists():
            shutil.rmtree(target_sparse)
        shutil.copytree(source_sparse, target_sparse)
        log(f"  ✓ 复制 sparse 文件夹")
    else:
        log(f"  ✗ 未找到 sparse 文件夹")
        return False

    return True

def generate_transforms_json(plant_name):
    """使用 colmap2nerf.py 生成 transforms.json"""
    log(f"生成 {plant_name} 的 transforms.json...")

    plant_dir = Path(TARGET_DIR) / plant_name
    script_path = Path(INSTANT_NGP_DIR) / "scripts" / "colmap2nerf.py"

    # 检查文件是否存在
    if not script_path.exists():
        log(f"  ✗ 未找到 colmap2nerf.py: {script_path}")
        return False

    # 运行 colmap2nerf.py
    cmd = [
        sys.executable,
        str(script_path),
        "--colmap_model", str(plant_dir / "sparse" / "0"),
        "--aabb_scale", "16",  # 适合植株大小
        "--images", str(plant_dir / "images"),
        "--out", str(plant_dir / "transforms.json")
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(plant_dir),
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            log(f"  ✓ 成功生成 transforms.json")

            # 验证文件
            transforms_file = plant_dir / "transforms.json"
            if transforms_file.exists():
                with open(transforms_file, 'r') as f:
                    data = json.load(f)
                    num_frames = len(data.get('frames', []))
                    log(f"  ✓ 包含 {num_frames} 帧")
                return True
            else:
                log(f"  ✗ transforms.json 未生成")
                return False
        else:
            log(f"  ✗ colmap2nerf.py 失败:")
            log(f"     {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        log(f"  ✗ 生成超时")
        return False
    except Exception as e:
        log(f"  ✗ 生成失败: {e}")
        return False

def train_plant(plant_name):
    """训练单个植株"""
    log(f"开始训练 {plant_name} ({TRAIN_STEPS} 步)...")

    plant_dir = Path(TARGET_DIR) / plant_name
    output_dir = plant_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # 检查 transforms.json
    transforms_file = plant_dir / "transforms.json"
    if not transforms_file.exists():
        log(f"  ✗ 未找到 transforms.json")
        return False

    # 运行训练
    script_path = Path(INSTANT_NGP_DIR) / "scripts" / "run.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--scene", str(plant_dir),
        "--n_steps", str(TRAIN_STEPS),
        "--save_snapshot", str(output_dir / f"{plant_name}_trained.msgpack")
    ]

    try:
        log(f"  训练命令: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=INSTANT_NGP_DIR,
            capture_output=False,  # 显示实时输出
            text=True,
            timeout=1800  # 30分钟超时
        )

        if result.returncode == 0:
            log(f"  ✓ 训练完成")
            return True
        else:
            log(f"  ✗ 训练失败")
            return False

    except subprocess.TimeoutExpired:
        log(f"  ✗ 训练超时")
        return False
    except Exception as e:
        log(f"  ✗ 训练失败: {e}")
        return False

def main():
    """主函数"""
    log("=" * 60)
    log("Instant-NGP 批处理训练脚本")
    log("=" * 60)
    log(f"源目录: {SOURCE_DIR}")
    log(f"目标目录: {TARGET_DIR}")
    log(f"训练步数: {TRAIN_STEPS}")
    log(f"植株数量: {len(PLANTS)}")
    log("=" * 60)

    # 创建目标目录
    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)

    # 统计
    stats = {
        "total": len(PLANTS),
        "copied": 0,
        "transformed": 0,
        "trained": 0,
        "failed": []
    }

    # 处理每个植株
    for i, plant in enumerate(PLANTS, 1):
        log(f"\n{'='*60}")
        log(f"处理植株 {i}/{len(PLANTS)}: {plant}")
        log(f"{'='*60}")

        # 1. 复制数据
        if not copy_plant_data(plant):
            stats["failed"].append((plant, "复制数据失败"))
            continue
        stats["copied"] += 1

        # 2. 生成 transforms.json
        if not generate_transforms_json(plant):
            stats["failed"].append((plant, "生成 transforms.json 失败"))
            continue
        stats["transformed"] += 1

        # 3. 训练
        if not train_plant(plant):
            stats["failed"].append((plant, "训练失败"))
            continue
        stats["trained"] += 1

    # 打印统计
    log("\n" + "=" * 60)
    log("处理完成！统计信息:")
    log("=" * 60)
    log(f"总植株数: {stats['total']}")
    log(f"复制成功: {stats['copied']}")
    log(f"生成 transforms.json 成功: {stats['transformed']}")
    log(f"训练成功: {stats['trained']}")

    if stats["failed"]:
        log(f"\n失败的植株 ({len(stats['failed'])}):")
        for plant, reason in stats["failed"]:
            log(f"  - {plant}: {reason}")

    log("=" * 60)

if __name__ == "__main__":
    main()
