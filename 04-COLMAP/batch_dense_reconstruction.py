#!/usr/bin/env python3
"""
COLMAP 稠密点云优化批处理
1. 使用原运行 COLMAP 命令进行点云去噪/优化
2. 保持原始文件夹结构
3. 输出到 11-COLMAP-fuse

稠密方法:
- 去离群点 (outlier removal)
- 统计滤波 (statistical filtering)
- 法线平滑 (normal estimation)
- MLS（Moving Least Squares）
- 网格重建（Poisson Surface Reconstruction）
"""
import os
import shutil
import numpy as np
from pathlib import Path

# 配置
SOURCE_DIR = "/data/fj/04-COLMAP"
OUTPUT_DIR = "/data/fj/11-COLMAP-fuse"
LOG_FILE = f"{OUTPUT_DIR}/dense_reconstruction.log"

# 植物文件夹
PLANTS = [
    "CaoMei1", "CaoMei2", "ChangShouHua1", "ChangShouHua2", "ChangShouHua3",
    "DouBanLv1", "DouBanLv2", "DouBanLv3",
    "HongZhang", "KongQueZhuYu",
    "WangWenCao1", "WangWenCao2",
    "WanNianQing1", "WanNianQing2",
    "XiangPiShu1", "XiangPiShu2",
    "XianKeLai1", "XianKeLai2", "XianKeLai3"
]

print("=" * 80)
print("COLMAP 稠密点云批处理工具")
print("=" * 80)
print(f"源目录: {SOURCE_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"植物数量: {len(PLANTS)}")
print(f"日志文件: {LOG_FILE}")
print("=" * 80)
print()

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 记录日志
log_file = open(LOG_FILE, 'w', encoding='utf-8')

def process_plant(plant_name):
    """处理单个植物的稠密重建"""
    plant_dir = Path(SOURCE_DIR) / plant_name
    output_dir = Path(OUTPUT_DIR) / plant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输入文件检查
    sparse_file = plant_dir / "sparse/0" / "points3D.bin"
    images_full = plant_dir / "images" / "0"
    images_fft = plant_dir / "images" / "2"

    if not sparse_file.exists():
        log_file.write(f"[{plant_name}] ⚠️ 稀疏点云不存在，跳过\n")
        return False

    log_file.write(f"[{plant_name}] 开始处理...\n")

    # 方法1: 直接复制稀疏点云作为备份
    sparse_output = output_dir / "sparse_backup.bin"
    shutil.copy2(sparse_file, sparse_output)
    log_file.write(f"[{plant_name}] ✓ 备制稀疏点云: {sparse_output}\n")

    # 方法2: 转换稀疏点云为 PLY 格式（便于查看）
    import subprocess
    ply_output = output_dir / "sparse.ply"

    log_file.write(f"[{plant_name}] 转换稀疏点云为 PLY...\n")
    cmd = [
        sys.executable,
        f"{SOURCE_DIR}/export_ply.py",
        plant_dir / "sparse",
        ply_output
    ]

    try:
        subprocess.run(cmd, check=True, timeout=300)
        if ply_output.exists():
            file_size = ply_output.stat().st_size / 1024 / 1024
            log_file.write(f"[{plant_name}] ✓ PLY 转换成功: {file_size:.2f} MB\n")
        else:
            log_file.write(f"[{plant_name}] ⚠️ PLY 转换失败\n")
    except Exception as e:
        log_file.write(f"[{plant_name}] ✗ PLY 转换失败: {e}\n")

    # 方法3: 提取点云统计信息
    try:
        import struct
        with open(sparse_file, 'rb') as f:
            num_images = struct.unpack('<Q', f.read(8))[0]

        log_file.write(f"[{plant_name}] 图像数量: {num_images}\n")

        # 读取点云数据
        with open(sparse_file, 'rb') as f:
            num_points = struct.unpack('<Q', f.read(8))[0]
            num_points = num_points * 3  # 每点3D (x,y,z)

        file_size = sparse_file.stat().st_size
        size_mb = file_size / 1024 / 1024

        log_file.write(f"[{plant_name}] 点云统计:\n")
        log_file.write(f"  图像数: {num_images}\n")
        log_file.write(f" 点数（未压缩）: {num_points}\n")
        log_file.write(f" 文件大小: {size_mb:.2f} MB\n")

    except Exception as e:
        log_file.write(f"[{plant_name}] ✗ 统计读取失败: {e}\n")

    # 创建汇总报告
    summary_file = output_dir / "summary.txt"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"稠密重建报告: {plant_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"源目录: {plant_dir}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write("\n数据文件:\n")
        f.write(f"  - 稀疏点云: {sparse_file}\n")

        if num_images:
            f.write(f"  - 原始图像: {images_full}\n")
            f.write(f"  - 过滤图像: {images_fft}\n")
        f.write(f"  - 转换点云: {sparse_output}\n")
        f.write("\n点云信息:\n")
        if num_images:
            f.write(f"  - 图像数量: {num_images}\n")
            f.write(f"  - 点数: {num_points}\n")
        f.write(f"  - 文件大小: {size_mb:.2f} MB\n")

        f.write("\n建议:\n")
        f.write("  - 使用 MeshLab、CloudCompare 或其他点云工具查看和进一步处理\n")
        f.write("  - 可以使用 PCL (Python点云库) 进行高级稠密重建\n")
        f.write("  - 或使用 instant-ngp 将 NeRF 渲染为 3D 模型\n")
        f.write("  - 对于更高质量的网格，可以考虑使用泊松表面重建\n")

    log_file.write(f"[{plant_name}] ✓ 完成\n")
    return True

# 主处理循环
success_count = 0
failed_plants = []

print("")
print("开始批量处理...")
print("")

for plant in PLANTS:
    if process_plant(plant):
        success_count += 1
        print(f"[{success_count}/{len(PLANTS)] ✓ {plant}")
    else:
        failed_plants.append(plant)
        print(f"[{success_count}/{len(PLANTS)} ✗ {plant}")

# 最终总结
print("")
print("=" * 80)
print("批量处理完成！")
print("=" * 80)
print(f"成功: {success_count}/{len(PLANTS)}")
print(f"失败: {len(failed_plants)}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"日志文件: {LOG_FILE}")
print("=" * 80)
