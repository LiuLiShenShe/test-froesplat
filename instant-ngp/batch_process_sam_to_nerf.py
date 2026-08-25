#!/usr/bin/env python3
"""
批量处理 SAM 分割图像，转换为 NeRF 数据集
- 将黑色背景转换为透明 (RGBA)
- 运行 COLMAP 计算相机位姿
- 生成 transforms.json
"""

import os
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# 配置
SOURCE_DIR = "/data/fj/03-SAM"
OUTPUT_DIR = "/data/fj/10-Instant-NGP"
INSTANT_NGP_DIR = "/data/fj/instant-ngp"
AABB_SCALE = 32

# 黑色背景阈值（低于此值视为黑色）
BLACK_THRESHOLD = 10  # 0-255


def convert_to_rgba_with_transparent_background(input_path, output_path):
    """
    将 PNG 转换为 RGBA，黑色背景设为透明
    """
    # 读取图像
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"⚠️ 无法读取图像: {input_path}")
        return False

    # 检查图像维度
    if len(img.shape) < 2:
        print(f"⚠️ 图像格式异常: {input_path}")
        return False

    # 如果已经是 RGBA，直接使用
    if len(img.shape) == 3 and img.shape[2] == 4:
        rgba = img
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # 保持 BGR 顺序，只添加 alpha 通道
        rgba = np.dstack((img, np.full(img.shape[:2], 255, dtype=np.uint8)))
    else:
        print(f"⚠️ 不支持的图像格式 (shape={img.shape}): {input_path}")
        return False

    # 检测黑色背景（BGR 都很低）
    # 创建掩码：非黑色区域
    bgr = rgba[:, :, :3]
    black_mask = np.all(bgr < BLACK_THRESHOLD, axis=2)

    # 设置 alpha 通道：黑色区域为 0（透明），其他为 255（不透明）
    rgba[:, :, 3] = np.where(black_mask, 0, 255)

    # 保存为 PNG
    cv2.imwrite(str(output_path), rgba)

    return True


def process_folder(folder_name):
    """
    处理单个植物文件夹
    """
    print(f"\n{'='*60}")
    print(f"处理: {folder_name}")
    print(f"{'='*60}")

    source_folder = Path(SOURCE_DIR) / folder_name
    output_folder = Path(OUTPUT_DIR) / folder_name

    # 创建输出目录
    output_folder.mkdir(parents=True, exist_ok=True)

    # 创建 images 子目录
    images_dir = output_folder / "images"
    images_dir.mkdir(exist_ok=True)

    # 步骤 1: 转换所有图像为 RGBA
    print(f"\n步骤 1: 转换图像为 RGBA（黑色背景透明）")
    png_files = sorted(source_folder.glob("*.png"))

    if not png_files:
        print(f"⚠️ 未找到 PNG 文件: {source_folder}")
        return False

    success_count = 0
    for png_file in tqdm(png_files, desc="转换图像"):
        output_path = images_dir / png_file.name
        if convert_to_rgba_with_transparent_background(png_file, output_path):
            success_count += 1

    print(f"✓ 成功转换 {success_count}/{len(png_files)} 张图像")

    # 步骤 2: 运行 colmap2nerf.py
    print(f"\n步骤 2: 运行 COLMAP 生成 transforms.json")

    # 切换到输出目录
    original_dir = os.getcwd()
    os.chdir(output_folder)

    # 构建命令
    colmap_script = Path(INSTANT_NGP_DIR) / "scripts" / "colmap2nerf.py"
    cmd = (
        f"python {colmap_script} "
        f"--images images "
        f"--run_colmap "
        f"--colmap_matcher sequential "
        f"--aabb_scale {AABB_SCALE} "
        f"--overwrite"
    )

    print(f"运行命令: {cmd}")
    ret = os.system(cmd)

    os.chdir(original_dir)

    if ret != 0:
        print(f"⚠️ COLMAP 运行失败: {folder_name}")
        return False

    print(f"✓ 完成: {folder_name}")
    return True


def main():
    """
    主函数：处理所有文件夹
    """
    print("="*60)
    print("批量 SAM 图像 → NeRF 数据集转换")
    print("="*60)
    print(f"源目录: {SOURCE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"AABB Scale: {AABB_SCALE}")
    print(f"黑色阈值: {BLACK_THRESHOLD}")

    # 创建输出主目录
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 获取所有子文件夹
    source_path = Path(SOURCE_DIR)
    folders = sorted([d for d in source_path.iterdir() if d.is_dir()])

    print(f"\n找到 {len(folders)} 个文件夹:")
    for folder in folders:
        print(f"  - {folder.name}")

    # 处理每个文件夹
    success_count = 0
    failed_folders = []

    for folder in folders:
        try:
            if process_folder(folder.name):
                success_count += 1
            else:
                failed_folders.append(folder.name)
        except Exception as e:
            print(f"❌ 处理失败 {folder.name}: {e}")
            failed_folders.append(folder.name)

    # 总结
    print("\n" + "="*60)
    print("处理完成")
    print("="*60)
    print(f"✓ 成功: {success_count}/{len(folders)}")
    if failed_folders:
        print(f"✗ 失败: {len(failed_folders)}")
        for folder in failed_folders:
            print(f"  - {folder}")

    print(f"\n输出位置: {OUTPUT_DIR}")
    print("\n后续步骤:")
    print("1. 检查 transforms.json 文件")
    print("2. 训练 NeRF:")
    print(f"   ./instant-ngp {OUTPUT_DIR}/CaoMei1")


if __name__ == "__main__":
    main()
