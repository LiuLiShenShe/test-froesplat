#!/usr/bin/env python3
"""
批量为已转换的图像运行 COLMAP
- 使用已有的 images/ 文件夹
- 运行 COLMAP 生成 transforms.json
"""

import os
import sys
from pathlib import Path
import subprocess

# 配置
OUTPUT_DIR = "/data/fj/10-Instant-NGP"
INSTANT_NGP_DIR = "/data/fj/instant-ngp"
AABB_SCALE = 32


def run_colmap_for_folder(folder_name):
    """
    为单个文件夹运行 COLMAP
    """
    print(f"\n{'='*60}")
    print(f"处理: {folder_name}")
    print(f"{'='*60}")

    folder_path = Path(OUTPUT_DIR) / folder_name
    images_dir = folder_path / "images"

    # 检查 images 文件夹是否存在
    if not images_dir.exists():
        print(f"⚠️ images 文件夹不存在: {images_dir}")
        return False

    # 检查是否已有 transforms.json
    transforms_file = folder_path / "transforms.json"
    if transforms_file.exists():
        print(f"✓ transforms.json 已存在，跳过")
        return True

    # 统计图像数量
    png_count = len(list(images_dir.glob("*.png")))
    print(f"找到 {png_count} 张 PNG 图像")

    # 切换到输出目录
    original_dir = os.getcwd()
    os.chdir(folder_path)

    # 构建 COLMAP 命令
    colmap_script = Path(INSTANT_NGP_DIR) / "scripts" / "colmap2nerf.py"
    cmd = [
        "python", str(colmap_script),
        "--images", "images",
        "--run_colmap",
        "--colmap_matcher", "sequential",
        "--aabb_scale", str(AABB_SCALE),
        "--overwrite"
    ]

    print(f"\n运行命令: {' '.join(cmd)}")
    print("-" * 60)

    # 设置环境变量（headless模式）
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    # 运行命令，实时显示输出
    try:
        result = subprocess.run(
            cmd,
            cwd=folder_path,
            capture_output=False,
            text=True,
            env=env
        )

        os.chdir(original_dir)

        if result.returncode != 0:
            print(f"⚠️ COLMAP 运行失败: {folder_name}")
            return False

        # 检查是否生成了 transforms.json
        if transforms_file.exists():
            print(f"✓ 完成: {folder_name}")
            return True
        else:
            print(f"⚠️ transforms.json 未生成: {folder_name}")
            return False

    except Exception as e:
        os.chdir(original_dir)
        print(f"❌ 处理失败 {folder_name}: {e}")
        return False


def main():
    """
    主函数：处理所有文件夹
    """
    print("="*60)
    print("批量运行 COLMAP 生成 transforms.json")
    print("="*60)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"AABB Scale: {AABB_SCALE}")

    # 获取所有子文件夹
    output_path = Path(OUTPUT_DIR)
    folders = sorted([d for d in output_path.iterdir() if d.is_dir()])

    print(f"\n找到 {len(folders)} 个文件夹:")
    for folder in folders:
        print(f"  - {folder.name}")

    # 处理每个文件夹
    success_count = 0
    failed_folders = []

    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] 处理: {folder.name}")
        try:
            if run_colmap_for_folder(folder.name):
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
    print("1. 检查 transforms.json 文件:")
    print(f"   ls {OUTPUT_DIR}/*/transforms.json")
    print("2. 训练 NeRF:")
    print(f"   ./instant-ngp {OUTPUT_DIR}/CaoMei1")


if __name__ == "__main__":
    main()
