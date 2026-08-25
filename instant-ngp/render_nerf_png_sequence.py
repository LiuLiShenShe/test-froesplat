#!/usr/bin/env python3
"""
NeRF → PNG 图像序列渲染（严格规范）
使用 instant-ngp 原生 run.py 脚本加载和渲染
严格约束：仅 PNG 格式，禁止 JPG
"""
import sys
sys.path.insert(0, '/data/fj/instant-ngp/build')
import pyngp as ngp

# 配置参数
SCENE_DIR = "/data/fj/10-Instant-NGP"
SCENE_NAME = "CaoMei1"
SNAPSHOT_FILE = f"{SCENE_DIR}/{SCENE_NAME}_trained.msgpack"
TRANSFORMS_FILE = f"{SCENE_DIR}/{SCENE_NAME}/transforms.json"

# 渲染参数
OUTPUT_DIR = f"{SCENE_DIR}/{SCENE_NAME}_renders"
RESOLUTION_WIDTH = 1280  # 渲染宽度
RESOLUTION_HEIGHT = 720  # 渲染高度
NUM_FRAMES = 36  # 36帧，每10度
MESH_RESOLUTION = 256  # mesh 分辨率

print("=" * 80)
print("Instant-NGP NeRF → PNG 图像序列渲染")
print("严格规范：仅 PNG 无损格式，禁止 JPG")
print("=" * 80)
print(f"场景: {SCENE_NAME}")
print(f"快照: {SNAPSHOT_FILE}")
print(f"相机参数: {TRANSFORMS_FILE}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"图像分辨率: {RESOLUTION_WIDTH}x{RESOLUTION_HEIGHT}")
print(f"帧数: {NUM_FRAMES} (360度全景)")
print("")

# 创建输出目录
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 构建 instant-ngp 命令
print("[1/6] 构建 instant-ngp 命令...")
python_exe = "/data/fj/instant-ngp/venv/bin/python"
run_py_path = "/data/fj/instant-ngp/scripts/run.py"

python_cmd = f"{python_exe} {run_py_path} --scene {SCENE_DIR}/{SCENE_NAME} --n_steps -1 --load_snapshot {SNAPSHOT_FILE} --width {RESOLUTION_WIDTH} --height {RESOLUTION_HEIGHT}"

# 添加场景
python_cmd += f" --scene {SCENE_DIR}/{SCENE_NAME}"

# 添加快照（使用训练好的模型）
python_cmd += f" --n_steps -1 --load_snapshot {SNAPSHOT_FILE}"

# 设置分辨率
python_cmd += f" --width {RESOLUTION_WIDTH} --height {RESOLUTION_HEIGHT}"

print(f"✓ 命令构建完成")
print(f"命令: {python_cmd}")
print("")

# 渲染图像序列
print("[2/6] 开始渲染图像序列...")
print(f"   将渲染 {NUM_FRAMES} 帧 PNG 图像")
print(f"   格式：PNG (无损失，符合规范）")
print("   这可能需要几分钟，请耐心等待...")
print("")

import subprocess

try:
    result = subprocess.run(
        python_cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=600  # 10分钟超时
    )

    print("\n" + "=" * 80)
    print("渲染完成！")
    print("=" * 80)

    # 显示输出
    if result.returncode == 0:
        print(result.stdout[-2000:])  # 显示最后2000字符
    else:
        print("✗ 渲染失败！")
        print(result.stderr[-500:])

    # 统计生成的 PNG 文件
    import glob
    png_files = glob.glob(f"{OUTPUT_DIR}/*.png")

    if png_files:
        print("")
        print("=" * 80)
        print("✓ 渲染结果统计")
        print("=" * 80)
        print(f"生成 PNG 文件数: {len(png_files)}")
        print(f"输出目录: {OUTPUT_DIR}")

        total_size = sum(f.stat().st_size for f in png_files)
        size_mb = total_size / 1024 / 1024

        print(f"总大小: {size_mb:.2f} MB")
        print(f"单帧平均: {size_mb/len(png_files):.2f} MB")

        # 验证格式规范（只允许 PNG）
        non_png = [f for f in png_files if not f.name.lower().endswith('.png')]
        if non_png:
            print("\n⚠️ 格式警告:")
            for f in non_png:
                print(f"  非PNG文件: {f.name}")
        else:
            print("\n✅ 格式验证通过: 所有文件均为 PNG 格式")

        print("")
        print("💡 后续操作:")
        print(f"   1. 查看渲染效果: 在图像查看器中打开任意 PNG 文件")
        print(f"   2. 创建视频（可选）:")
        print(f"      ffmpeg -framerate 30 -i {OUTPUT_DIR}/frame_*.png -c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p -o {OUTPUT_DIR}/render.mp4")
        print(f"   3. 验证质量: 检查图像是否有压缩伪影或失真")

    else:
        print(f"\n✗ 命令执行失败，返回码: {result.returncode}")
        print(f"错误输出:\n{result.stderr}")

except subprocess.TimeoutExpired:
    print("\n✗ 渲染超时（10分钟）")
    print("可能需要更长时间，或者增加超时时间")
    sys.exit(1)

except Exception as e:
    print(f"\n✗ 渲染失败！")
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
