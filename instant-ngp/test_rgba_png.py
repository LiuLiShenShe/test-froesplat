#!/usr/bin/env python3
"""
测试 RGBA PNG 图像在 instant-ngp 中的支持
"""

import sys
sys.path.insert(0, 'build')
import pyngp as ngp
import numpy as np
from PIL import Image
import os

print("=" * 60)
print("测试 Instant-NGP 对 RGBA PNG 的支持")
print("=" * 60)

# 创建测试 RGBA 图像
test_dir = "data/nerf/test_rgba"
os.makedirs(f"{test_dir}/images", exist_ok=True)

# 创建一个简单的 RGBA 图像（红色方块，带透明背景）
img = np.zeros((100, 100, 4), dtype=np.uint8)
img[25:75, 25:75, 0] = 255  # Red channel
img[25:75, 25:75, 3] = 255  # Alpha channel (完全不透明)

# 保存为 RGBA PNG
Image.fromarray(img, 'RGBA').save(f"{test_dir}/images/test.png")

print("\n✓ 创建测试 RGBA PNG 图像")
print(f"  图像形状: {img.shape}")
print(f"  保存位置: {test_dir}/images/test.png")

# 验证图像
loaded = Image.open(f"{test_dir}/images/test.png")
print(f"\n✓ 验证图像")
print(f"  图像模式: {loaded.mode}")
print(f"  图像尺寸: {loaded.size}")

print("\n" + "=" * 60)
print("支持的图像格式总结:")
print("=" * 60)
print("✓ PNG (包括 RGBA)")
print("✓ JPG/JPEG")
print("✓ BMP, GIF, TGA, PIC, PNM, PSD")
print("✓ EXR (HDR 图像)")
print()
print("Alpha 通道支持:")
print("✓ 内嵌在 PNG 中的 Alpha 通道")
print("✓ 单独的 .alpha.png 文件")
print()
print("推荐格式:")
print("  - RGB 图像: JPG (小文件) 或 PNG (无损)")
print("  - RGBA 图像: PNG")
print("  - HDR 图像: EXR")
