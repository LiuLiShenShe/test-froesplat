#!/usr/bin/env python3
"""
将 NeRF 渲染为点云（Point Cloud）- 简化版本
用于导入 CloudCompare 等工具
"""
import os
import sys
sys.path.insert(0, '/data/fj/instant-ngp/build')
import pyngp as ngp
import numpy as np

snapshot_path = "/data/fj/10-Instant-NGP/CaoMei1_trained.msgpack"
output_dir = "/data/fj/10-Instant-NGP/CaoMei1_pointcloud"
os.makedirs(output_dir, exist_ok=True)

# 创建 testbed
testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

# 加载快照
print(f"加载快照: {snapshot_path}")
testbed.load_snapshot(snapshot_path)
print("✓ 快照加载成功")

# 渲染点云
print("正在渲染点云...")
points = testbed.get_point_cloud()

print(f"  点云形状: {points.shape}")
print(f"   X: {points.shape[0]}, Y: {points.shape[1]}, Z: {points.shape[2]}")
print(f"   总点数: {points.shape[0]}")

# 保存为多种格式

# 1. PLY 格式（CloudCompare 原生支持）
ply_path = f"{output_dir}/pointcloud.ply"
with open(ply_path, 'w') as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex\n')
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    f.write('end_header\n')
    f.write(f'element vertex {points.shape[0]}\n')
    for i in range(points.shape[0]):
        f.write(f'{points[i,0]} {points[i,1]} {points[i,2]} 255 255 255\n')
    f.write('end_header\n')

print(f"✓ PLY 格式: {ply_path}")

# 2. XYZ 格式（纯文本）
xyz_path = f"{output_dir}/pointcloud.xyz"
np.savetxt(xyz_path, points, fmt='%.6f')
print(f"✓ XYZ 格式: {xyz_path}")

# 3. TXT 格式（简单文本）
txt_path = f"{output_dir}/pointcloud.txt"
np.savetxt(txt_path, points, fmt='%.6f', header='X Y Z')
print(f"✓ TXT 格式: {txt_path}")

print("\n✓ 导出完成！")
print(f"\n📂 输出目录: {output_dir}")
print("\n💡 如何在 CloudCompare 中使用:")
print("   1. 打开 CloudCompare")
print(f"   2. 导入文件: {ply_path}")
print("   3. 调整显示: 设置点大小为 1-2 像素")
print("   4. 查看点云: 使用颜色模式")
