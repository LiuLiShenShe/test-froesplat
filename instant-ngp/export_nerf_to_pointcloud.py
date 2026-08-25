#!/usr/bin/env python3
"""
将 NeRF 渲染为点云（Point Cloud）
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

# 渲染设置（不设置相机矩阵，使用默认视角）
testbed.nerf.training_step = testbed.training_step  # 使用训练好的模型
testbed.nerf.render_min_depth = 0.01  # 最小渲染深度
testbed.nerf.render_max_depth = 100  # 最大渲染深度

# 渲染网格点云
print("正在渲染点云...")
points = testbed.get_point_cloud()

# 保存为 PLY 格式
ply_path = f"{output_dir}/pointcloud.ply"
save_ply(ply_path, points, testbed.nerf.camera.width, testbed.nerf.camera.height)

print(f"✓ 点云已保存: {ply_path}")
print(f"   点数: {points.shape[0]}")
print(f"   格式: PLY (可导入 CloudCompare)")

def save_ply(filename, points, width, height):
    """保存点云为 PLY 格式"""
    header = [
        'ply',
        'format ascii 1.0',
        'element vertex',
        'property float x',
        'property float y',
        'property float z',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        'end_header',
        'element face',
        'property list uchar int vertex_index',
        'end_header'
    ]

    with open(filename, 'w') as f:
        for line in header:
            f.write(line + '\n')

        # 写入顶点
        f.write(f'element vertex {points.shape[0]}\n')
        for i in range(points.shape[0]):
            x, y, z = points[i, 0], points[i, 1], points[i, 2]
            r, g, b = int(min(255, max(0, z * 255)))
            f.write(f'{x} {y} {z} {r} {g} {b}\n')

        f.write('end_header\n')
        f.write('element face 0\n')
        f.write('end_header\n')

print("\n✓ PLY 文件生成完成！")
print(f"\n📥 文件位置: {ply_path}")
print(f"💡 使用方法:")
print(f"   1. 在 CloudCompare 中打开 {ply_path}")
print(f"   2. 或者使用 MeshLab、Blender 等工具查看")
print(f"   3. PLY 是标准的点云格式，广泛支持")
