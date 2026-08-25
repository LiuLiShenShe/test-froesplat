#!/usr/bin/env python3
"""
简化版本的 NeRF → Mesh 导出工具
使用 instant-ngp 原生方法
"""
import sys
sys.path.insert(0, '/data/fj/instant-ngp/build')
import pyngp as ngp
import numpy as np
import struct

# 快照路径
snapshot_path = "/data/fj/10-Instant-NGP/CaoMei1_trained.msgpack"
output_obj = "/data/fj/10-Instant-NGP/CaoMei1_mesh.obj"
output_ply = "/data/fj/10-Instant-NGP/CaoMei1_mesh.ply"

print("=" * 70)
print("Instant-NGP NeRF → Mesh 导出（原生方法）")
print("=" * 70)
print(f"快照: {snapshot_path}")
print(f"输出 OBJ: {output_obj}")
print(f"输出 PLY: {output_ply}")
print("分辨率: 256³ (平衡质量和速度）")
print("")

# 创建 testbed
testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

# 加载快照
print("[1/4] 加载 NeRF 模型...")
testbed.load_snapshot(snapshot_path)
print("✓ 模型加载成功")

# 使用原生方法提取 mesh (返回 numpy 数组)
print("")
print("[2/4] 提取 Mesh (Marching Cubes 256³）...")
print("这可能需要几分钟，请耐心等待...")
print("")

try:
    # 使用原生方法返回 dict
    result = testbed.compute_marching_cubes_mesh(resolution=256, thresh=2.5)

    V = result['V']  # vertices (N x 3)
    N = result['N']  # normals (N x 3)
    C = result['C']  # colors (N x 3)
    F = result['F']  # faces (M x 3)

    print(f"✓ Mesh 提取完成！")
    print(f"   顶点数: {V.shape[0]}")
    print(f"   三角面数: {F.shape[0]}")
    print("")

    # 保存为 OBJ 格式
    print("[3/4] 保存为 OBJ 格式...")
    with open(output_obj, 'w') as f:
        # 顶点
        for i, (x, y, z) in enumerate(V):
            f.write(f"v {x} {y} {z}")
            if C is not None and i < len(C):
                f.write(f" {C[i,0]} {C[i,1]} {C[i,2]}")

        # 面和法线
        for i in range(F.shape[0]):
            f.write(f"vn {N[i,0]} {N[i,1]} {N[i,2]}")
            face = F[i]
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}")
            f.write(f"vn {face[0]} {face[1]} {face[2]}")

    print(f"✓ OBJ 已保存: {output_obj}")

    # 保存为 PLY 格式
    print("")
    print("[4/4] 保存为 PLY 格式...")
    with open(output_ply, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('element face\n')
        f.write('property list uchar int vertex_index\n')
        f.write('end_header\n')
        f.write(f'element vertex {V.shape[0]}\n')

        # 顶点
        for i in range(V.shape[0]):
            f.write(f'{V[i,0]} {V[i,1]} {V[i,2]}')
            if C is not None:
                r = int(min(255, max(0, C[i,0])))
                g = int(min(255, max(0, C[i,1])))
                b = int(min(255, max(0, C[i,2])))
                f.write(f'{r} {g} {b}\n')
            else:
                f.write('255 255 255\n')

        f.write('end_header\n')
        f.write(f'element face {F.shape[0]}\n')

        # 面索引
        for i in range(F.shape[0]):
            f.write(f'{F[i,0]+1} {F[i,1]+1} {F[i,2]+1}\n')

        f.write('end_header\n')

    print(f"✓ PLY 已保存: {output_ply}")

    # 检查文件大小
    obj_size = (output_obj.stat().st_size / 1024 / 1024)
    ply_size = (output_ply.stat().st_size / 1024 / 1024)

    print("")
    print("=" * 70)
    print("✓ 导出成功！")
    print("=" * 70)
    print("")
    print("文件信息:")
    print(f"  OBJ 文件: {output_obj}")
    print(f"    大小: {obj_size:.2f} MB")
    print("")
    print(f"  PLY 文件: {output_ply}")
    print(f"    大小: {ply_size:.2f} MB")
    print("")
    print("💡 如何在 CloudCompare 中使用:")
    print(f"   1. 打开 CloudCompare")
    print(f"   2. 导入: File → Import Mesh")
    print(f"   3. 选择以下任一文件:")
    print(f"      {output_obj} (OBJ 格式，完整 3D 模型)")
    print(f"      {output_ply} (PLY 格式，点云 + 颜色)")
    print("")

except Exception as e:
    print(f"")
    print("✗ 导出失败！")
    print(f"错误: {e}")
    print("")
    import traceback
    traceback.print_exc()
    sys.exit(1)
