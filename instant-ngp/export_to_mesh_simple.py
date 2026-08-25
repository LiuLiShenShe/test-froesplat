#!/usr/bin/env python3
"""
简化版本的 NeRF → Mesh 导出工具
使用 instant-ngp 原生方法
"""
import sys
sys.path.insert(0, '/data/fj/instant-ngp/build')
import pyngp as ngp

# 快照路径
snapshot_path = "/data/fj/10-Instant-NGP/CaoMei1_trained.msgpack"
output_path = "/data/fj/10-Instant-NGP/CaoMei1_mesh.obj"

print("=" * 70)
print("Instant-NGP NeRF → Mesh 导出")
print("=" * 70)
print(f"快照: {snapshot_path}")
print(f"输出: {output_path}")
print("分辨率: 512³ (默认）")
print("")

# 创建 testbed
testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

# 加载快照
print("[1/3] 加载 NeRF 模型...")
testbed.load_snapshot(snapshot_path)
print("✓ 模型加载成功")

# 导出 mesh (使用原生的 compute_and_save_marching_cubes_mesh 方法）
print("")
print("[2/3] 提取 Mesh (Marching Cubes 512³）...")
print("这可能需要几分钟，请耐心等待...")
print("")

try:
    # 使用原生的 mesh 导出方法
    testbed.compute_and_save_marching_cubes_mesh(str(output_path), resolution=512, density_thresh=2.5)

    print("")
    print("=" * 70)
    print("✓ Mesh 导出成功！")
    print("=" * 70)
    print("")
    print(f"输出文件: {output_path}")
    print("")
    print("💡 如何在 CloudCompare 中使用:")
    print(f"   1. 打开 CloudCompare")
    print(f"   2. 导入: File → Import Mesh")
    print(f"   3. 选择: {output_path}")
    print(f"   4. OBJ 格式，支持法线、UV 等完整信息")
    print("")

except Exception as e:
    print(f"")
    print("✗ 导出失败！")
    print(f"错误: {e}")
    print("")
    sys.exit(1)
