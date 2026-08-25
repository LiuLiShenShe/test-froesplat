#!/usr/bin/env python3
"""
将 Instant-NGP 训练的 NeRF 模型导出为 3D mesh
支持 OBJ/PLY 格式，可在 Blender、MeshLab 等软件中查看
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, '/data/fj/instant-ngp/build')
import pyngp as ngp

def export_nerf_to_mesh(snapshot_path, output_mesh, resolution=512, density_thresh=2.5):
    """
    将 NeRF 快照导出为 mesh

    Args:
        snapshot_path: 训练好的 .msgpack 快照路径
        output_mesh: 输出 mesh 文件路径（.obj 或 .ply）
        resolution: marching cubes 分辨率（越大越精细，但更慢）
        density_thresh: 密度阈值（越大越稀疏）
    """

    print("=" * 70)
    print("Instant-NGP NeRF → Mesh 导出工具")
    print("=" * 70)
    print(f"快照: {snapshot_path}")
    print(f"输出: {output_mesh}")
    print(f"分辨率: {resolution}³")
    print(f"密度阈值: {density_thresh}")
    print("=" * 70)

    # 创建 testbed
    print("\n[1/3] 加载 NeRF 模型...")
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    # 加载快照
    testbed.load_snapshot(str(snapshot_path))
    print("✓ 模型加载成功")

    # 导出 mesh
    print(f"\n[2/3] 提取 mesh（Marching Cubes {resolution}³）...")
    print("  这可能需要几分钟，请耐心等待...")

    output_path = Path(output_mesh)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 调用 pyngp 的 mesh 导出功能
    testbed.save_mesh(
        str(output_path),
        resolution=resolution,
        density_thresh=density_thresh,
        colors=True,  # 导出顶点颜色
        normals=True  # 导出法线
    )

    print(f"✓ Mesh 提取完成")

    # 检查文件
    if output_path.exists():
        file_size = output_path.stat().st_size / 1024 / 1024
        print(f"\n[3/3] 导出成功！")
        print(f"  文件: {output_path}")
        print(f"  大小: {file_size:.2f} MB")
        print(f"\n可以在以下软件中查看:")
        print(f"  - Blender: File → Import → {output_path.suffix.upper()}")
        print(f"  - MeshLab: File → Import Mesh")
        print(f"  - CloudCompare: File → Open")
        return True
    else:
        print("\n✗ 导出失败")
        return False

def main():
    parser = argparse.ArgumentParser(description="导出 Instant-NGP NeRF 为 3D mesh")
    parser.add_argument("snapshot", help="训练好的快照文件 (.msgpack)")
    parser.add_argument("-o", "--output", help="输出 mesh 文件路径")
    parser.add_argument("-r", "--resolution", type=int, default=512,
                        help="Marching Cubes 分辨率 (默认: 512)")
    parser.add_argument("-d", "--density", type=float, default=2.5,
                        help="密度阈值 (默认: 2.5，越大越稀疏)")

    args = parser.parse_args()

    # 自动生成输出路径
    if not args.output:
        snapshot_path = Path(args.snapshot)
        args.output = snapshot_path.parent / f"{snapshot_path.stem}_mesh.obj"

    success = export_nerf_to_mesh(
        args.snapshot,
        args.output,
        args.resolution,
        args.density
    )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
