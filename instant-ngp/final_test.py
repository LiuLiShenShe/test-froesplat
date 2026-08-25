#!/usr/bin/env python3
"""
Final validation script for instant-ngp installation.
"""

import sys
sys.path.insert(0, 'build')
import pyngp as ngp

print("=" * 60)
print("Instant-NGP 最终验证测试")
print("=" * 60)

print("\n✓ 测试 1: 导入 pyngp 模块")
print("  成功导入 pyngp")

print("\n✓ 测试 2: 创建 Testbed 实例")
testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
print("  Testbed 创建成功")

print("\n✓ 测试 3: 加载 NeRF 数据集")
testbed.load_training_data('data/nerf/fox')
print("  fox 数据集加载成功")

print("\n✓ 测试 4: 执行单步训练")
testbed.train()
print(f"  训练步骤完成")

print("\n✓ 测试 5: 训练快照保存测试")
testbed.save_snapshot("/tmp/test_snapshot.msgpack")
print("  快照保存成功")

print("\n" + "=" * 60)
print("所有测试通过！")
print("=" * 60)

print("\n可用的功能:")
print("  - Python 绑定 (pyngp)")
print("  - 命令行工具 (./build/instant-ngp)")
print("  - NeRF 训练")
print("  - SDF 训练")
print("  - 图像和体积渲染")

print("\n快速开始命令:")
print("  # Python 方式")
print("  source venv/bin/activate")
print("  python scripts/run.py --scene data/nerf/fox --n_steps 1000")
print()
print("  # 命令行方式（无GUI）")
print("  ./build/instant-ngp data/nerf/fox --no-gui")
