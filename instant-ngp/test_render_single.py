#!/usr/bin/env python3
"""
测试单帧渲染 - 验证 instant-ngp 渲染功能
"""
import sys
import os
sys.path.insert(0, '/data/fj/instant-ngp/build')
import pyngp as ngp

scene_path = "/data/fj/10-Instant-NGP/CaoMei1"
snapshot_path = "/data/fj/10-Instant-NGP/CaoMei1_trained.msgpack"
output_file = f"{scene_path}/test_render.png"

print("=" * 60)
print("测试单帧渲染")
print("=" * 60)
print(f"场景: {scene_path}")
print(f"快照: {snapshot_path}")
print(f"输出: {output_file}")
print()

# 创建 testbed
testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

# 加载快照
print("[1/2] 加载快照...")
testbed.load_snapshot(snapshot_path)
print("✓ 快照加载成功")

# 渲染一帧
print("\n[2/2] 渲染一帧...")
testbed.render(output_file)

# 检查结果
if os.path.exists(output_file):
    file_size = os.path.getsize(output_file) / 1024
    print(f"✓ 渲染成功！")
    print(f"输出: {output_file}")
    print(f"大小: {file_size} KB")
else:
    print("✗ 渲染失败！")
    sys.exit(1)
