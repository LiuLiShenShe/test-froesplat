#!/usr/bin/env python3
"""
测试加载 NeRF 快照并渲染一帧
"""
import sys
sys.path.insert(0, '/data/fj/instant-ngp/build')
import pyngp as ngp

# 创建 testbed
testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

# 加载快照
snapshot_path = "/data/fj/10-Instant-NGP/CaoMei1_trained.msgpack"
print(f"加载快照: {snapshot_path}")

try:
    testbed.load_snapshot(snapshot_path)
    print("✓ 快照加载成功！")
    print(f"模式: {testbed.mode}")
    print(f"训练步数: {testbed.training_step}")
    print(f"渲染分辨率: {testbed.render_width} x {testbed.render_height}")

except Exception as e:
    print(f"❌ 加载快照失败: {e}")
    sys.exit(1)

print("\n快照验证完成！")
