#!/usr/bin/env python3
"""
使用 pycolmap 从 COLMAP 二进制文件生成 transforms.json
"""

import numpy as np
import json
from pathlib import Path
import sys

try:
    import pycolmap
except ImportError:
    print("错误: 请安装 pycolmap: pip install pycolmap")
    sys.exit(1)

def qvec2rotmat(qvec):
    """四元数转旋转矩阵"""
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def create_transform_matrix(qvec, tvec):
    """创建 4x4 变换矩阵 (camera-to-world)"""
    R = qvec2rotmat(qvec)
    t = np.array(tvec)

    # COLMAP 使用 world-to-camera 变换
    # NeRF 需要 camera-to-world 变换
    R = R.T
    t = -R @ t

    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t

    return transform

def colmap_to_nerf(colmap_dir, images_dir, output_file, aabb_scale=16):
    """将 COLMAP 二进制文件转换为 NeRF transforms.json"""

    colmap_path = Path(colmap_dir)

    try:
        # 使用 pycolmap 读取模型
        reconstruction = pycolmap.Reconstruction(colmap_path)
    except Exception as e:
        print(f"错误: 无法读取 COLMAP 模型: {e}")
        return 0

    # 构建 transforms.json
    transforms = {
        "aabb_scale": aabb_scale,
        "frames": []
    }

    # 获取相机参数（假设所有图像使用相同相机）
    cameras = list(reconstruction.cameras.values())
    if not cameras:
        print("错误: 未找到相机")
        return 0

    camera = cameras[0]
    width = camera.width
    height = camera.height

    # 根据相机模型获取参数
    params = camera.params if hasattr(camera, 'params') else None

    if params is not None and len(params) >= 4:
        # OPENCV 模型: fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
    elif params is not None and len(params) >= 1:
        # SIMPLE_RADIAL 模型: f, cx, cy, k
        fx = params[0]
        fy = params[0]
        cx = params[1] if len(params) > 1 else width / 2
        cy = params[2] if len(params) > 2 else height / 2
    else:
        # 默认值
        fx = width / 2
        fy = width / 2
        cx = width / 2
        cy = height / 2

    # 计算视场角
    camera_angle_x = 2 * np.arctan(width / (2 * fx))
    camera_angle_y = 2 * np.arctan(height / (2 * fy))

    transforms['camera_angle_x'] = float(camera_angle_x)
    transforms['camera_angle_y'] = float(camera_angle_y)
    transforms['fl_x'] = float(fx)
    transforms['fl_y'] = float(fy)
    transforms['cx'] = float(cx)
    transforms['cy'] = float(cy)
    transforms['w'] = width
    transforms['h'] = height

    # 畸变参数（如果有的话）
    if hasattr(camera, 'params') and len(camera.params) >= 8:
        transforms['k1'] = float(camera.params[4])
        transforms['k2'] = float(camera.params[5])
        transforms['p1'] = float(camera.params[6])
        transforms['p2'] = float(camera.params[7])

    # 处理每张图像
    num_frames = 0
    for image_id, image in reconstruction.images.items():
        if not image.has_pose:
            continue

        # 获取 cam_from_world 变换（调用方法）
        pose = image.cam_from_world()

        # 提取旋转（四元数）和平移
        # pycolmap 使用 xyzw 顺序，我们需要 wxyz
        rotation = pose.rotation
        quat_xyzw = rotation.quat  # 或者 rotation.xyzw
        qvec = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # wxyz
        tvec = list(pose.translation)

        name = image.name

        # 创建变换矩阵
        transform = create_transform_matrix(qvec, tvec)

        # 添加到 frames
        frame = {
            "file_path": f"images/{name}",
            "transform_matrix": transform.tolist()
        }
        transforms['frames'].append(frame)
        num_frames += 1

    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(transforms, f, indent=2)

    return num_frames

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: python pycolmap_to_nerf.py <colmap_dir> <images_dir> <output_file> [aabb_scale]")
        print("示例: python pycolmap_to_nerf.py sparse/0 images transforms.json 16")
        sys.exit(1)

    colmap_dir = sys.argv[1]
    images_dir = sys.argv[2]
    output_file = sys.argv[3]
    aabb_scale = int(sys.argv[4]) if len(sys.argv) > 4 else 16

    try:
        num_frames = colmap_to_nerf(colmap_dir, images_dir, output_file, aabb_scale)
        if num_frames > 0:
            print(f"✓ 成功生成 {output_file}")
            print(f"  包含 {num_frames} 帧")
            print(f"  aabb_scale = {aabb_scale}")
        else:
            print("✗ 生成失败")
            sys.exit(1)
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
