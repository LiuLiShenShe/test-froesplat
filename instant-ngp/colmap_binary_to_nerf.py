#!/usr/bin/env python3
"""
直接从 COLMAP 二进制文件生成 transforms.json
无需 colmap 命令行工具
"""

import numpy as np
import json
import struct
from pathlib import Path

def read_colmap_binary_cameras(binary_file):
    """读取 COLMAP cameras.bin 文件"""
    cameras = {}
    with open(binary_file, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack('<I', f.read(4))[0]
            model_id = struct.unpack('<I', f.read(4))[0]
            width = struct.unpack('<I', f.read(4))[0]
            height = struct.unpack('<I', f.read(4))[0]

            # 读取参数
            num_params = struct.unpack('<I', f.read(4))[0]
            params = struct.unpack('<' + 'd'*num_params, f.read(8*num_params))

            cameras[camera_id] = {
                'id': camera_id,
                'model': model_id,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def read_colmap_binary_images(binary_file):
    """读取 COLMAP images.bin 文件"""
    images = {}
    with open(binary_file, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack('<I', f.read(4))[0]
            qw, qx, qy, qz = struct.unpack('<dddd', f.read(32))
            tx, ty, tz = struct.unpack('<ddd', f.read(24))
            camera_id = struct.unpack('<I', f.read(4))[0]

            # 读取文件名
            name_len = struct.unpack('<B', f.read(1))[0]
            name = f.read(name_len).decode('utf-8')

            # 跳过 2D 点
            num_points2D = struct.unpack('<Q', f.read(8))[0]
            f.read(num_points2D * 24)  # x, y, point3D_id

            images[image_id] = {
                'id': image_id,
                'qvec': [qw, qx, qy, qz],
                'tvec': [tx, ty, tz],
                'camera_id': camera_id,
                'name': name
            }
    return images

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
    """创建 4x4 变换矩阵"""
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
    cameras_file = colmap_path / "cameras.bin"
    images_file = colmap_path / "images.bin"

    if not cameras_file.exists() or not images_file.exists():
        raise FileNotFoundError("未找到 COLMAP 二进制文件")

    # 读取数据
    cameras = read_colmap_binary_cameras(cameras_file)
    images = read_colmap_binary_images(images_file)

    # 构建 transforms.json
    transforms = {
        "aabb_scale": aabb_scale,
        "frames": []
    }

    # 获取相机参数（假设所有图像使用相同相机）
    camera = list(cameras.values())[0]
    width = camera['width']
    height = camera['height']
    params = camera['params']

    # 根据相机模型设置参数
    # OPENCV 模型: fx, fy, cx, cy, k1, k2, p1, p2
    if len(params) >= 4:
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]

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

        # 畸变参数
        if len(params) >= 8:
            transforms['k1'] = float(params[4])
            transforms['k2'] = float(params[5])
            transforms['p1'] = float(params[6])
            transforms['p2'] = float(params[7])

    # 处理每张图像
    for img_id, img_data in images.items():
        qvec = img_data['qvec']
        tvec = img_data['tvec']
        name = img_data['name']

        # 创建变换矩阵
        transform = create_transform_matrix(qvec, tvec)

        # 添加到 frames
        frame = {
            "file_path": f"images/{name}",
            "transform_matrix": transform.tolist()
        }
        transforms['frames'].append(frame)

    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(transforms, f, indent=2)

    return len(transforms['frames'])

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("用法: python colmap_binary_to_nerf.py <colmap_dir> <images_dir> <output_file> [aabb_scale]")
        sys.exit(1)

    colmap_dir = sys.argv[1]
    images_dir = sys.argv[2]
    output_file = sys.argv[3]
    aabb_scale = int(sys.argv[4]) if len(sys.argv) > 4 else 16

    try:
        num_frames = colmap_to_nerf(colmap_dir, images_dir, output_file, aabb_scale)
        print(f"✓ 成功生成 transforms.json，包含 {num_frames} 帧")
    except Exception as e:
        print(f"✗ 失败: {e}")
        sys.exit(1)
