"""
fix_images_with_crops.py
========================
修复 rerun 文件夹的 images/ 目录：
将原图替换为 SAM crop 图片（经过与 COLMAP 一致的去畸变处理）

流程: 
1. 读取 distorted/sparse/0/cameras.bin 获取 OPENCV 畸变参数
2. 读取 sparse/0/cameras.bin 获取 PINHOLE 无畸变参数（目标尺寸）
3. 对每个 SAM crop_XXXX.png 做去畸变，生成与 COLMAP undistorter 一致的输出
4. 保存为 XXXX.jpg 替换 images/ 中的原图

同时处理 CaoMei2 和 ChangShouHua3 的恢复（用 crop 图重跑 COLMAP）
"""
import struct, os, sys, shutil, json, time
import numpy as np

try:
    import cv2
except ImportError:
    print("需要 opencv-python, 尝试安装...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2

from PIL import Image

# ============ Configuration ============
SAM_DIR     = r"D:\CAAS\03-SAM"
COLMAP_DIR  = r"D:\CAAS\04-COLMAP"

# Correct number of params per COLMAP camera model
COLMAP_NUM_PARAMS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 14, 7: 5, 8: 4, 9: 5, 10: 12,
}

# 9 folders that were successfully rerun with originals (100% reg, good 3D)
# Need to replace images/ with undistorted SAM crops
RERUN_OK_FOLDERS = [
    "CaoMei1", "ChangShouHua2", "DouBanLv1", "DouBanLv2", "HongZhang",
    "WanNianQing1", "WanNianQing2", "WangWenCao2", "XianKeLai2",
]


# ============ Camera I/O ============
def read_cameras_bin(path):
    cameras = {}
    with open(path, 'rb') as f:
        num = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num):
            cid = struct.unpack('<i', f.read(4))[0]
            model_id = struct.unpack('<i', f.read(4))[0]
            w = struct.unpack('<Q', f.read(8))[0]
            h = struct.unpack('<Q', f.read(8))[0]
            np_ = COLMAP_NUM_PARAMS.get(model_id, 4)
            params = struct.unpack(f'<{np_}d', f.read(8*np_))
            cameras[cid] = {
                'model_id': model_id, 'w': w, 'h': h, 'params': params
            }
    return cameras


def undistort_crop_image(crop_path, dist_cam, undist_cam, output_path):
    """
    对 SAM crop 图做与 COLMAP image_undistorter 一致的去畸变处理
    
    dist_cam: OPENCV 相机参数 (from distorted/sparse/0/)
    undist_cam: PINHOLE 相机参数 (from sparse/0/)
    """
    # Read crop image
    crop_img = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
    if crop_img is None:
        return False
    
    # OPENCV model: fx, fy, cx, cy, k1, k2, p1, p2
    fx, fy, cx, cy, k1, k2, p1, p2 = dist_cam['params']
    
    # Distorted camera matrix
    K_dist = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Distortion coefficients (k1, k2, p1, p2)
    dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float64)
    
    # Undistorted camera matrix (PINHOLE) 
    fx_u, fy_u, cx_u, cy_u = undist_cam['params']
    K_undist = np.array([
        [fx_u, 0, cx_u],
        [0, fy_u, cy_u],
        [0, 0, 1]
    ], dtype=np.float64)
    
    target_w = undist_cam['w']
    target_h = undist_cam['h']
    
    # Create undistortion maps
    map1, map2 = cv2.initUndistortRectifyMap(
        K_dist, dist_coeffs, None, K_undist,
        (target_w, target_h), cv2.CV_32FC1
    )
    
    # Apply the mapping to the crop image
    undistorted = cv2.remap(crop_img, map1, map2, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    # Save as JPEG (matching the expected filename)
    cv2.imwrite(output_path, undistorted, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return True


def process_folder(folder_name):
    """处理一个 rerun 文件夹：替换 images/ 中的原图为去畸变后的 crop 图"""
    folder_path = os.path.join(COLMAP_DIR, folder_name)
    images_dir = os.path.join(folder_path, "images")
    sam_folder = os.path.join(SAM_DIR, folder_name)
    
    # Read camera params
    dist_cam_path = os.path.join(folder_path, "distorted", "sparse", "0", "cameras.bin")
    undist_cam_path = os.path.join(folder_path, "sparse", "0", "cameras.bin")
    
    if not os.path.exists(dist_cam_path):
        print(f"  ⚠ {folder_name}: distorted cameras not found, skipping")
        return {"status": "skip", "reason": "no distorted cameras"}
    
    dist_cams = read_cameras_bin(dist_cam_path)
    undist_cams = read_cameras_bin(undist_cam_path)
    
    # Get the first (and typically only) camera
    dist_cam = list(dist_cams.values())[0]
    undist_cam = list(undist_cams.values())[0]
    
    print(f"  Camera: OPENCV {dist_cam['w']}x{dist_cam['h']} → PINHOLE {undist_cam['w']}x{undist_cam['h']}")
    
    # List images in images/ directory
    img_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    
    replaced = 0
    missing = 0
    failed = 0
    
    for img_file in img_files:
        # Map: 0000.jpg → crop_0000.png
        base = os.path.splitext(img_file)[0]  # "0000"
        crop_name = f"crop_{base}.png"
        crop_path = os.path.join(sam_folder, crop_name)
        
        if not os.path.exists(crop_path):
            missing += 1
            continue
        
        output_path = os.path.join(images_dir, img_file)
        
        if undistort_crop_image(crop_path, dist_cam, undist_cam, output_path):
            replaced += 1
        else:
            failed += 1
    
    print(f"  Replaced: {replaced}, Missing crops: {missing}, Failed: {failed}")
    return {"status": "ok", "replaced": replaced, "missing": missing, "failed": failed}


def verify_replacement(folder_name):
    """验证替换后的图片：检查黑色像素比例（crop 图应有大量黑色背景）"""
    images_dir = os.path.join(COLMAP_DIR, folder_name, "images")
    img_files = sorted(os.listdir(images_dir))
    
    if not img_files:
        return None
    
    # Sample first image
    sample = cv2.imread(os.path.join(images_dir, img_files[0]))
    if sample is None:
        return None
    
    h, w = sample.shape[:2]
    total = h * w
    black = np.sum(np.all(sample < 5, axis=2))
    black_pct = black / total * 100
    
    return {"file": img_files[0], "size": f"{w}x{h}", "black_pct": black_pct}


# ============ Main ============
if __name__ == "__main__":
    print("=" * 70)
    print("修复 rerun 文件夹 images/ : 替换原图为去畸变后的 SAM crop 图")
    print("=" * 70)
    
    results = {}
    
    for folder in RERUN_OK_FOLDERS:
        print(f"\n[{folder}]")
        result = process_folder(folder)
        results[folder] = result
        
        # Verify
        v = verify_replacement(folder)
        if v:
            print(f"  验证: {v['file']} size={v['size']} black={v['black_pct']:.1f}%")
            tag = "✅ CROP" if v['black_pct'] > 30 else "⚠ 可能有问题"
            print(f"  {tag}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    for f, r in results.items():
        print(f"  {f:20s}: {r}")
    
    # Save log
    log_path = os.path.join(COLMAP_DIR, "fix_images_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nLog saved to {log_path}")
