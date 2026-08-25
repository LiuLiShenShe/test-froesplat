"""
fix_images_with_crops.py - v2
替换 rerun 文件夹 images/ 中的原图为去畸变后的 SAM crop 图片
"""
import struct, os, sys, json, time
import numpy as np
import cv2
from pathlib import Path

SAM_DIR     = Path(r"D:\CAAS\03-SAM")
COLMAP_DIR  = Path(r"D:\CAAS\04-COLMAP")

COLMAP_NUM_PARAMS = {0:3, 1:4, 2:4, 3:5, 4:8, 5:8, 6:14, 7:5, 8:4, 9:5, 10:12}

RERUN_OK_FOLDERS = [
    "CaoMei1", "ChangShouHua2", "DouBanLv1", "DouBanLv2", "HongZhang",
    "WanNianQing1", "WanNianQing2", "WangWenCao2", "XianKeLai2",
]

def read_cameras_bin(path):
    cameras = {}
    with open(path, 'rb') as f:
        num = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num):
            cid = struct.unpack('<i', f.read(4))[0]
            mid = struct.unpack('<i', f.read(4))[0]
            w = struct.unpack('<Q', f.read(8))[0]
            h = struct.unpack('<Q', f.read(8))[0]
            np_ = COLMAP_NUM_PARAMS.get(mid, 4)
            params = struct.unpack(f'<{np_}d', f.read(8*np_))
            cameras[cid] = {'model_id': mid, 'w': w, 'h': h, 'params': params}
    return cameras

def process_folder(folder_name):
    folder = COLMAP_DIR / folder_name
    images_dir = folder / "images"
    sam_folder = SAM_DIR / folder_name
    
    dist_cam = list(read_cameras_bin(folder / "distorted" / "sparse" / "0" / "cameras.bin").values())[0]
    undist_cam = list(read_cameras_bin(folder / "sparse" / "0" / "cameras.bin").values())[0]
    
    # Build undistortion maps once (same for all images in this folder)
    fx, fy, cx, cy, k1, k2, p1, p2 = dist_cam['params']
    K_dist = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float64)
    
    fx_u, fy_u, cx_u, cy_u = undist_cam['params']
    K_undist = np.array([[fx_u,0,cx_u],[0,fy_u,cy_u],[0,0,1]], dtype=np.float64)
    target_w, target_h = int(undist_cam['w']), int(undist_cam['h'])
    
    map1, map2 = cv2.initUndistortRectifyMap(
        K_dist, dist_coeffs, None, K_undist, (target_w, target_h), cv2.CV_32FC1
    )
    
    img_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg','.png'))])
    replaced = 0
    missing = 0
    
    for i, img_file in enumerate(img_files):
        base = os.path.splitext(img_file)[0]
        crop_path = sam_folder / f"crop_{base}.png"
        
        if not crop_path.exists():
            missing += 1
            continue
        
        crop_img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        undistorted = cv2.remap(crop_img, map1, map2, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        output_path = images_dir / img_file
        cv2.imwrite(str(output_path), undistorted, [cv2.IMWRITE_JPEG_QUALITY, 100])
        replaced += 1
        
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{len(img_files)}...", flush=True)
    
    return replaced, missing, len(img_files)

# Main
print("=" * 60, flush=True)
print("修复 images/ : 原图 → 去畸变 SAM crop", flush=True)
print("=" * 60, flush=True)

results = {}
for folder in RERUN_OK_FOLDERS:
    t0 = time.time()
    print(f"\n[{folder}]", flush=True)
    replaced, missing, total = process_folder(folder)
    dt = time.time() - t0
    print(f"  Done: {replaced}/{total} replaced, {missing} missing, {dt:.1f}s", flush=True)
    results[folder] = {"replaced": replaced, "total": total, "missing": missing, "time": f"{dt:.1f}s"}

# Verify
print("\n" + "=" * 60, flush=True)
print("验证:", flush=True)
for folder in RERUN_OK_FOLDERS:
    imgs = sorted(os.listdir(COLMAP_DIR / folder / "images"))
    if imgs:
        sample = cv2.imread(str(COLMAP_DIR / folder / "images" / imgs[0]))
        h, w = sample.shape[:2]
        black = np.sum(np.all(sample < 5, axis=2)) / (h*w) * 100
        print(f"  {folder:20s}: {imgs[0]} {w}x{h} black={black:.1f}%", flush=True)

with open(COLMAP_DIR / "fix_images_log.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDone!", flush=True)
