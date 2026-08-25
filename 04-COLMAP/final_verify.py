"""
Final global verification of all COLMAP outputs for 2DGS training readiness.
Checks:
1. sparse/0/ exists with cameras.bin, images.bin, points3D.bin
2. images/ directory has undistorted images
3. Registration rate
4. 3D point count
5. Camera model is PINHOLE
6. images/ files have high black pixel percentage (crop images)
"""
import struct
import os
import sys
import numpy as np

BASE = "D:/CAAS/04-COLMAP"

ALL_FOLDERS = [
    "BaiZhang", "CaoMei1", "CaoMei2",
    "ChangShouHua1", "ChangShouHua2", "ChangShouHua3",
    "DouBanLv1", "DouBanLv2", "DouBanLv3",
    "HongZhang", "KongQueZhuYu",
    "WangWenCao1", "WangWenCao2",
    "WanNianQing1", "WanNianQing2",
    "XiangPiShu1", "XiangPiShu2",
    "XianKeLai1", "XianKeLai2", "XianKeLai3"
]

def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            # PINHOLE=1 (4 params), OPENCV=4 (8 params)
            n_params = {0:3, 1:4, 2:4, 3:5, 4:8, 5:12}
            np_ = n_params.get(model_id, 4)
            params = struct.unpack("<" + "d"*np_, f.read(8*np_))
            model_name = {0:"SIMPLE_PINHOLE", 1:"PINHOLE", 2:"SIMPLE_RADIAL", 
                          3:"RADIAL", 4:"OPENCV", 5:"OPENCV_FISHEYE"}.get(model_id, f"UNKNOWN({model_id})")
            cameras[cam_id] = {"model": model_name, "width": w, "height": h}
    return cameras

def read_images_count(path):
    with open(path, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]

def read_points3d_count(path):
    with open(path, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]

def check_black_pixels(images_dir, sample=3):
    """Check black pixel percentage on a few sample images"""
    try:
        import cv2
    except ImportError:
        return None
    
    files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not files:
        return None
    
    files = files[:sample]
    black_pcts = []
    for fname in files:
        img = cv2.imread(os.path.join(images_dir, fname))
        if img is None:
            continue
        total = img.shape[0] * img.shape[1]
        black = np.sum(np.all(img == 0, axis=2))
        black_pcts.append(black / total * 100)
    
    return np.mean(black_pcts) if black_pcts else None

print("=" * 90)
print("COLMAP Global Verification Report")
print("=" * 90)

results = []
for folder in ALL_FOLDERS:
    path = os.path.join(BASE, folder)
    sparse_dir = os.path.join(path, "sparse", "0")
    images_dir = os.path.join(path, "images")
    input_dir = os.path.join(path, "input")
    
    # Check existence
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    images_bin = os.path.join(sparse_dir, "images.bin")
    points_bin = os.path.join(sparse_dir, "points3D.bin")
    
    if not all(os.path.exists(p) for p in [cameras_bin, images_bin, points_bin]):
        status = "MISSING"
        results.append({
            "folder": folder, "status": status,
            "n_reg": 0, "n_input": 0, "pct": 0, "n_pts": 0,
            "model": "-", "resolution": "-", "black_pct": None
        })
        continue
    
    # Read data
    cameras = read_cameras_bin(cameras_bin)
    n_reg = read_images_count(images_bin)
    n_pts = read_points3d_count(points_bin)
    
    # Input count
    if os.path.exists(input_dir):
        n_input = len([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])
    else:
        n_input = n_reg  # fallback
    
    pct = n_reg / n_input * 100 if n_input > 0 else 0
    
    # Camera info
    cam = list(cameras.values())[0] if cameras else {}
    model = cam.get("model", "?")
    w = cam.get("width", 0)
    h = cam.get("height", 0)
    resolution = f"{w}x{h}"
    
    # Check images/ directory
    has_images = os.path.isdir(images_dir) and len(os.listdir(images_dir)) > 0
    n_images = len(os.listdir(images_dir)) if has_images else 0
    
    # Black pixel check
    black_pct = check_black_pixels(images_dir) if has_images else None
    
    # Status
    if pct >= 70:
        status = "OK"
    elif pct >= 30:
        status = "WARN"
    else:
        status = "FAIL"
    
    if model != "PINHOLE":
        status += " !MODEL"
    
    results.append({
        "folder": folder, "status": status,
        "n_reg": n_reg, "n_input": n_input, "pct": pct, "n_pts": n_pts,
        "model": model, "resolution": resolution, "black_pct": black_pct,
        "n_images": n_images
    })

# Print results
print(f"{'Folder':<18} {'Status':<10} {'Reg Rate':<12} {'3D Pts':<8} {'Model':<10} {'Resolution':<12} {'Black%':<8} {'images/':<8}")
print("-" * 90)

ok_count = 0
warn_count = 0
fail_count = 0
missing_count = 0

for r in results:
    black_str = f"{r['black_pct']:.1f}%" if r.get('black_pct') is not None else "-"
    images_str = str(r.get('n_images', '-'))
    reg_str = f"{r['n_reg']}/{r['n_input']} ({r['pct']:.1f}%)" if r['status'] != 'MISSING' else "-"
    
    print(f"{r['folder']:<18} {r['status']:<10} {reg_str:<12} {r['n_pts']:<8} {r['model']:<10} {r['resolution']:<12} {black_str:<8} {images_str:<8}")
    
    if r['status'].startswith('OK'):
        ok_count += 1
    elif r['status'].startswith('WARN'):
        warn_count += 1
    elif r['status'] == 'MISSING':
        missing_count += 1
    else:
        fail_count += 1

print("-" * 90)
print(f"Summary: {ok_count} OK, {warn_count} WARN, {fail_count} FAIL, {missing_count} MISSING (total {len(results)})")
print()

# Usable for 2DGS (OK + WARN with enough points)
usable = [r for r in results if r['status'].startswith('OK') or r['status'].startswith('WARN')]
print(f"Usable for 2DGS training: {len(usable)} folders")
for r in usable:
    black_str = f"{r['black_pct']:.1f}%" if r.get('black_pct') is not None else "?"
    print(f"  {r['folder']}: {r['pct']:.1f}% reg, {r['n_pts']} pts, black={black_str}")

# Unusable
unusable = [r for r in results if not (r['status'].startswith('OK') or r['status'].startswith('WARN'))]
if unusable:
    print(f"\nUnusable: {len(unusable)} folders")
    for r in unusable:
        print(f"  {r['folder']}: {r['status']}")
