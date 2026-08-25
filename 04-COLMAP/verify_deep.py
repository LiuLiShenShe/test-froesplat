"""
Deep verification of COLMAP rerun results.
Checks: camera model, image dimensions, sparse quality, 2DGS compatibility.
"""
import struct
import os
import json
from PIL import Image
import numpy as np

COLMAP_DIR = r"D:\CAAS\04-COLMAP"
SAM_DIR = r"D:\CAAS\03-SAM"

# All folders including OK ones and rerun ones
ALL_FOLDERS = [
    "BaiZhang", "CaoMei1", "CaoMei2", "ChangShouHua1", "ChangShouHua2",
    "ChangShouHua3", "DouBanLv1", "DouBanLv2", "DouBanLv3", "HongZhang",
    "KongQueZhuYu", "WanNianQing1", "WanNianQing2", "WangWenCao1", "WangWenCao2",
    "XianKeLai1", "XianKeLai2", "XianKeLai3", "XiangPiShu1", "XiangPiShu2"
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
            nparams = {0:3,1:4,2:4,3:5,4:8}.get(mid, 0)
            params = struct.unpack(f'<{nparams}d', f.read(8*nparams))
            models = {0:'SIMPLE_PINHOLE',1:'PINHOLE',2:'SIMPLE_RADIAL',3:'RADIAL',4:'OPENCV'}
            cameras[cid] = {'model': models.get(mid, f'UNK({mid})'), 'width': w, 'height': h, 'params': params}
    return cameras

def read_images_count(path):
    with open(path, 'rb') as f:
        return struct.unpack('<Q', f.read(8))[0]

def read_points_count(path):
    with open(path, 'rb') as f:
        return struct.unpack('<Q', f.read(8))[0]

print("=" * 90)
print("DEEP VERIFICATION OF ALL COLMAP OUTPUTS")
print("=" * 90)
print()

# Header
print(f"{'Folder':<18s} {'CamModel':<10s} {'Resolution':<13s} {'Input':>5s} "
      f"{'Reg':>5s} {'Rate':>7s} {'3DPts':>8s} {'ImgDir':>6s} {'ImgFmt':<6s} {'2DGS':>5s}")
print("-" * 90)

issues = []

for folder in ALL_FOLDERS:
    base = os.path.join(COLMAP_DIR, folder)
    sp = os.path.join(base, 'sparse', '0')
    img_dir = os.path.join(base, 'images')
    inp_dir = os.path.join(base, 'input')
    
    if not os.path.exists(sp):
        print(f"{folder:<18s}  ** sparse/0 NOT FOUND **")
        issues.append((folder, "no sparse/0"))
        continue
    
    # Camera info
    cams = read_cameras_bin(os.path.join(sp, 'cameras.bin'))
    cam = list(cams.values())[0]
    model = cam['model']
    res = f"{cam['width']}x{cam['height']}"
    
    # Registration
    n_reg = read_images_count(os.path.join(sp, 'images.bin'))
    n_pts = read_points_count(os.path.join(sp, 'points3D.bin'))
    n_input = len(os.listdir(inp_dir)) if os.path.exists(inp_dir) else 0
    rate = n_reg / n_input * 100 if n_input > 0 else 0
    
    # Images directory
    n_img = len([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]) if os.path.exists(img_dir) else 0
    
    # Check actual image format in images/
    img_fmt = "?"
    if n_img > 0:
        first = sorted(os.listdir(img_dir))[0]
        img_fmt = os.path.splitext(first)[1]
    
    # 2DGS compatibility check
    compatible = True
    problems = []
    
    # Must be PINHOLE or SIMPLE_PINHOLE
    if model not in ('PINHOLE', 'SIMPLE_PINHOLE'):
        compatible = False
        problems.append(f"bad_cam:{model}")
    
    # Must have sparse/0 with reasonable points
    if n_reg < 10:
        compatible = False
        problems.append(f"low_reg:{n_reg}")
    
    if n_pts < 100:
        compatible = False
        problems.append(f"low_pts:{n_pts}")
    
    # images/ count should match registered count
    if n_img != n_reg:
        problems.append(f"img_mismatch:{n_img}vs{n_reg}")
    
    status = "OK" if compatible and not problems else "WARN" if compatible else "FAIL"
    
    print(f"{folder:<18s} {model:<10s} {res:<13s} {n_input:>5d} "
          f"{n_reg:>5d} {rate:>6.1f}% {n_pts:>8d} {n_img:>6d} {img_fmt:<6s} {status:>5s}"
          + (f"  {'; '.join(problems)}" if problems else ""))
    
    if not compatible:
        issues.append((folder, "; ".join(problems)))

# Summary
print()
print("=" * 90)
print("ISSUES FOUND:")
if issues:
    for folder, issue in issues:
        print(f"  {folder}: {issue}")
else:
    print("  None!")

# Check a specific successful rerun folder in detail
print()
print("=" * 90)
print("DETAILED CHECK: CaoMei1 (rerun success)")
print("=" * 90)
base = os.path.join(COLMAP_DIR, "CaoMei1")
sp = os.path.join(base, "sparse", "0")

# Camera
cams = read_cameras_bin(os.path.join(sp, 'cameras.bin'))
for cid, cam in cams.items():
    print(f"  Camera {cid}: {cam['model']} {cam['width']}x{cam['height']}")
    print(f"    Params: {[round(p,2) for p in cam['params']]}")

# Images in images/
img_dir = os.path.join(base, "images")
imgs = sorted(os.listdir(img_dir))[:5]
print(f"  Images in images/: {len(os.listdir(img_dir))} total")
print(f"    First 5: {imgs}")

# Check first image actual content
if imgs:
    first_img = Image.open(os.path.join(img_dir, imgs[0]))
    print(f"    First image size: {first_img.size}, mode: {first_img.mode}")
    arr = np.array(first_img)
    print(f"    Pixel range: min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
    # Is it mostly black (like crop) or full scene (like original)?
    black_pct = (arr.sum(axis=2) == 0).sum() / (arr.shape[0] * arr.shape[1]) * 100
    print(f"    Black pixel %: {black_pct:.1f}% {'(crop-like)' if black_pct > 30 else '(original-like)'}")

# Check input/
inp_dir = os.path.join(base, "input")
inps = sorted(os.listdir(inp_dir))[:5]
print(f"  Images in input/: {len(os.listdir(inp_dir))} total")
print(f"    First 5: {inps}")
if inps:
    first_inp = Image.open(os.path.join(inp_dir, inps[0]))
    print(f"    First input size: {first_inp.size}, mode: {first_inp.mode}")
    arr_i = np.array(first_inp)
    black_pct_i = (arr_i.sum(axis=2) == 0).sum() / (arr_i.shape[0] * arr_i.shape[1]) * 100
    print(f"    Black pixel %: {black_pct_i:.1f}%")

# Also check a successful crop-based folder for comparison
print()
print("=" * 90)
print("DETAILED CHECK: BaiZhang (original crop-based, OK)")
print("=" * 90)
base2 = os.path.join(COLMAP_DIR, "BaiZhang")
img_dir2 = os.path.join(base2, "images")
imgs2 = sorted(os.listdir(img_dir2))[:3]
print(f"  Images in images/: {len(os.listdir(img_dir2))} total")
print(f"    First 3: {imgs2}")
if imgs2:
    first_img2 = Image.open(os.path.join(img_dir2, imgs2[0]))
    print(f"    First image size: {first_img2.size}, mode: {first_img2.mode}")
    arr2 = np.array(first_img2)
    black_pct2 = (arr2.sum(axis=2) == 0).sum() / (arr2.shape[0] * arr2.shape[1]) * 100
    print(f"    Black pixel %: {black_pct2:.1f}% {'(crop-like)' if black_pct2 > 30 else '(original-like)'}")

print()
print("KEY QUESTION: Are undistorted images in images/ originals or crops?")
print("If originals, 2DGS will train on full scene (background included).")
print("If crops, 2DGS trains on plant only.")
