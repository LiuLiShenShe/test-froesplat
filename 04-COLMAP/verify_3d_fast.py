"""快速验证3D质量 - 简化版"""
import struct, os, sys
import numpy as np
from pathlib import Path

BASE = Path(r"D:\CAAS\04-COLMAP")

def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cid = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            num_params = {0:3,1:4,2:4,3:5,4:4,5:5,6:4,7:8,8:12,9:13}.get(model_id,4)
            params = struct.unpack(f"<{num_params}d", f.read(8*num_params))
            cameras[cid] = {"model_id": model_id, "w": w, "h": h, "params": params}
    return cameras

def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name_bytes += c
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n2d * 24)  # skip 2d points for speed
            images[img_id] = {"qvec": (qw,qx,qy,qz), "tvec": (tx,ty,tz), "cam_id": cam_id, "name": name_bytes.decode()}
    return images

def read_points3D_bin(path):
    pts = []
    errs = []
    tracks = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            f.read(8)  # pid
            x,y,z = struct.unpack("<3d", f.read(24))
            f.read(3)  # rgb
            err = struct.unpack("<d", f.read(8))[0]
            tlen = struct.unpack("<Q", f.read(8))[0]
            f.read(tlen * 8)  # skip track
            pts.append((x,y,z))
            errs.append(err)
            tracks.append(tlen)
    return np.array(pts) if pts else np.zeros((0,3)), np.array(errs), np.array(tracks)

def qvec2rotmat(q):
    w,x,y,z = q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],[2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],[2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]])

folders = sorted([d.name for d in BASE.iterdir() if d.is_dir() and (d/"sparse"/"0").exists()])

print(f"{'Folder':<20} {'Reg':>4} {'Pts':>7} {'MErr':>6} {'TrkLen':>6} {'PtsSpan(X,Y,Z)':>28} {'CamSpan(X,Y,Z)':>28} {'PtDimR':>7} {'CmDimR':>7} {'3D':>3}")
print("="*140)

for fname in folders:
    sp = BASE/fname/"sparse"/"0"
    images = read_images_bin(sp/"images.bin")
    pts_xyz, errs, tracks = read_points3D_bin(sp/"points3D.bin")
    
    n_reg = len(images)
    n_pts = len(pts_xyz)
    
    if n_pts < 10 or n_reg < 3:
        print(f"{fname:<20} {n_reg:>4} {n_pts:>7}  ** TOO FEW **")
        continue
    
    # Camera centers
    centers = []
    for img in images.values():
        R = qvec2rotmat(img["qvec"])
        t = np.array(img["tvec"])
        centers.append(-R.T @ t)
    centers = np.array(centers)
    cam_span = centers.max(0) - centers.min(0)
    cov_c = np.cov((centers - centers.mean(0)).T)
    eig_c = np.sort(np.linalg.eigvalsh(cov_c))[::-1]
    cdim = eig_c[2]/(eig_c[0]+1e-10)
    
    # Points PCA
    if n_pts > 5000:
        idx = np.random.choice(n_pts, 5000, replace=False)
        ps = pts_xyz[idx]
    else:
        ps = pts_xyz
    pts_span = pts_xyz.max(0) - pts_xyz.min(0)
    cov_p = np.cov((ps - ps.mean(0)).T)
    eig_p = np.sort(np.linalg.eigvalsh(cov_p))[::-1]
    pdim = eig_p[2]/(eig_p[0]+1e-10)
    
    is3d = pdim > 0.01 and cdim > 0.01
    tag = "YES" if is3d else "NO!"
    
    print(f"{fname:<20} {n_reg:>4} {n_pts:>7} {errs.mean():>6.2f} {tracks.mean():>6.1f} ({pts_span[0]:>7.2f},{pts_span[1]:>7.2f},{pts_span[2]:>7.2f}) ({cam_span[0]:>7.2f},{cam_span[1]:>7.2f},{cam_span[2]:>7.2f}) {pdim:>7.4f} {cdim:>7.4f} {tag:>3}")

print("\n判断标准: PtDimR/CmDimR > 0.01 表示三维分布(非退化平面), 越大越好")
print("MErr = 平均重投影误差(像素), TrkLen = 平均轨迹长度(每个3D点被多少张图看到)")
