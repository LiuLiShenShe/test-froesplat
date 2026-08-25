"""
验证 COLMAP 100% 注册文件夹的 3D 重建质量:
1. 点云是否真正三维分布（非退化平面）
2. 相机位姿分布是否合理
3. 重投影误差
4. 是否可用于 2DGS 训练
"""
import struct, os, sys
import numpy as np
from pathlib import Path

BASE = Path(r"D:\CAAS\04-COLMAP")

# ---- Binary readers ----
def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cid = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            # PINHOLE has 4 params
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
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            n2d = struct.unpack("<Q", f.read(8))[0]
            pts2d = []
            for _ in range(n2d):
                x, y = struct.unpack("<2d", f.read(16))
                p3d_id = struct.unpack("<q", f.read(8))[0]
                pts2d.append((x, y, p3d_id))
            images[img_id] = {
                "qvec": (qw, qx, qy, qz),
                "tvec": (tx, ty, tz),
                "cam_id": cam_id,
                "name": name,
                "pts2d": pts2d
            }
    return images

def read_points3D_bin(path):
    points = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            pid = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            err = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            track = []
            for _ in range(track_len):
                im_id = struct.unpack("<i", f.read(4))[0]
                pt2d_idx = struct.unpack("<i", f.read(4))[0]
                track.append((im_id, pt2d_idx))
            points[pid] = {"xyz": (x, y, z), "rgb": (r, g, b), "error": err, "track": track}
    return points

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])

def get_camera_center(qvec, tvec):
    R = qvec2rotmat(qvec)
    t = np.array(tvec)
    return -R.T @ t

# ---- Analysis ----
def analyze_folder(folder_name):
    sparse_dir = BASE / folder_name / "sparse" / "0"
    if not sparse_dir.exists():
        return {"status": "MISSING", "folder": folder_name}
    
    cameras = read_cameras_bin(sparse_dir / "cameras.bin")
    images = read_images_bin(sparse_dir / "images.bin")
    points = read_points3D_bin(sparse_dir / "points3D.bin")
    
    n_reg = len(images)
    n_pts = len(points)
    
    if n_pts < 10 or n_reg < 3:
        return {"status": "TOO_FEW", "folder": folder_name, "n_reg": n_reg, "n_pts": n_pts}
    
    # 1. 相机中心分布
    centers = np.array([get_camera_center(img["qvec"], img["tvec"]) for img in images.values()])
    cam_mean = centers.mean(axis=0)
    cam_std = centers.std(axis=0)
    cam_span = centers.max(axis=0) - centers.min(axis=0)
    
    # PCA on cameras - check if cameras are in 3D or degenerate
    cam_centered = centers - cam_mean
    cov_cam = np.cov(cam_centered.T)
    eig_vals_cam = np.sort(np.linalg.eigvalsh(cov_cam))[::-1]
    # Ratio of smallest to largest eigenvalue indicates dimensionality
    cam_dim_ratio = eig_vals_cam[2] / (eig_vals_cam[0] + 1e-10)
    
    # 2. 点云分布
    pts_xyz = np.array([p["xyz"] for p in points.values()])
    pts_mean = pts_xyz.mean(axis=0)
    pts_std = pts_xyz.std(axis=0)
    pts_span = pts_xyz.max(axis=0) - pts_xyz.min(axis=0)
    
    # PCA on points
    pts_centered = pts_xyz - pts_mean
    # Sample if too many points
    if len(pts_centered) > 10000:
        idx = np.random.choice(len(pts_centered), 10000, replace=False)
        pts_sample = pts_centered[idx]
    else:
        pts_sample = pts_centered
    cov_pts = np.cov(pts_sample.T)
    eig_vals_pts = np.sort(np.linalg.eigvalsh(cov_pts))[::-1]
    pts_dim_ratio = eig_vals_pts[2] / (eig_vals_pts[0] + 1e-10)
    
    # 3. 重投影误差
    errors = [p["error"] for p in points.values()]
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    max_err = np.max(errors)
    
    # 4. Track length statistics
    track_lens = [len(p["track"]) for p in points.values()]
    mean_track = np.mean(track_lens)
    
    # 5. Points per image
    pts_per_img = []
    for img in images.values():
        valid = sum(1 for _, _, pid in img["pts2d"] if pid >= 0)
        pts_per_img.append(valid)
    mean_pts_per_img = np.mean(pts_per_img)
    
    # 判断是否退化
    is_planar_pts = pts_dim_ratio < 0.01  # 点云几乎在一个平面上
    is_planar_cam = cam_dim_ratio < 0.01  # 相机几乎在一个平面上
    
    return {
        "status": "ANALYZED",
        "folder": folder_name,
        "n_reg": n_reg,
        "n_pts": n_pts,
        "cam_span": cam_span,
        "cam_eig": eig_vals_cam,
        "cam_dim_ratio": cam_dim_ratio,
        "pts_span": pts_span,
        "pts_eig": eig_vals_pts,
        "pts_dim_ratio": pts_dim_ratio,
        "mean_reproj_err": mean_err,
        "median_reproj_err": median_err,
        "max_reproj_err": max_err,
        "mean_track_len": mean_track,
        "mean_pts_per_img": mean_pts_per_img,
        "is_planar_pts": is_planar_pts,
        "is_planar_cam": is_planar_cam,
    }

# ---- Main ----
# 选择要分析的文件夹：所有有 sparse/0 的
folders = sorted([d.name for d in BASE.iterdir() if d.is_dir() and (d / "sparse" / "0").exists()])

print(f"{'Folder':<20} {'Reg':>5} {'Pts':>7} {'MeanErr':>8} {'MedErr':>8} {'AvgTrack':>8} {'PtsPerImg':>9} {'PtsSpanXYZ':>30} {'CamSpanXYZ':>30} {'PtsDimR':>8} {'CamDimR':>8} {'3D?':>4}")
print("=" * 180)

ok_folders = []
issues = []

for f in folders:
    r = analyze_folder(f)
    if r["status"] == "MISSING":
        print(f"{f:<20} ** sparse/0 NOT FOUND **")
        continue
    if r["status"] == "TOO_FEW":
        print(f"{f:<20} {r['n_reg']:>5} {r['n_pts']:>7}  ** TOO FEW FOR ANALYSIS **")
        issues.append(f"{f}: only {r['n_reg']} images, {r['n_pts']} points")
        continue
    
    pts_span_str = f"({r['pts_span'][0]:.2f}, {r['pts_span'][1]:.2f}, {r['pts_span'][2]:.2f})"
    cam_span_str = f"({r['cam_span'][0]:.2f}, {r['cam_span'][1]:.2f}, {r['cam_span'][2]:.2f})"
    
    is_3d = not r["is_planar_pts"] and not r["is_planar_cam"]
    tag = "YES" if is_3d else "NO!"
    
    print(f"{f:<20} {r['n_reg']:>5} {r['n_pts']:>7} {r['mean_reproj_err']:>8.3f} {r['median_reproj_err']:>8.3f} {r['mean_track_len']:>8.1f} {r['mean_pts_per_img']:>9.0f} {pts_span_str:>30} {cam_span_str:>30} {r['pts_dim_ratio']:>8.4f} {r['cam_dim_ratio']:>8.4f} {tag:>4}")
    
    if is_3d and r['n_pts'] >= 100:
        ok_folders.append(f)
    else:
        reason = []
        if r["is_planar_pts"]:
            reason.append("pts_planar")
        if r["is_planar_cam"]:
            reason.append("cam_planar")
        if r['n_pts'] < 100:
            reason.append(f"low_pts:{r['n_pts']}")
        issues.append(f"{f}: {', '.join(reason)}")

print("\n" + "=" * 80)
print(f"✅ 3D OK ({len(ok_folders)}): {', '.join(ok_folders)}")
if issues:
    print(f"❌ Issues ({len(issues)}):")
    for iss in issues:
        print(f"   {iss}")

# 额外打印：对比原始crop文件夹(OK)和rerun文件夹(100%)的点云规模
print("\n" + "=" * 80)
print("详细对比 - 重建质量:")
crop_folders = ["BaiZhang","DouBanLv3","KongQueZhuYu","WangWenCao1","XianKeLai1","XianKeLai3","XiangPiShu1"]
rerun_ok = ["CaoMei1","ChangShouHua2","DouBanLv1","DouBanLv2","HongZhang","WanNianQing1","WanNianQing2","WangWenCao2","XianKeLai2"]

print(f"\n--- Crop图直接成功的 (基准) ---")
for f in crop_folders:
    r = analyze_folder(f)
    if r["status"] == "ANALYZED":
        print(f"  {f:<20} {r['n_reg']:>5} imgs, {r['n_pts']:>7} pts, err={r['mean_reproj_err']:.3f}, track={r['mean_track_len']:.1f}, dimR_pts={r['pts_dim_ratio']:.4f}, dimR_cam={r['cam_dim_ratio']:.4f}")

print(f"\n--- 用原图重跑并mask过滤的 ---")
for f in rerun_ok:
    r = analyze_folder(f)
    if r["status"] == "ANALYZED":
        print(f"  {f:<20} {r['n_reg']:>5} imgs, {r['n_pts']:>7} pts, err={r['mean_reproj_err']:.3f}, track={r['mean_track_len']:.1f}, dimR_pts={r['pts_dim_ratio']:.4f}, dimR_cam={r['cam_dim_ratio']:.4f}")
    elif r["status"] in ("MISSING","TOO_FEW"):
        print(f"  {f:<20} ** {r['status']} **")
