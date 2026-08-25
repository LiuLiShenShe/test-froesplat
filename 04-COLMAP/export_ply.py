"""Convert COLMAP sparse points3D.bin to PLY for visualization."""
import struct
import os

def read_points3D_bin(path):
    points = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            pid = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<ddd", f.read(24))
            r, g, b = struct.unpack("<BBB", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            n_track = struct.unpack("<Q", f.read(8))[0]
            f.read(n_track * 8)  # skip track entries (image_id + point2D_idx)
            points.append((x, y, z, r, g, b))
    return points

def write_ply(points, out_path):
    with open(out_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

BASE = "D:/CAAS/04-COLMAP"
for folder in ["DouBanLv1", "DouBanLv2", "DouBanLv3"]:
    pts_bin = os.path.join(BASE, folder, "sparse", "0", "points3D.bin")
    out_ply = os.path.join(BASE, folder, "sparse", "0", "points3D.ply")
    if not os.path.exists(pts_bin):
        print(f"{folder}: points3D.bin not found, skip")
        continue
    points = read_points3D_bin(pts_bin)
    write_ply(points, out_ply)
    print(f"{folder}: {len(points)} points -> {out_ply}")
