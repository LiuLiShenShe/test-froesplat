#!/usr/bin/env python3
"""Render single-view SfM pictures with camera frustums and denoised sparse points."""

from __future__ import annotations

import argparse
import csv
import math
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# COLMAP model_id -> number of camera params.
CAMERA_MODEL_NUM_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE
    1: 4,   # PINHOLE
    2: 4,   # SIMPLE_RADIAL
    3: 5,   # RADIAL
    4: 8,   # OPENCV
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12, # THIN_PRISM_FISHEYE
}


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def read_cameras_bin(path: Path) -> dict[int, dict[str, object]]:
    cams = {}
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid = struct.unpack("<I", f.read(4))[0]
            mid = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            np_ = CAMERA_MODEL_NUM_PARAMS[mid]
            params = struct.unpack("<" + "d" * np_, f.read(8 * np_))
            cams[cid] = {
                "model_id": int(mid),
                "width": int(width),
                "height": int(height),
                "params": np.asarray(params, dtype=np.float64),
            }
    return cams


def read_images_bin(path: Path) -> dict[int, dict[str, object]]:
    images = {}
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<I", f.read(4))[0]

            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8")

            n_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n_pts2d * 24)
            images[image_id] = {
                "name": name,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": int(camera_id),
            }
    return images


def read_points3d_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            f.read(8)
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    if not xyz:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)


def extract_poses(images: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    poses = []
    for img in images.values():
        r = qvec_to_rotmat(img["qvec"])
        t = img["tvec"]
        c = -r.T @ t
        poses.append({"name": img["name"], "camera_id": img["camera_id"], "R": r, "C": c})
    poses.sort(key=lambda d: d["name"])
    return poses


def camera_focal(cams: dict[int, dict[str, object]], camera_id: int) -> tuple[float, float, float, float]:
    cam = cams[camera_id]
    model_id = cam["model_id"]
    p = cam["params"]
    w = float(cam["width"])
    h = float(cam["height"])
    if model_id in {0, 2, 3, 4, 8, 9, 10}:
        fx = float(p[0])
        fy = float(p[0])
    else:
        fx = float(p[0])
        fy = float(p[1])
    if fx <= 1e-9:
        fx = w
    if fy <= 1e-9:
        fy = h
    return fx, fy, w, h


def build_frustum_segments(poses: list[dict[str, object]], cams: dict[int, dict[str, object]], depth: float) -> np.ndarray:
    segs = []
    for pose in poses:
        fx, fy, w, h = camera_focal(cams, pose["camera_id"])
        xh = depth * (w / (2.0 * fx))
        yh = depth * (h / (2.0 * fy))
        corners_cam = np.array(
            [[xh, yh, depth], [xh, -yh, depth], [-xh, -yh, depth], [-xh, yh, depth]],
            dtype=np.float64,
        )
        c = pose["C"]
        r = pose["R"]
        corners_w = (r.T @ corners_cam.T).T + c[None, :]
        for i in range(4):
            segs.append(np.vstack([c, corners_w[i]]))
        for i in range(4):
            segs.append(np.vstack([corners_w[i], corners_w[(i + 1) % 4]]))
    if not segs:
        return np.zeros((0, 2, 3), dtype=np.float64)
    return np.stack(segs, axis=0)


def set_equal_3d_axes(ax, pts: np.ndarray) -> None:
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    center = (mins + maxs) / 2.0
    r = float(np.max(maxs - mins)) / 2.0
    if not math.isfinite(r) or r <= 0:
        r = 1.0
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def sample_points(xyz: np.ndarray, rgb: np.ndarray, max_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = xyz.shape[0]
    if n <= max_points:
        return xyz, rgb
    idx = rng.choice(n, size=max_points, replace=False)
    return xyz[idx], rgb[idx]


def filter_main_subject_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel_ratio: float,
    min_voxel_points: int,
    min_keep_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n = xyz.shape[0]
    if n == 0:
        return xyz, rgb, {"n_input": 0, "n_output": 0, "keep_ratio": 0.0}
    if n < 400:
        return xyz, rgb, {"n_input": n, "n_output": n, "keep_ratio": 1.0}

    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    span = float(np.max(maxs - mins))
    voxel_size = max(span * voxel_ratio, 1e-9)

    vox = np.floor((xyz - mins[None, :]) / voxel_size).astype(np.int32)
    voxel_points: dict[tuple[int, int, int], list[int]] = {}
    for i, v in enumerate(vox):
        key = (int(v[0]), int(v[1]), int(v[2]))
        voxel_points.setdefault(key, []).append(i)

    dense_voxels = {k for k, ids in voxel_points.items() if len(ids) >= min_voxel_points}
    if not dense_voxels:
        return xyz, rgb, {"n_input": n, "n_output": n, "keep_ratio": 1.0}

    visited = set()
    comps: list[list[tuple[int, int, int]]] = []
    neighbors = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if not (dx == dy == dz == 0)]
    for seed in dense_voxels:
        if seed in visited:
            continue
        stack = [seed]
        visited.add(seed)
        comp = [seed]
        while stack:
            cur = stack.pop()
            cx, cy, cz = cur
            for dx, dy, dz in neighbors:
                nb = (cx + dx, cy + dy, cz + dz)
                if nb in dense_voxels and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
                    comp.append(nb)
        comps.append(comp)

    if not comps:
        return xyz, rgb, {"n_input": n, "n_output": n, "keep_ratio": 1.0}

    def comp_points_count(comp_vox: list[tuple[int, int, int]]) -> int:
        return sum(len(voxel_points[k]) for k in comp_vox)

    largest = max(comps, key=comp_points_count)
    keep_voxels = set(largest)
    keep_idx = []
    for k in keep_voxels:
        keep_idx.extend(voxel_points[k])
    keep_idx = np.asarray(sorted(set(keep_idx)), dtype=np.int64)

    if keep_idx.size == 0:
        return xyz, rgb, {"n_input": n, "n_output": n, "keep_ratio": 1.0}

    xyz_keep = xyz[keep_idx]
    rgb_keep = rgb[keep_idx]

    # Axis-wise robust trimming (MAD) for residual outliers.
    med = np.median(xyz_keep, axis=0)
    mad = np.median(np.abs(xyz_keep - med[None, :]), axis=0)
    mad = np.where(mad < 1e-12, 1e-12, mad)
    z = np.abs(xyz_keep - med[None, :]) / (1.4826 * mad[None, :])
    inlier = np.all(z < 6.0, axis=1)
    if np.any(inlier):
        xyz_keep = xyz_keep[inlier]
        rgb_keep = rgb_keep[inlier]

    keep_ratio = xyz_keep.shape[0] / float(n)
    if keep_ratio < min_keep_ratio:
        return xyz, rgb, {"n_input": n, "n_output": n, "keep_ratio": 1.0}

    return xyz_keep, rgb_keep, {"n_input": n, "n_output": int(xyz_keep.shape[0]), "keep_ratio": float(keep_ratio)}


def render_single_subject_style(
    xyz: np.ndarray,
    rgb: np.ndarray,
    frustum_segments: np.ndarray,
    n_cameras: int,
    n_points: int,
    out_path: Path,
    text_chinese: bool = True,
    show_stats_text: bool = True,
    show_frame_border: bool = True,
    show_axes_box: bool = False,
) -> None:
    fig = plt.figure(figsize=(6.4, 4.6), dpi=220)
    fig.patch.set_facecolor("#ffffff")
    if show_frame_border:
        fig.patch.set_edgecolor("black")
        fig.patch.set_linewidth(1.0)
    else:
        fig.patch.set_edgecolor("#ffffff")
        fig.patch.set_linewidth(0.0)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#ffffff")

    if xyz.shape[0] > 0:
        colors = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=0.35, alpha=0.8, linewidths=0)

    if frustum_segments.shape[0] > 0:
        lc = Line3DCollection(frustum_segments, colors=(1.0, 0.1, 0.02, 0.62), linewidths=0.7)
        ax.add_collection3d(lc)

    pts = []
    if xyz.shape[0] > 0:
        pts.append(xyz)
    if frustum_segments.shape[0] > 0:
        pts.append(frustum_segments.reshape(-1, 3))
    if pts:
        set_equal_3d_axes(ax, np.vstack(pts))

    ax.view_init(elev=9.0, azim=-66.0)
    if show_axes_box:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.grid(False)
    else:
        ax.set_axis_off()
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))

    if show_stats_text:
        if text_chinese:
            text = f"相机: {n_cameras}\n点数: {n_points}"
        else:
            text = f"Cameras: {n_cameras}\nPoints: {n_points}"
        ax.text2D(0.03, 0.95, text, transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold", color="black")

    fig.tight_layout(pad=0.25 if show_frame_border else 0.0)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def discover_models(src_root: Path) -> list[tuple[str, str, Path]]:
    models = []
    for plant_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        sparse_dir = plant_dir / "sparse"
        if not sparse_dir.is_dir():
            continue
        for model_dir in sorted(p for p in sparse_dir.iterdir() if p.is_dir()):
            if (model_dir / "cameras.bin").exists() and (model_dir / "images.bin").exists() and (model_dir / "points3D.bin").exists():
                models.append((plant_dir.name, model_dir.name, model_dir))
    return models


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render single SfM pictures with frustums and denoised main subject points.")
    parser.add_argument("--src-root", type=Path, default=Path("/data/fj/04-COLMAP"))
    parser.add_argument("--out-root", type=Path, default=Path("/data/fj/04-COLMAP-Picture"))
    parser.add_argument("--max-points", type=int, default=150000)
    parser.add_argument("--voxel-ratio", type=float, default=0.01)
    parser.add_argument("--min-voxel-points", type=int, default=2)
    parser.add_argument("--min-keep-ratio", type=float, default=0.03)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--english-text", action="store_true")
    parser.add_argument("--hide-stats-text", action="store_true")
    parser.add_argument("--hide-frame-border", action="store_true")
    parser.add_argument("--show-axes-box", action="store_true")
    args = parser.parse_args()

    src_root = args.src_root
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)
    rows = []

    models = discover_models(src_root)
    for plant, model, model_dir in models:
        out_model = out_root / plant / "sparse" / model
        out_model.mkdir(parents=True, exist_ok=True)

        status = "ok"
        msg = ""
        n_cams = 0
        n_points_total = 0
        n_points_used = 0
        keep_ratio = 0.0

        try:
            cameras = read_cameras_bin(model_dir / "cameras.bin")
            images = read_images_bin(model_dir / "images.bin")
            xyz, rgb = read_points3d_bin(model_dir / "points3D.bin")
            poses = extract_poses(images)
            cams = np.asarray([p["C"] for p in poses], dtype=np.float64) if poses else np.zeros((0, 3), dtype=np.float64)

            xyz_s, rgb_s = sample_points(xyz, rgb, args.max_points, rng)
            xyz_f, rgb_f, filt = filter_main_subject_points(
                xyz=xyz_s,
                rgb=rgb_s,
                voxel_ratio=args.voxel_ratio,
                min_voxel_points=args.min_voxel_points,
                min_keep_ratio=args.min_keep_ratio,
            )

            scene_pts = []
            if xyz_f.shape[0] > 0:
                scene_pts.append(xyz_f)
            if cams.shape[0] > 0:
                scene_pts.append(cams)
            if scene_pts:
                all_pts = np.vstack(scene_pts)
                scene_span = float(np.max(np.max(all_pts, axis=0) - np.min(all_pts, axis=0)))
            else:
                scene_span = 1.0
            depth = max(scene_span * 0.035, 0.01)
            frustums = build_frustum_segments(poses, cameras, depth)

            render_single_subject_style(
                xyz=xyz_f,
                rgb=rgb_f,
                frustum_segments=frustums,
                n_cameras=int(cams.shape[0]),
                n_points=int(xyz_f.shape[0]),
                out_path=out_model / "sfm_subject_clean_view.png",
                text_chinese=(not args.english_text),
                show_stats_text=(not args.hide_stats_text),
                show_frame_border=(not args.hide_frame_border),
                show_axes_box=args.show_axes_box,
            )

            n_cams = int(cams.shape[0])
            n_points_total = int(xyz.shape[0])
            n_points_used = int(xyz_f.shape[0])
            keep_ratio = float(filt["keep_ratio"])
        except Exception as exc:
            status = "failed"
            msg = str(exc)

        rows.append(
            {
                "plant": plant,
                "model": model,
                "status": status,
                "num_registered_images": n_cams,
                "num_points_total": n_points_total,
                "num_points_used_subject": n_points_used,
                "points_keep_ratio_after_filter": keep_ratio,
                "source_model_dir": str(model_dir),
                "output_image": str(out_model / "sfm_subject_clean_view.png"),
                "message": msg,
            }
        )

    write_csv(
        out_root / "sfm_subject_clean_summary.csv",
        rows,
        [
            "plant",
            "model",
            "status",
            "num_registered_images",
            "num_points_total",
            "num_points_used_subject",
            "points_keep_ratio_after_filter",
            "source_model_dir",
            "output_image",
            "message",
        ],
    )

    print(f"[DONE] Rendered subject-clean single views for {len(rows)} sparse models.")
    print(f"[DONE] Summary: {out_root / 'sfm_subject_clean_summary.csv'}")


if __name__ == "__main__":
    main()
