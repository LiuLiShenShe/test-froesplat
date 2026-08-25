#!/usr/bin/env python3
"""Render SfM-stage visualizations from COLMAP sparse models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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
    cameras = {}
    with path.open("rb") as f:
        n_cams = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_cams):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            n_params = CAMERA_MODEL_NUM_PARAMS.get(model_id)
            if n_params is None:
                raise ValueError(f"Unknown camera model_id={model_id} in {path}")
            params = struct.unpack("<" + "d" * n_params, f.read(8 * n_params))
            cameras[camera_id] = {
                "model_id": int(model_id),
                "width": int(width),
                "height": int(height),
                "params": np.asarray(params, dtype=np.float64),
            }
    return cameras


def read_images_bin(path: Path) -> dict[int, dict[str, object]]:
    images = {}
    with path.open("rb") as f:
        n_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_images):
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

            n_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n_points2d * 24)
            images[image_id] = {"qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": name}
    return images


def read_points3d_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz_list: list[tuple[float, float, float]] = []
    rgb_list: list[tuple[int, int, int]] = []
    with path.open("rb") as f:
        n_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_points):
            f.read(8)  # point3D_id
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            f.read(8)  # error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)
            xyz_list.append((x, y, z))
            rgb_list.append((r, g, b))
    if not xyz_list:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(xyz_list, dtype=np.float64), np.asarray(rgb_list, dtype=np.uint8)


def extract_camera_poses(images: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    poses = []
    for img in images.values():
        qvec = img["qvec"]
        tvec = img["tvec"]
        rmat = qvec_to_rotmat(qvec)
        center = -rmat.T @ tvec
        poses.append(
            {
                "name": str(img["name"]),
                "camera_id": int(img["camera_id"]),
                "R": rmat,
                "C": center,
            }
        )
    poses.sort(key=lambda x: x["name"])
    return poses


def camera_focal(cameras: dict[int, dict[str, object]], camera_id: int) -> tuple[float, float, float, float]:
    cam = cameras[camera_id]
    model_id = cam["model_id"]
    w = float(cam["width"])
    h = float(cam["height"])
    p = cam["params"]
    # See COLMAP camera model definitions.
    if model_id in {0, 2, 3, 4, 8, 9, 10}:  # SIMPLE_* and RADIAL/FOV/THIN_PRISM_FISHEYE
        fx = float(p[0])
        fy = float(p[0])
    elif model_id in {1, 5, 6, 7}:  # PINHOLE/OPENCV/OPENCV_FISHEYE/FULL_OPENCV
        fx = float(p[0])
        fy = float(p[1])
    else:
        fx = max(w, 1.0)
        fy = max(h, 1.0)
    fx = fx if fx > 1e-9 else max(w, 1.0)
    fy = fy if fy > 1e-9 else max(h, 1.0)
    return fx, fy, w, h


def build_frustum_segments(
    poses: list[dict[str, object]],
    cameras: dict[int, dict[str, object]],
    frustum_depth: float,
) -> np.ndarray:
    segs: list[np.ndarray] = []
    for pose in poses:
        fx, fy, w, h = camera_focal(cameras, pose["camera_id"])
        xh = frustum_depth * (w / (2.0 * fx))
        yh = frustum_depth * (h / (2.0 * fy))
        corners_cam = np.array(
            [
                [xh, yh, frustum_depth],
                [xh, -yh, frustum_depth],
                [-xh, -yh, frustum_depth],
                [-xh, yh, frustum_depth],
            ],
            dtype=np.float64,
        )
        c = pose["C"]
        r = pose["R"]
        corners_world = (r.T @ corners_cam.T).T + c[None, :]

        for i in range(4):
            segs.append(np.vstack([c, corners_world[i]]))
        for i in range(4):
            j = (i + 1) % 4
            segs.append(np.vstack([corners_world[i], corners_world[j]]))
    if not segs:
        return np.zeros((0, 2, 3), dtype=np.float64)
    return np.stack(segs, axis=0)


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins)) / 2.0
    if not math.isfinite(radius) or radius <= 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def sample_points(xyz: np.ndarray, rgb: np.ndarray, max_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = xyz.shape[0]
    if n <= max_points:
        return xyz, rgb
    idx = rng.choice(n, size=max_points, replace=False)
    return xyz[idx], rgb[idx]


def draw_scene_basic(ax, xyz: np.ndarray, rgb: np.ndarray, cams: np.ndarray) -> None:
    if xyz.shape[0] > 0:
        colors = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=0.4, alpha=0.75, linewidths=0)
    if cams.shape[0] > 0:
        ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], c="red", s=10, marker="^", alpha=0.9)
        if cams.shape[0] >= 2:
            ax.plot(cams[:, 0], cams[:, 1], cams[:, 2], c="red", linewidth=0.5, alpha=0.65)

    pts = []
    if xyz.shape[0] > 0:
        pts.append(xyz)
    if cams.shape[0] > 0:
        pts.append(cams)
    if pts:
        set_equal_3d_axes(ax, np.vstack(pts))


def render_3d_overview(xyz: np.ndarray, rgb: np.ndarray, cams: np.ndarray, out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    draw_scene_basic(ax, xyz, rgb, cams)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_orthographic_views(xyz: np.ndarray, rgb: np.ndarray, cams: np.ndarray, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    view_specs = [(0, 1, "X", "Y", "Top (XY)"), (0, 2, "X", "Z", "Side (XZ)"), (1, 2, "Y", "Z", "Side (YZ)")]

    colors = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0) if xyz.shape[0] > 0 else None
    for ax, (i, j, lx, ly, t) in zip(axes, view_specs):
        if xyz.shape[0] > 0:
            ax.scatter(xyz[:, i], xyz[:, j], c=colors, s=0.4, alpha=0.75, linewidths=0)
        if cams.shape[0] > 0:
            ax.scatter(cams[:, i], cams[:, j], c="red", s=10, marker="^", alpha=0.9)
            if cams.shape[0] >= 2:
                ax.plot(cams[:, i], cams[:, j], c="red", linewidth=0.6, alpha=0.65)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.set_title(t)
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.45)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_scene_frustum_style(
    ax,
    xyz: np.ndarray,
    rgb: np.ndarray,
    frustum_segments: np.ndarray,
    cams_count: int,
    points_count: int,
    elev: float,
    azim: float,
) -> None:
    ax.set_facecolor("#e6e6e6")
    if xyz.shape[0] > 0:
        colors = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=0.35, alpha=0.78, linewidths=0)
    if frustum_segments.shape[0] > 0:
        lc = Line3DCollection(frustum_segments, colors=(1.0, 0.1, 0.02, 0.6), linewidths=0.7)
        ax.add_collection3d(lc)

    pts = []
    if xyz.shape[0] > 0:
        pts.append(xyz)
    if frustum_segments.shape[0] > 0:
        pts.append(frustum_segments.reshape(-1, 3))
    if pts:
        set_equal_3d_axes(ax, np.vstack(pts))

    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    ax.text2D(
        0.03,
        0.95,
        f"Cameras: {cams_count}\nPoints: {points_count}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color="black",
    )


def render_frustum_single_view(
    xyz: np.ndarray,
    rgb: np.ndarray,
    frustum_segments: np.ndarray,
    cams_count: int,
    points_count: int,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(7.8, 5.2), dpi=200)
    fig.patch.set_facecolor("#dcdcdc")
    ax = fig.add_subplot(111, projection="3d")
    draw_scene_frustum_style(
        ax=ax,
        xyz=xyz,
        rgb=rgb,
        frustum_segments=frustum_segments,
        cams_count=cams_count,
        points_count=points_count,
        elev=9.0,
        azim=-66.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_frustum_montage(
    xyz: np.ndarray,
    rgb: np.ndarray,
    frustum_segments: np.ndarray,
    cams_count: int,
    points_count: int,
    out_path: Path,
) -> None:
    views = [(8, -68), (14, -20), (8, 35), (26, -120), (18, 80), (5, 140)]
    fig = plt.figure(figsize=(16, 8.3), dpi=180)
    fig.patch.set_facecolor("#dcdcdc")
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
    for ax, (elev, azim) in zip(axes, views):
        draw_scene_frustum_style(
            ax=ax,
            xyz=xyz,
            rgb=rgb,
            frustum_segments=frustum_segments,
            cams_count=cams_count,
            points_count=points_count,
            elev=float(elev),
            azim=float(azim),
        )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def discover_models(src_root: Path) -> list[tuple[str, str, Path]]:
    models: list[tuple[str, str, Path]] = []
    for plant_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        sparse_dir = plant_dir / "sparse"
        if not sparse_dir.is_dir():
            continue
        for model_dir in sorted(p for p in sparse_dir.iterdir() if p.is_dir()):
            if (model_dir / "images.bin").exists() and (model_dir / "points3D.bin").exists() and (model_dir / "cameras.bin").exists():
                models.append((plant_dir.name, model_dir.name, model_dir))
    return models


def make_gallery_pages(tiles: list[tuple[str, Path]], out_root: Path, per_page: int = 6, ncols: int = 3) -> list[Path]:
    if not tiles:
        return []

    tile_w = 900
    tile_h = 560
    gap = 18
    margin = 24
    nrows = int(math.ceil(per_page / ncols))

    pages = []
    for page_idx in range(int(math.ceil(len(tiles) / per_page))):
        start = page_idx * per_page
        chunk = tiles[start : start + per_page]
        canvas_h = margin * 2 + nrows * tile_h + (nrows - 1) * gap
        canvas_w = margin * 2 + ncols * tile_w + (ncols - 1) * gap
        canvas = np.full((canvas_h, canvas_w, 3), 220, dtype=np.uint8)

        for i, (_label, img_path) in enumerate(chunk):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            row = i // ncols
            col = i % ncols
            x0 = margin + col * (tile_w + gap)
            y0 = margin + row * (tile_h + gap)
            img_r = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = img_r

        out_page = out_root / f"sfm_frustum_gallery_page_{page_idx + 1:02d}.png"
        cv2.imwrite(str(out_page), canvas)
        pages.append(out_page)

    return pages


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SfM visualization pictures from COLMAP sparse models.")
    parser.add_argument("--src-root", type=Path, default=Path("/data/fj/04-COLMAP"))
    parser.add_argument("--out-root", type=Path, default=Path("/data/fj/04-COLMAP-Picture"))
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--random-seed", type=int, default=1234)
    args = parser.parse_args()

    src_root = args.src_root
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)
    rows: list[dict[str, object]] = []
    gallery_tiles: list[tuple[str, Path]] = []

    # Keep plant-level folder naming.
    for plant_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and (p / "sparse").is_dir()):
        (out_root / plant_dir.name).mkdir(parents=True, exist_ok=True)

    models = discover_models(src_root)
    for plant, model_name, model_dir in models:
        out_model_dir = out_root / plant / "sparse" / model_name
        out_model_dir.mkdir(parents=True, exist_ok=True)

        status = "ok"
        msg = ""
        n_cams = 0
        n_points_total = 0
        n_points_plotted = 0
        n_frustum_segments = 0
        generated_files = []

        try:
            cameras = read_cameras_bin(model_dir / "cameras.bin")
            images = read_images_bin(model_dir / "images.bin")
            xyz, rgb = read_points3d_bin(model_dir / "points3D.bin")

            poses = extract_camera_poses(images)
            cams = np.asarray([p["C"] for p in poses], dtype=np.float64) if poses else np.zeros((0, 3), dtype=np.float64)

            xyz_s, rgb_s = sample_points(xyz, rgb, args.max_points, rng)
            scene_pts = []
            if xyz_s.shape[0] > 0:
                scene_pts.append(xyz_s)
            if cams.shape[0] > 0:
                scene_pts.append(cams)
            if scene_pts:
                all_pts = np.vstack(scene_pts)
                scene_span = float(np.max(np.max(all_pts, axis=0) - np.min(all_pts, axis=0)))
            else:
                scene_span = 1.0
            frustum_depth = max(scene_span * 0.035, 0.01)
            frustum_segments = build_frustum_segments(poses, cameras, frustum_depth)

            title = f"{plant} | sparse/{model_name} | cams={cams.shape[0]}, points={xyz.shape[0]}"

            render_3d_overview(xyz_s, rgb_s, cams, out_model_dir / "sfm_3d_overview.png", title)
            generated_files.append("sfm_3d_overview.png")

            render_orthographic_views(xyz_s, rgb_s, cams, out_model_dir / "sfm_orthographic_views.png", title)
            generated_files.append("sfm_orthographic_views.png")

            render_frustum_single_view(
                xyz=xyz_s,
                rgb=rgb_s,
                frustum_segments=frustum_segments,
                cams_count=int(cams.shape[0]),
                points_count=int(xyz.shape[0]),
                out_path=out_model_dir / "sfm_frustum_view.png",
            )
            generated_files.append("sfm_frustum_view.png")

            render_frustum_montage(
                xyz=xyz_s,
                rgb=rgb_s,
                frustum_segments=frustum_segments,
                cams_count=int(cams.shape[0]),
                points_count=int(xyz.shape[0]),
                out_path=out_model_dir / "sfm_frustum_montage.png",
            )
            generated_files.append("sfm_frustum_montage.png")

            meta = {
                "plant": plant,
                "model": model_name,
                "num_registered_images": int(cams.shape[0]),
                "num_points_total": int(xyz.shape[0]),
                "num_points_plotted": int(xyz_s.shape[0]),
                "num_frustum_segments": int(frustum_segments.shape[0]),
                "source_model_dir": str(model_dir),
                "generated_files": generated_files,
            }
            (out_model_dir / "sfm_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            generated_files.append("sfm_summary.json")

            gallery_tiles.append((f"{plant}/sparse/{model_name}", out_model_dir / "sfm_frustum_view.png"))

            n_cams = int(cams.shape[0])
            n_points_total = int(xyz.shape[0])
            n_points_plotted = int(xyz_s.shape[0])
            n_frustum_segments = int(frustum_segments.shape[0])
        except Exception as exc:
            status = "failed"
            msg = str(exc)

        rows.append(
            {
                "plant": plant,
                "model": model_name,
                "status": status,
                "num_registered_images": n_cams,
                "num_points_total": n_points_total,
                "num_points_plotted": n_points_plotted,
                "num_frustum_segments": n_frustum_segments,
                "source_model_dir": str(model_dir),
                "output_model_dir": str(out_model_dir),
                "generated_files": "|".join(generated_files),
                "message": msg,
            }
        )

    gallery_pages = make_gallery_pages(gallery_tiles, out_root=out_root, per_page=6, ncols=3)
    gallery_rows = [{"page_index": i + 1, "page_path": str(p)} for i, p in enumerate(gallery_pages)]

    write_csv(
        out_root / "sfm_visualization_summary.csv",
        rows,
        [
            "plant",
            "model",
            "status",
            "num_registered_images",
            "num_points_total",
            "num_points_plotted",
            "num_frustum_segments",
            "source_model_dir",
            "output_model_dir",
            "generated_files",
            "message",
        ],
    )
    write_csv(out_root / "sfm_frustum_gallery_pages.csv", gallery_rows, ["page_index", "page_path"])

    print(f"[DONE] Rendered {len(rows)} sparse models -> {out_root}")
    print(f"[DONE] Summary: {out_root / 'sfm_visualization_summary.csv'}")
    print(f"[DONE] Gallery pages: {len(gallery_pages)}")


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


if __name__ == "__main__":
    main()
