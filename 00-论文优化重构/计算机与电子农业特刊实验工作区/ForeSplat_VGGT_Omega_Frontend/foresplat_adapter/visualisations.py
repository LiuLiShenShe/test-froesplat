"""Per-plant visualisations for the full-sequence pipeline.

Each figure is written into ``<plant_output>/visualizations/``:

=========================  =================================================
file                       content
=========================  =================================================
rgb_preview.png            RGB grid of sampled frames
mask_overlay.png           foreground mask overlay for the same frames
depth_map.png              depth maps (viridis)
confidence_map.png         depth-confidence maps (magma)
camera_distribution.png    camera centres of the full sequence (3-D)
pointcloud_rgb.png         fused cloud in top / side / back views (RGB)
raw_vs_fused.png           raw per-view cloud vs fused cloud (top view)
fused_pointcloud.ply       the fused cloud (interactive PLY for viewers)
fused_pointcloud.html      interactive 3-D scatter (if plotly is available)
=========================  =================================================
"""

from __future__ import annotations

import importlib.util
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["write_all_visualisations", "read_ply_cloud"]


def _sample_indices(num_frames: int, count: int = 6) -> list[int]:
    if num_frames <= count:
        return list(range(num_frames))
    return sorted({int(round(i * (num_frames - 1) / (count - 1))) for i in range(count)})


def _grid(
    frames: list[np.ndarray],
    titles: list[str],
    path: str,
    cmap: str | None = None,
    columns: int = 3,
    overlay_mask: list[np.ndarray] | None = None,
) -> None:
    rows = (len(frames) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 2.8 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for k, frame in enumerate(frames):
        ax = axes[k // columns][k % columns]
        if overlay_mask is not None:
            base = np.asarray(frame, dtype=np.float32).copy()
            overlay = np.asarray(overlay_mask[k], dtype=np.float32)
            base[overlay > 0.5] = base[overlay > 0.5] * 0.35 + np.array([0.1, 0.85, 0.2]) * 0.65
            ax.imshow(np.clip(base, 0, 1))
        elif cmap:
            ax.imshow(frame, cmap=cmap)
        else:
            ax.imshow(frame)
        ax.set_title(titles[k], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _camera_distribution(extrinsics: np.ndarray, path: str) -> None:
    centers = -np.asarray(extrinsics[:, :3, :3]) @ np.asarray(extrinsics[:, :3, 3])[..., None]
    centers = centers[..., 0]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=np.arange(len(centers)), cmap="plasma", s=6)
    for k in range(0, len(centers), max(1, len(centers) // 40)):
        R = extrinsics[k, :3, :3]
        look = R[:3, 2]
        ax.plot(*zip(centers[k], centers[k] + look * 0.05), color="k", lw=0.5)
    ax.set_title(f"camera centres (N={len(centers)})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _point_cloud_views(xyz: np.ndarray, rgb: np.ndarray, path: str, title: str, max_points: int = 200_000) -> None:
    if len(xyz) > max_points:
        pick = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[pick], rgb[pick]
    views = [
        ("top (x,y)", 90, -90),
        ("side (x,z)", 0, -90),
        ("back (y,z)", 0, 0),
    ]
    fig = plt.figure(figsize=(12, 4.4))
    for k, (name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb / 255.0, s=0.4, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _raw_vs_fused(raw_xyz: np.ndarray, raw_rgb: np.ndarray, fused_xyz: np.ndarray, fused_rgb: np.ndarray, path: str):
    fig = plt.figure(figsize=(12, 5))
    for k, (xyz, rgb, title) in enumerate(
        [
            (raw_xyz, raw_rgb, f"raw per-view points (N={len(raw_xyz)})"),
            (fused_xyz, fused_rgb, f"fused points (N={len(fused_xyz)})"),
        ]
    ):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        cap = 150_000
        if len(xyz) > cap:
            pick = np.random.default_rng(0).choice(len(xyz), cap, replace=False)
            xyz, rgb = xyz[pick], rgb[pick]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb / 255.0, s=0.4, depthshade=False)
        ax.view_init(elev=90, azim=-90)
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _interactive_html(xyz: np.ndarray, rgb: np.ndarray, path: str, max_points: int = 120_000) -> bool:
    if importlib.util.find_spec("plotly") is None:
        return False
    import plotly.graph_objects as go  # type: ignore

    if len(xyz) > max_points:
        pick = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[pick], rgb[pick]
    colors = ["#%02x%02x%02x" % tuple(int(c) for c in rgb[i]) for i in range(len(xyz))]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                marker=dict(size=1.2, color=colors), text=colors, hoverinfo="x+y+z",
            )
        ]
    )
    fig.write_html(path)
    return True


def read_ply_cloud(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Minimal binary-LE PLY reader honouring the header property order."""
    type_map = {
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        "uchar": "u1", "uint8": "u1", "ushort": "<u2", "uint16": "<u2",
        "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    }
    with open(path, "rb") as handle:
        count = 0
        fields: list[tuple[str, str]] = []
        while True:
            line = handle.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property"):
                _, ptype, pname = line.split()
                fields.append((pname, type_map[ptype]))
            elif line == "end_header":
                break
        dtype = np.dtype(fields)
        data = np.frombuffer(handle.read(count * dtype.itemsize), dtype=dtype)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.float32)
    return xyz, rgb


def write_all_visualisations(
    visual_dir: str,
    dataset_dir: str,
    per_frame_dir: str,
    camera_meta: dict,
    raw_ply: str,
    fused_ply: str,
    sample_count: int = 6,
) -> list[str]:
    """Write every per-plant figure; returns the list of files created."""
    os.makedirs(visual_dir, exist_ok=True)
    created: list[str] = []

    cameras = camera_meta["frames"]
    model_h, model_w = camera_meta["model_height"], camera_meta["model_width"]
    indices = _sample_indices(len(cameras), sample_count)
    stems = [cameras[i]["stem"] for i in indices]

    def load_npy(kind: str) -> list[np.ndarray]:
        out = []
        for stem in stems:
            out.append(np.load(os.path.join(per_frame_dir, f"{kind}_{stem}.npy")))
        return out

    rgb_frames = [np.load(os.path.join(per_frame_dir, f"model_rgb_{s}.npy")).astype(np.float32) / 255.0 for s in stems]
    mask_frames = [np.load(os.path.join(per_frame_dir, f"model_mask_{s}.npy")).astype(np.float32) for s in stems]
    depth_frames = load_npy("depth")
    conf_frames = load_npy("confidence")

    written = {
        "rgb_preview": "rgb_preview.png",
        "mask_overlay": "mask_overlay.png",
        "depth_map": "depth_map.png",
        "confidence_map": "confidence_map.png",
    }
    _grid(rgb_frames, stems, os.path.join(visual_dir, written["rgb_preview"]))
    _grid(rgb_frames, stems, os.path.join(visual_dir, written["mask_overlay"]), overlay_mask=mask_frames)
    _grid(depth_frames, stems, os.path.join(visual_dir, written["depth_map"]), cmap="viridis")
    _grid(conf_frames, stems, os.path.join(visual_dir, written["confidence_map"]), cmap="magma")
    created.extend(written.values())

    extrinsics = np.array([cam["extrinsic_model_w2c"] for cam in cameras], dtype=np.float64)
    _camera_distribution(extrinsics, os.path.join(visual_dir, "camera_distribution.png"))
    created.append("camera_distribution.png")

    fused_xyz, fused_rgb = read_ply_cloud(fused_ply)
    _point_cloud_views(fused_xyz, fused_rgb, os.path.join(visual_dir, "pointcloud_rgb.png"),
                       f"fused cloud (N={len(fused_xyz)})")
    created.append("pointcloud_rgb.png")

    if os.path.isfile(raw_ply):
        raw_xyz, raw_rgb = read_ply_cloud(raw_ply)
        _raw_vs_fused(raw_xyz, raw_rgb, fused_xyz, fused_rgb,
                      os.path.join(visual_dir, "raw_vs_fused.png"))
        created.append("raw_vs_fused.png")

    if _interactive_html(fused_xyz, fused_rgb, os.path.join(visual_dir, "fused_pointcloud.html")):
        created.append("fused_pointcloud.html")
    return created
