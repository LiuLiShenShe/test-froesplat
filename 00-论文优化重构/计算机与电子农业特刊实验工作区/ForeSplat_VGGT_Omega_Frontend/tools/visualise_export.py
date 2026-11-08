#!/usr/bin/env python3
"""Visualise a ForeSplat GGGS export: foreground point cloud + mask overlay previews.

Usage::

    python tools/visualise_export.py --dataset-dir GGGS_Dataset [--max-points 80000]
"""

from __future__ import annotations

import argparse
import os
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, sys_path)


def read_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary PLY point cloud, honouring the properties declared in its header."""
    type_map = {
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        "uchar": "u1", "uint8": "u1", "int": "<i4", "int32": "<i4",
    }
    with open(path, "rb") as handle:
        header_lines = []
        while True:
            line = handle.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        payload = handle.read()

    count = 0
    fields = []
    for line in header_lines:
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
        elif line.startswith("property"):
            _, ptype, pname = line.split()
            fields.append((pname, type_map[ptype]))
    dtype = np.dtype(fields)
    arr = np.frombuffer(payload[: count * dtype.itemsize], dtype=dtype)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    rgb = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1)
    return xyz, rgb


def render_pointcloud(xyz: np.ndarray, rgb: np.ndarray, out_path: str) -> None:
    fig = plt.figure(figsize=(12, 6))
    views = [(0, 90, "front view"), (30, 30, "oblique view"), (90, 0, "side view")]
    for k, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, k, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb / 255.0, s=0.6, depthshade=False)
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_mask_overlay(images_dir: str, masks_dir: str, out_path: str, num: int = 4) -> None:
    names = sorted(os.listdir(images_dir))[:num]
    if not names:
        return
    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.4))
    if len(names) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])
    for k, name in enumerate(names):
        stem = os.path.splitext(name)[0]
        img = Image.open(os.path.join(images_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(masks_dir, stem + ".png")).convert("L")
        img_small = img.resize((640, int(640 * img.height / img.width)))
        mask_small = mask.resize(img_small.size)
        overlay = np.asarray(img_small).astype(np.float32) * 0.55 + (
            np.stack([np.asarray(mask_small)] * 3, axis=-1).astype(np.float32) * 0.45
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        axes[0][k].imshow(np.asarray(img_small))
        axes[0][k].set_title(f"{name} (foreground RGB)")
        axes[0][k].set_axis_off()
        axes[1][k].imshow(overlay)
        axes[1][k].set_title("mask overlay")
        axes[1][k].set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ForeSplat export visualisation")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--max-points", type=int, default=80000)
    args = parser.parse_args(argv)

    debug_dir = os.path.join(args.dataset_dir, "debug")
    ply_path = os.path.join(debug_dir, "foreground_pointcloud.ply")
    xyz, rgb = read_ply(ply_path)
    if xyz.shape[0] > args.max_points:
        pick = np.linspace(0, xyz.shape[0] - 1, args.max_points).astype(np.int64)
        xyz, rgb = xyz[pick], rgb[pick]
    render_pointcloud(xyz, rgb, os.path.join(debug_dir, "preview_pointcloud.png"))

    images_dir = os.path.join(args.dataset_dir, "images")
    masks_dir = os.path.join(args.dataset_dir, "masks")
    render_mask_overlay(images_dir, masks_dir, os.path.join(debug_dir, "preview_mask_overlay.png"))

    print(f"pointcloud preview: {debug_dir}/preview_pointcloud.png ({xyz.shape[0]} sampled pts)")
    print(f"mask overlay preview: {debug_dir}/preview_mask_overlay.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
