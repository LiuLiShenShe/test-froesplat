#!/usr/bin/env python3
"""Prepare camera and foreground-track manifests for ForeSplat."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def mask_centroid(mask_path: Path) -> tuple[float, float, int]:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    yy, xx = np.nonzero(mask > 127)
    if len(xx) == 0:
        return 0.5, 0.5, 0
    return float(xx.mean() / mask.shape[1]), float(yy.mean() / mask.shape[0]), int(len(xx))


def camera_pose(index: int, total: int) -> list[list[float]]:
    angle = 2.0 * math.pi * index / max(total, 1)
    radius = 1.5
    x = radius * math.cos(angle)
    z = radius * math.sin(angle)
    return [
        [math.cos(angle), 0.0, -math.sin(angle), x],
        [0.0, 1.0, 0.0, 0.0],
        [math.sin(angle), 0.0, math.cos(angle), z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(args.image_dir)
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")

    frames = []
    track_rows = []
    for idx, image_path in enumerate(images):
        with Image.open(image_path) as image:
            width, height = image.size
        mask_path = args.mask_dir / f"{image_path.stem}.png"
        cx, cy, support = mask_centroid(mask_path) if mask_path.exists() else (0.5, 0.5, 0)
        frames.append(
            {
                "file_path": f"../images/{image_path.name}",
                "mask_path": f"../masks/{image_path.stem}.png",
                "w": width,
                "h": height,
                "fl_x": 0.85 * width,
                "fl_y": 0.85 * width,
                "cx": width / 2.0,
                "cy": height / 2.0,
                "transform_matrix": camera_pose(idx, len(images)),
            }
        )
        track_rows.append(
            {
                "frame_id": idx,
                "image": image_path.name,
                "foreground_centroid_x_norm": f"{cx:.6f}",
                "foreground_centroid_y_norm": f"{cy:.6f}",
                "foreground_pixel_support": support,
            }
        )

    transforms = {
        "camera_model": "demo_pinhole",
        "source": "synthetic_or_colmap_converted_manifest",
        "frames": frames,
    }
    (args.output_dir / "transforms_demo.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")

    with (args.output_dir / "foreground_tracks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(track_rows[0].keys()))
        writer.writeheader()
        writer.writerows(track_rows)

    print(f"Wrote camera manifest and foreground tracks to {args.output_dir}")


if __name__ == "__main__":
    main()
