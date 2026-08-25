#!/usr/bin/env python3
"""Prepare KongQueZhuYu GT masks and selected input frames for experiments 2/3."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def shape_to_mask(shape: dict, h: int, w: int) -> np.ndarray:
    points = shape.get("points", [])
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 3:
        return mask
    pts = np.asarray(points, dtype=np.float32)
    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def convert_labelme(json_path: Path) -> tuple[np.ndarray, dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    h = int(data["imageHeight"])
    w = int(data["imageWidth"])
    mask = np.zeros((h, w), dtype=np.uint8)
    shape_rows = []
    for idx, shape in enumerate(data.get("shapes", []), start=1):
        sm = shape_to_mask(shape, h, w)
        mask = np.maximum(mask, sm)
        pts = shape.get("points", [])
        close_dist = ""
        if len(pts) >= 2:
            p0 = np.asarray(pts[0], dtype=np.float32)
            p1 = np.asarray(pts[-1], dtype=np.float32)
            close_dist = float(np.linalg.norm(p0 - p1))
        shape_rows.append(
            {
                "shape_index": idx,
                "label": shape.get("label", ""),
                "shape_type": shape.get("shape_type", ""),
                "points": len(pts),
                "close_distance_px": close_dist,
                "area_px": int((sm > 0).sum()),
            }
        )
    return mask, {"width": w, "height": h, "shapes": shape_rows}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_json_dir", type=Path, default=Path("03-GT/KongQueZhuYu"))
    parser.add_argument(
        "--source_image_dir",
        type=Path,
        default=Path("00-论文优化重构/数据管理/01-输入图像/02-fft_frames/KongQueZhuYu"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("00-论文优化重构/数据管理/05-评测结果/S21_KongQueZhuYu_E2_E3"),
    )
    args = parser.parse_args()

    gt_mask_dir = args.output_root / "gt_masks"
    selected_dir = args.output_root / "selected_frames"
    gt_mask_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    shape_rows = []
    for json_path in sorted(args.gt_json_dir.glob("*.json")):
        stem = json_path.stem
        source_image = args.source_image_dir / f"{stem}.jpg"
        if not source_image.exists():
            raise FileNotFoundError(f"Source image not found: {source_image}")
        mask, meta = convert_labelme(json_path)
        mask_path = gt_mask_dir / f"mask_{stem}.png"
        cv2.imwrite(str(mask_path), mask)
        selected_image = selected_dir / f"{stem}.jpg"
        if not selected_image.exists():
            shutil.copy2(source_image, selected_image)
        index_rows.append(
            {
                "frame": stem,
                "source_image": str(source_image),
                "selected_image": str(selected_image),
                "gt_json": str(json_path),
                "gt_mask": str(mask_path),
                "width": meta["width"],
                "height": meta["height"],
                "gt_area_ratio": float((mask > 0).sum() / mask.size),
                "num_shapes": len(meta["shapes"]),
            }
        )
        for row in meta["shapes"]:
            shape_rows.append({"frame": stem, **row})

    write_csv(
        args.output_root / "gt_index.csv",
        index_rows,
        [
            "frame",
            "source_image",
            "selected_image",
            "gt_json",
            "gt_mask",
            "width",
            "height",
            "gt_area_ratio",
            "num_shapes",
        ],
    )
    write_csv(
        args.output_root / "gt_shape_summary.csv",
        shape_rows,
        ["frame", "shape_index", "label", "shape_type", "points", "close_distance_px", "area_px"],
    )
    print(f"Prepared {len(index_rows)} GT frames under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
