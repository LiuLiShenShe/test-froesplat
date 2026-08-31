#!/usr/bin/env python3
"""Split 2x3 visualization grid images into individual panel images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

PANEL_NAMES = [
    "01_original",
    "02_gt",
    "03_sam",
    "04_seem",
    "05_sam_error",
    "06_seem_error",
]


def split_grid_image(grid_path: Path, out_dir: Path) -> bool:
    img = cv2.imread(str(grid_path), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    if h % 2 != 0 or w % 3 != 0:
        return False

    panel_h = h // 2
    panel_w = w // 3

    panels = [
        img[0:panel_h, 0:panel_w],
        img[0:panel_h, panel_w : 2 * panel_w],
        img[0:panel_h, 2 * panel_w : 3 * panel_w],
        img[panel_h : 2 * panel_h, 0:panel_w],
        img[panel_h : 2 * panel_h, panel_w : 2 * panel_w],
        img[panel_h : 2 * panel_h, 2 * panel_w : 3 * panel_w],
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, panel in zip(PANEL_NAMES, panels):
        cv2.imwrite(str(out_dir / f"{name}.jpg"), panel)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Split visualization grid JPGs to panel JPGs.")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation directory, e.g. /data/fj/03-GT/eval_20260303_163457",
    )
    parser.add_argument(
        "--input-subdir",
        type=str,
        default="visualizations",
        help="Subdirectory in each plant folder storing grid images.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="visualization_panels",
        help="Subdirectory in each plant folder to store split panels.",
    )
    args = parser.parse_args()

    eval_dir: Path = args.eval_dir
    plants = sorted(d for d in eval_dir.iterdir() if d.is_dir())

    total_ok = 0
    total_fail = 0

    for plant_dir in plants:
        in_dir = plant_dir / args.input_subdir
        if not in_dir.is_dir():
            continue

        out_root = plant_dir / args.output_subdir
        grids = sorted(in_dir.glob("*.jpg"))
        for grid_path in grids:
            frame = grid_path.stem
            ok = split_grid_image(grid_path, out_root / frame)
            if ok:
                total_ok += 1
            else:
                total_fail += 1

    print(f"[DONE] split ok={total_ok}, failed={total_fail}, eval_dir={eval_dir}")


if __name__ == "__main__":
    main()
