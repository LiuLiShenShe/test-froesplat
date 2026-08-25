#!/usr/bin/env python3
"""Prepare a compact image/mask manifest for a COLMAP or COLMAP-converted backend."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    rows = []
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        matching_masks = sorted(args.mask_dir.glob(f"{image_path.stem.split('_')[0]}*.png"))
        rows.append(
            {
                "image": str(image_path),
                "width": width,
                "height": height,
                "available_masks": ";".join(str(p) for p in matching_masks),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "width", "height", "available_masks"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} manifest rows to {args.output}")


if __name__ == "__main__":
    main()
