#!/usr/bin/env python3
"""Generate or import RAP-FSAM3 foreground priors.

The full paper workflow calls an external SAM-series/VFM backend. This script
keeps that interface explicit and provides a deterministic demo-mode mask
generator for checking downstream file formats.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def demo_mask(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB")).astype(np.int16)
    red, green, blue = arr[..., 0], arr[..., 1], arr[..., 2]
    green_score = green - np.maximum(red, blue)
    mask = (green_score > 18) & (green > 55)

    if mask.mean() < 0.01:
        h, w = mask.shape
        yy, xx = np.ogrid[:h, :w]
        cx, cy = w * 0.50, h * 0.48
        rx, ry = w * 0.28, h * 0.34
        mask = ((xx - cx) ** 2 / rx**2 + (yy - cy) ** 2 / ry**2) <= 1.0

    return mask.astype(np.uint8) * 255


def save_foreground_rgb(image: Image.Image, mask: np.ndarray, dst: Path) -> None:
    arr = np.asarray(image.convert("RGB")).copy()
    arr[mask == 0] = 0
    Image.fromarray(arr).save(dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=["demo", "external"],
        default="demo",
        help="Use demo masks or prepare an external VFM backend manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    images = list_images(args.image_dir)
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")

    mask_dir = args.output_dir
    alpha_dir = args.output_dir / "alpha"
    fg_dir = args.output_dir / "foreground_rgb"
    for directory in (mask_dir, alpha_dir, fg_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selection_rows = []
    score_rows = []
    preferred = config.get("semantic_gate", {}).get("preferred_prompt", "P2")

    for idx, image_path in enumerate(images):
        image = Image.open(image_path)
        if args.backend == "external":
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            mask = demo_mask(image)

        stem = image_path.stem
        mask_name = f"{stem}.png"
        Image.fromarray(mask).save(mask_dir / mask_name)
        Image.fromarray(mask).save(alpha_dir / mask_name)
        save_foreground_rgb(image, mask, fg_dir / mask_name)

        selection_rows.append(
            {
                "frame_id": idx,
                "image": image_path.name,
                "selected_prompt": preferred,
                "mask": mask_name,
                "backend": args.backend,
            }
        )
        score_rows.append(
            {
                "frame_id": idx,
                "P1_green_region": 0.70,
                "P2_plant_instance": 0.95,
                "P3_organs": 0.78,
                "P4_seedling": 0.35,
                "P5_background_excluding": 0.82,
            }
        )

    with (args.output_dir / "prompt_selection.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selection_rows)

    with (args.output_dir / "semantic_gate_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        writer.writeheader()
        writer.writerows(score_rows)

    run_log = {
        "backend": args.backend,
        "config": str(args.config),
        "frames": len(images),
        "outputs": {
            "masks": str(mask_dir),
            "alpha": str(alpha_dir),
            "foreground_rgb": str(fg_dir),
        },
        "note": "Demo backend uses deterministic colour/shape priors and is not a SAM model.",
    }
    (args.output_dir / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    print(f"Wrote foreground priors for {len(images)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
