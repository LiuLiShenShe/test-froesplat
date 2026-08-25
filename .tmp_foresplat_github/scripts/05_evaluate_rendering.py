#!/usr/bin/env python3
"""Evaluate foreground rendering metrics for demo-style outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.ones(shape, dtype=bool)
    mask = np.asarray(Image.open(path).convert("L"))
    return mask > 127


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    diff = (a - b)[mask]
    if diff.size == 0:
        return float("nan")
    mse = float(np.mean(diff**2))
    if mse == 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def simple_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = a[mask].reshape(-1)
    y = b[mask].reshape(-1)
    if x.size == 0:
        return float("nan")
    c1 = 0.01**2
    c2 = 0.03**2
    mux, muy = float(x.mean()), float(y.mean())
    vx, vy = float(x.var()), float(y.var())
    cov = float(((x - mux) * (y - muy)).mean())
    return ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_paths = list_images(args.renders)
    rows = []
    for render_path in render_paths:
        ref_path = args.references / render_path.name
        mask_path = args.masks / f"{render_path.stem}.png"
        if not ref_path.exists():
            continue
        render = read_rgb(render_path)
        ref = read_rgb(ref_path)
        mask = read_mask(mask_path, render.shape[:2])
        outside = ~mask
        outside_non_black = float((render[outside].mean(axis=1) > 0.02).mean()) if outside.any() else 0.0
        leakage_energy = float(render[outside].mean()) if outside.any() else 0.0
        rows.append(
            {
                "image": render_path.name,
                "psnr_fg": f"{psnr(render, ref, mask):.4f}",
                "ssim_fg": f"{simple_ssim(render, ref, mask):.4f}",
                "outside_mask_non_black_ratio": f"{outside_non_black:.6f}",
                "leakage_energy_ratio": f"{leakage_energy:.6f}",
            }
        )

    if not rows:
        raise SystemExit("No matching render/reference pairs found.")

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} metric rows to {args.output}")


if __name__ == "__main__":
    main()
