#!/usr/bin/env python3
"""Validate that a real RGB image and foreground mask are spatially aligned."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image)
    mask = Image.open(args.mask).convert("L")
    if image.size != mask.size:
        raise SystemExit(f"Size mismatch: image={image.size}, mask={mask.size}")
    arr = np.asarray(mask)
    foreground_ratio = float((arr > 127).mean())
    print(f"image: {args.image} {image.size} {image.mode}")
    print(f"mask:  {args.mask} {mask.size} foreground_ratio={foreground_ratio:.6f}")


if __name__ == "__main__":
    main()
