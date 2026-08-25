#!/usr/bin/env python3
"""Inspect the real non-full ForeSplat data subset."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def describe_image(path: Path) -> str:
    with Image.open(path) as image:
        return f"{image.width}x{image.height} {image.mode}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("examples"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.root.exists():
        raise SystemExit(f"Root does not exist: {args.root}")

    files = sorted(p for p in args.root.rglob("*") if p.is_file())
    print(f"root: {args.root}")
    print(f"files: {len(files)}")
    for path in files:
        rel = path.relative_to(args.root)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            print(f"IMAGE {rel} {path.stat().st_size} bytes {describe_image(path)}")
        else:
            print(f"DATA  {rel} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
