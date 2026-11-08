#!/usr/bin/env python3
"""Build the full-sequence batch manifest for all plant scene directories.

Example::

    python tools/build_batch_manifest.py \
        --root /data/fj/F2DMAS/03-SAM \
        --output-root runs/full_sequence_batch \
        --manifest runs/full_sequence_batch/batch_manifest.csv

Frames whose files are corrupt are quarantined (listed in
``<output-root>/<Plant>/metadata/quarantined_frames.csv``) and excluded; every
valid paired frame is kept, in acquisition order.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foresplat_adapter.discovery import (  # noqa: E402
    discover_plants,
    write_manifest,
    write_quarantine_csvs,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Discover plant scenes and write batch_manifest.csv")
    parser.add_argument("--root", required=True, help="Root folder with one sub-directory per plant")
    parser.add_argument("--output-root", required=True, help="Per-plant output root")
    parser.add_argument("--manifest", required=True, help="Path of the batch_manifest.csv to write")
    parser.add_argument("--image-prefix", default="crop_")
    parser.add_argument("--mask-prefix", default="mask_")
    args = parser.parse_args(argv)

    plants = discover_plants(args.root, image_prefix=args.image_prefix, mask_prefix=args.mask_prefix)
    write_manifest(plants, args.manifest, args.output_root)
    write_quarantine_csvs(plants, args.output_root)

    print(f"{'plant':16s} {'paired':>6s} {'valid':>6s} {'quarantined':>11s}  resolution")
    for plant in plants:
        res = f"{plant.image_resolution[0]}x{plant.image_resolution[1]}" if plant.image_resolution else "?"
        print(
            f"{plant.plant_id:16s} {plant.matched_images:6d} {plant.matched_images:6d} "
            f"{len(plant.quarantined):11d}  {res}"
        )
    total = sum(p.matched_images for p in plants)
    print(f"\n{len(plants)} plants, {total} valid frames -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())