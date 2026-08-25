#!/usr/bin/env python3
"""Document the expected trait-measurement inputs and outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--scale-marker", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = {
        "mesh": str(args.mesh),
        "scale_marker": str(args.scale_marker),
        "landmarks": str(args.landmarks) if args.landmarks else None,
        "expected_traits": ["plant_height", "canopy_width", "leaf_length", "leaf_width"],
        "expected_output": "CSV with trait, replicate, virtual measurement and optional manual reference columns.",
        "note": "The full measurement tool is interactive/semi-automatic and is not bundled with this lightweight subset.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote measurement manifest to {args.output}")


if __name__ == "__main__":
    main()
