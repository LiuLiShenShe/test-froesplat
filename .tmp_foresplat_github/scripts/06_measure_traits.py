#!/usr/bin/env python3
"""Measure simple plant traits from a foreground mesh bounding box."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_ply_vertices(path: Path) -> list[tuple[float, float, float]]:
    vertices: list[tuple[float, float, float]] = []
    vertex_count = 0
    in_header = True
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if in_header:
                if stripped.startswith("element vertex"):
                    vertex_count = int(stripped.split()[-1])
                elif stripped == "end_header":
                    in_header = False
                continue
            if len(vertices) < vertex_count:
                x, y, z = map(float, stripped.split()[:3])
                vertices.append((x, y, z))
            else:
                break
    return vertices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--scale", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vertices = read_ply_vertices(args.mesh)
    if not vertices:
        raise SystemExit(f"No vertices found in {args.mesh}")
    scale = json.loads(args.scale.read_text(encoding="utf-8"))
    units_per_meter = float(scale.get("units_per_meter", 1.0))
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    plant_height_cm = (max(ys) - min(ys)) / units_per_meter * 100.0
    canopy_width_cm = max(max(xs) - min(xs), max(zs) - min(zs)) / units_per_meter * 100.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"trait": "plant_height", "value_cm": f"{plant_height_cm:.3f}", "method": "mesh_bounding_range"},
        {"trait": "canopy_width", "value_cm": f"{canopy_width_cm:.3f}", "method": "mesh_bounding_range"},
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote trait measurements to {args.output}")


if __name__ == "__main__":
    main()
