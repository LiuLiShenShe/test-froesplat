#!/usr/bin/env python3
"""Create or document a foreground-object mesh extraction step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def write_demo_mesh(path: Path, height_m: float = 0.28, width_m: float = 0.18) -> None:
    hw = width_m / 2.0
    vertices = [
        (-hw, 0.0, -hw),
        (hw, 0.0, -hw),
        (hw, 0.0, hw),
        (-hw, 0.0, hw),
        (0.0, height_m, 0.0),
    ]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for vertex in vertices:
            f.write("{:.6f} {:.6f} {:.6f}\n".format(*vertex))
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="ForeSplat run manifest or checkpoint path.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    mesh_path = args.output / "plant_mesh.ply"
    write_demo_mesh(mesh_path)
    manifest = {
        "run": str(args.run),
        "mesh": str(mesh_path),
        "mesh_extraction": config.get("mesh_extraction", {}),
        "note": "Demo mesh documents the expected PLY output path; full runs should replace it with TSDF-style fusion output.",
    }
    (args.output / "mesh_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote demo mesh to {mesh_path}")


if __name__ == "__main__":
    main()
