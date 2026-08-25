#!/usr/bin/env python3
"""Build soft view weights from existing H-VQG score CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_scores(path: Path, score_column: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("image_name")
            value = row.get(score_column)
            if not name or value is None:
                continue
            scores[Path(name).stem] = float(value)
    return scores


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def aliases(name: str) -> list[str]:
    stem = Path(name).stem
    values = [stem]
    for prefix in ("crop_", "mask_"):
        if stem.startswith(prefix):
            values.append(stem[len(prefix) :])
    return values


def lookup_score(scores: dict[str, float], name: str, default: float) -> float:
    for key in aliases(name):
        if key in scores:
            return scores[key]
    return default


def read_scene_image_names(scene_images_dir: Path | None) -> list[str]:
    if scene_images_dir is None:
        return []
    names = []
    for path in sorted(scene_images_dir.iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            names.append(path.stem)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-scores", required=True, type=Path)
    parser.add_argument("--mask-scores", required=True, type=Path)
    parser.add_argument("--geo-scores", type=Path, default=None)
    parser.add_argument("--scene-images-dir", type=Path, default=None)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--raw-weight", type=float, default=0.35)
    parser.add_argument("--mask-weight", type=float, default=0.45)
    parser.add_argument("--geo-weight", type=float, default=0.20)
    parser.add_argument("--min-weight", type=float, default=0.60)
    parser.add_argument("--max-weight", type=float, default=1.00)
    parser.add_argument("--default-geo-score", type=float, default=0.75)
    args = parser.parse_args()

    raw_scores = read_scores(args.raw_scores, "raw_score")
    mask_scores = read_scores(args.mask_scores, "mask_score")
    geo_scores = read_scores(args.geo_scores, "geo_score") if args.geo_scores else {}

    scene_names = read_scene_image_names(args.scene_images_dir)
    names = sorted(scene_names or (set(raw_scores) | set(mask_scores) | set(geo_scores)))
    if not names:
        raise ValueError("No image names found in score files.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        raw = lookup_score(raw_scores, name, 1.0)
        mask = lookup_score(mask_scores, name, 1.0)
        geo = lookup_score(geo_scores, name, args.default_geo_score)
        quality = args.raw_weight * raw + args.mask_weight * mask + args.geo_weight * geo
        weight = args.min_weight + (args.max_weight - args.min_weight) * clamp(quality, 0.0, 1.0)
        rows.append(
            {
                "image_name": name,
                "weight": f"{weight:.6f}",
                "quality_score": f"{quality:.6f}",
                "raw_score": f"{raw:.6f}",
                "mask_score": f"{mask:.6f}",
                "geo_score": f"{geo:.6f}",
            }
        )

    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_name", "weight", "quality_score", "raw_score", "mask_score", "geo_score"],
        )
        writer.writeheader()
        writer.writerows(rows)

    weights = [float(row["weight"]) for row in rows]
    summary = {
        "num_views": len(rows),
        "min_weight": min(weights),
        "max_weight": max(weights),
        "mean_weight": sum(weights) / len(weights),
        "settings": {
            "raw_weight": args.raw_weight,
            "mask_weight": args.mask_weight,
            "geo_weight": args.geo_weight,
            "min_weight": args.min_weight,
            "max_weight": args.max_weight,
            "default_geo_score": args.default_geo_score,
        },
        "output_csv": str(args.output_csv),
    }
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
