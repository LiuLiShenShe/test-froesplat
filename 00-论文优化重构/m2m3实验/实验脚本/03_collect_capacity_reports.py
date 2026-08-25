#!/usr/bin/env python3
"""Collect M2M3 capacity summaries and final point-cloud sizes into CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def latest_ply(model_dir: Path) -> tuple[str, int]:
    candidates = []
    for ply in model_dir.glob("point_cloud/iteration_*/point_cloud.ply"):
        try:
            iteration = int(ply.parent.name.split("_")[-1])
        except ValueError:
            iteration = -1
        candidates.append((iteration, ply))
    if not candidates:
        return "", 0
    iteration, ply = sorted(candidates)[-1]
    return str(iteration), ply.stat().st_size


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect(output_root: Path, run_tag: str | None) -> list[dict[str, object]]:
    runs_root = output_root / "runs"
    pattern = f"{run_tag}/*/*" if run_tag else "*/*/*"
    rows = []
    for model_dir in sorted(runs_root.glob(pattern)):
        if not model_dir.is_dir():
            continue
        rel = model_dir.relative_to(runs_root)
        if len(rel.parts) != 3:
            continue
        tag, scene, method = rel.parts
        capacity_summary = read_json(model_dir / "capacity_control" / "capacity_summary.json")
        foreground_init = read_json(model_dir / "foreground_init_pcd_report.json")
        final_iter, ply_bytes = latest_ply(model_dir)
        rows.append(
            {
                "run_tag": tag,
                "scene": scene,
                "method": method,
                "model_dir": str(model_dir),
                "final_iteration": final_iter,
                "final_ply_bytes": ply_bytes,
                "capacity_mode": capacity_summary.get("capacity_control_mode", "none"),
                "initial_count": capacity_summary.get("initial_count", ""),
                "max_seen_count": capacity_summary.get("max_seen_count", ""),
                "final_count": capacity_summary.get("final_count", ""),
                "rounds": capacity_summary.get("rounds", ""),
                "floor_ratio": capacity_summary.get("capacity_floor_ratio", ""),
                "floor_reference": capacity_summary.get("capacity_floor_reference", ""),
                "foreground_init_points_before": foreground_init.get("points_before", ""),
                "foreground_init_points_after": foreground_init.get("points_after", ""),
                "foreground_init_kept_ratio": foreground_init.get("kept_ratio", ""),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出", type=Path)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--output-csv", default=None, type=Path)
    args = parser.parse_args()

    rows = collect(args.output_root, args.run_tag or None)
    report_dir = args.output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv if args.output_csv is not None else report_dir / "capacity_and_pointcloud_summary.csv"
    fieldnames = [
        "run_tag",
        "scene",
        "method",
        "model_dir",
        "final_iteration",
        "final_ply_bytes",
        "capacity_mode",
        "initial_count",
        "max_seen_count",
        "final_count",
        "rounds",
        "floor_ratio",
        "floor_reference",
        "foreground_init_points_before",
        "foreground_init_points_after",
        "foreground_init_kept_ratio",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
