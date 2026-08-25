#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize experiment 4b downstream 2DGS mask-source closed-loop results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData


ROOT = Path("/data/fj/F2DMAS")
PAPER_ROOT = ROOT / "00-论文优化重构"
DATA_ROOT = PAPER_ROOT / "数据管理"
WORKSPACE_DIR = PAPER_ROOT / "计算机与电子农业特刊实验工作区"
RESULT_TABLE_DIR = WORKSPACE_DIR / "04-结果表格模板"
DEFAULT_RESULT_ROOT = DATA_ROOT / "05-评测结果/S24_E4b_mask_source_2DGS_closed_loop"


SUMMARY_HEADER = [
    "Mask source",
    "n samples",
    "completed samples",
    "PSNR_fg mean",
    "SSIM_fg mean",
    "LPIPS_fg mean",
    "outside ratio mean",
    "leakage energy mean",
    "Gaussian count mean",
    "mesh connected components mean",
    "largest component ratio mean",
    "status",
    "notes",
]

PER_SAMPLE_HEADER = [
    "Sample",
    "Mask source",
    "mask_source_id",
    "model_path",
    "source_path",
    "mask_dir",
    "prepared_mask_dir",
    "status",
    "missing_mask_count",
    "eval_images",
    "PSNR_fg",
    "SSIM_fg",
    "LPIPS_fg",
    "outside ratio",
    "leakage energy",
    "Gaussian count",
    "mesh connected components",
    "largest component ratio",
    "post mesh vertices",
    "post mesh triangles",
    "full PSNR",
    "full SSIM",
    "full LPIPS",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULT_TABLE_DIR)
    parser.add_argument("--iterations", type=int, default=30000)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def fmt(value: Any, digits: int = 4) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def read_foreground_metrics(model_path: Path, iterations: int) -> dict[str, Any]:
    data = read_json(model_path / "foreground_object_results.json")
    if not isinstance(data, dict):
        return {}
    method = f"ours_{iterations}"
    metrics = data.get(method)
    if isinstance(metrics, dict):
        return metrics
    for key, value in data.items():
        if key.startswith("ours_") and isinstance(value, dict):
            return value
    return {}


def read_full_metrics(model_path: Path, iterations: int) -> dict[str, Any]:
    data = read_json(model_path / "results.json")
    if not isinstance(data, dict):
        return {}
    method = f"ours_{iterations}"
    metrics = data.get(method)
    if isinstance(metrics, dict):
        return metrics
    for key, value in data.items():
        if key.startswith("ours_") and isinstance(value, dict):
            return value
    return {}


def gaussian_count(model_path: Path, iterations: int) -> int | None:
    ply_path = model_path / f"point_cloud/iteration_{iterations}/point_cloud.ply"
    if not ply_path.exists():
        return None
    try:
        ply = PlyData.read(ply_path)
        return int(ply["vertex"].count)
    except Exception:
        return None


def find_post_mesh(model_path: Path, iterations: int) -> Path | None:
    train_dir = model_path / f"train/ours_{iterations}"
    if not train_dir.exists():
        return None
    candidates = [
        train_dir / "fuse_post.ply",
        train_dir / "fuse_standard_post.ply",
        train_dir / "fuse_small_trunc_post.ply",
        train_dir / "fuse_post_boundary_post.ply",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    extra = sorted(path for path in train_dir.glob("*_post.ply") if path.is_file())
    return extra[0] if extra else None


def mesh_topology(mesh_path: Path | None) -> dict[str, Any]:
    if mesh_path is None or not mesh_path.exists():
        return {}
    metrics_path = mesh_path.parents[2] / "mesh_metrics.json"
    stored_metrics = read_json(metrics_path)
    stored_topology: dict[str, Any] = {}
    if isinstance(stored_metrics, dict):
        post_mesh = stored_metrics.get("post_mesh")
        if isinstance(post_mesh, dict):
            stored_topology = {
                "post_vertices": post_mesh.get("vertices", ""),
                "post_triangles": post_mesh.get("triangles", ""),
            }
    try:
        import open3d as o3d
    except Exception:
        if stored_topology:
            stored_topology["mesh_warning"] = "open3d_unavailable_for_topology"
            return stored_topology
        return {"mesh_error": "open3d_unavailable"}

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        return {"mesh_error": "empty_mesh"}
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_count = int(vertices.shape[0])
    triangle_count = int(triangles.shape[0])
    if triangle_count == 0:
        return {
            "post_vertices": vertex_count,
            "post_triangles": triangle_count,
            "connected_components": 0,
            "largest_component_ratio": 0.0,
        }
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters_np = np.asarray(triangle_clusters)
    connected_components = int(len(cluster_n_triangles))
    component_vertices: list[int] = []
    for component_idx in range(connected_components):
        component_triangles = triangles[triangle_clusters_np == component_idx]
        if component_triangles.size:
            component_vertices.append(int(np.unique(component_triangles.reshape(-1)).shape[0]))
    largest_component_vertices = max(component_vertices) if component_vertices else 0
    return {
        "post_vertices": vertex_count,
        "post_triangles": triangle_count,
        "connected_components": connected_components,
        "largest_component_ratio": float(largest_component_vertices / vertex_count) if vertex_count else 0.0,
    }


def row_status(manifest_row: dict[str, str], fg: dict[str, Any], gaussians: int | None, mesh: dict[str, Any]) -> tuple[str, str]:
    missing_mask_count = int(manifest_row.get("missing_mask_count") or 0)
    if missing_mask_count > 0:
        return "missing_masks", f"{missing_mask_count} masks missing"
    if not Path(manifest_row.get("model_path", "")).exists():
        return "not_started", "model output directory missing"
    if not fg:
        return "missing_foreground_metrics", "run render + foreground-metrics"
    if gaussians is None:
        return "missing_gaussians", "point cloud ply missing"
    if not mesh:
        return "missing_mesh", "run mesh step"
    if mesh.get("mesh_error"):
        return "mesh_error", str(mesh["mesh_error"])
    if mesh.get("mesh_warning"):
        return "complete_partial_mesh_topology", str(mesh["mesh_warning"])
    return "complete", ""


def per_sample_row(manifest_row: dict[str, str], iterations: int) -> dict[str, Any]:
    model_path = Path(manifest_row.get("model_path", ""))
    fg = read_foreground_metrics(model_path, iterations)
    full = read_full_metrics(model_path, iterations)
    gaussians = gaussian_count(model_path, iterations)
    mesh = mesh_topology(find_post_mesh(model_path, iterations))
    status, notes = row_status(manifest_row, fg, gaussians, mesh)

    return {
        "Sample": manifest_row.get("sample", ""),
        "Mask source": manifest_row.get("mask_source", ""),
        "mask_source_id": manifest_row.get("mask_source_id", ""),
        "model_path": str(model_path),
        "source_path": manifest_row.get("source_path", ""),
        "mask_dir": manifest_row.get("raw_mask_dir", ""),
        "prepared_mask_dir": manifest_row.get("prepared_mask_dir", ""),
        "status": status,
        "missing_mask_count": manifest_row.get("missing_mask_count", ""),
        "eval_images": fg.get("num_images", ""),
        "PSNR_fg": fmt(fg.get("PSNR_fg"), 4),
        "SSIM_fg": fmt(fg.get("SSIM_fg"), 4),
        "LPIPS_fg": fmt(fg.get("LPIPS_fg_black_bg"), 4),
        "outside ratio": fmt(fg.get("outside_nonblack_ratio_mean"), 6),
        "leakage energy": fmt(fg.get("leakage_energy_ratio_mean"), 6),
        "Gaussian count": gaussians if gaussians is not None else "",
        "mesh connected components": mesh.get("connected_components", ""),
        "largest component ratio": fmt(mesh.get("largest_component_ratio"), 6),
        "post mesh vertices": mesh.get("post_vertices", ""),
        "post mesh triangles": mesh.get("post_triangles", ""),
        "full PSNR": fmt(full.get("PSNR"), 4),
        "full SSIM": fmt(full.get("SSIM"), 4),
        "full LPIPS": fmt(full.get("LPIPS"), 4),
        "notes": notes,
    }


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = as_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def mean_or_blank(rows: list[dict[str, Any]], key: str, digits: int = 4) -> str:
    values = numeric_values(rows, key)
    if not values:
        return ""
    return f"{sum(values) / len(values):.{digits}f}"


def summarize(per_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in per_rows:
        by_source.setdefault(str(row["Mask source"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for source, rows in by_source.items():
        complete = [row for row in rows if row.get("status") == "complete"]
        blockers = sorted({str(row.get("status")) for row in rows if row.get("status") != "complete"})
        status = "complete" if len(complete) == len(rows) else "partial"
        summary_rows.append(
            {
                "Mask source": source,
                "n samples": len(rows),
                "completed samples": len(complete),
                "PSNR_fg mean": mean_or_blank(complete, "PSNR_fg", 4),
                "SSIM_fg mean": mean_or_blank(complete, "SSIM_fg", 4),
                "LPIPS_fg mean": mean_or_blank(complete, "LPIPS_fg", 4),
                "outside ratio mean": mean_or_blank(complete, "outside ratio", 6),
                "leakage energy mean": mean_or_blank(complete, "leakage energy", 6),
                "Gaussian count mean": mean_or_blank(complete, "Gaussian count", 0),
                "mesh connected components mean": mean_or_blank(complete, "mesh connected components", 2),
                "largest component ratio mean": mean_or_blank(complete, "largest component ratio", 6),
                "status": status,
                "notes": "; ".join(blockers),
            }
        )
    return summary_rows


def main() -> int:
    args = parse_args()
    manifest = args.manifest or args.result_root / "run_manifest.csv"
    manifest_rows = read_csv_rows(manifest)
    if not manifest_rows:
        raise FileNotFoundError(f"Manifest not found or empty: {manifest}")

    per_rows = [per_sample_row(row, args.iterations) for row in manifest_rows]
    summary_rows = summarize(per_rows)

    per_sample_path = args.output_dir / "实验四_下游三方法2DGS小闭环逐样本表.csv"
    summary_path = args.output_dir / "实验四_下游三方法2DGS小闭环主表.csv"
    write_csv(per_sample_path, PER_SAMPLE_HEADER, per_rows)
    write_csv(summary_path, SUMMARY_HEADER, summary_rows)
    args.result_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.result_root / "experiment4b_per_sample.csv", PER_SAMPLE_HEADER, per_rows)
    write_csv(args.result_root / "experiment4b_summary.csv", SUMMARY_HEADER, summary_rows)
    (args.result_root / "experiment4b_per_sample.json").write_text(
        json.dumps(per_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.result_root / "experiment4b_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {per_sample_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
