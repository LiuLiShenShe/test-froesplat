#!/usr/bin/env python3
"""Compute mesh-only structural metrics for S19 M5 outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


def read_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Empty or unreadable mesh: {path}")
    return mesh


def compute_topology_metrics(mesh: o3d.geometry.TriangleMesh, small_component_triangle_threshold: int = 100) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_count = int(vertices.shape[0])
    triangle_count = int(triangles.shape[0])

    used_vertices = set(int(v) for v in triangles.reshape(-1)) if triangle_count else set()
    isolated_vertices = vertex_count - len(used_vertices)

    if triangle_count == 0:
        return {
            "vertices": vertex_count,
            "triangles": triangle_count,
            "connected_components": 0,
            "largest_component_triangles": 0,
            "largest_component_vertices": 0,
            "largest_component_ratio": 0.0,
            "small_component_count": 0,
            "isolated_vertices": int(isolated_vertices),
            "boundary_edge_count": 0,
            "non_manifold_edge_count": 0,
        }

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters_np = np.asarray(triangle_clusters)
    cluster_n_triangles_np = np.asarray(cluster_n_triangles)
    connected_components = int(len(cluster_n_triangles_np))

    component_vertices: list[int] = []
    for component_idx in range(connected_components):
        component_triangles = triangles[triangle_clusters_np == component_idx]
        if component_triangles.size == 0:
            component_vertices.append(0)
        else:
            component_vertices.append(int(np.unique(component_triangles.reshape(-1)).shape[0]))

    largest_component_triangles = int(cluster_n_triangles_np.max()) if connected_components else 0
    largest_component_vertices = int(max(component_vertices)) if component_vertices else 0
    largest_component_ratio = float(largest_component_vertices / vertex_count) if vertex_count else 0.0
    small_component_count = int(np.sum(cluster_n_triangles_np < small_component_triangle_threshold))

    edge_counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        tri_edges = (
            (int(tri[0]), int(tri[1])),
            (int(tri[1]), int(tri[2])),
            (int(tri[2]), int(tri[0])),
        )
        for a, b in tri_edges:
            key = (a, b) if a < b else (b, a)
            edge_counts[key] = edge_counts.get(key, 0) + 1
    boundary_edge_count = sum(1 for count in edge_counts.values() if count == 1)
    non_manifold_edge_count = sum(1 for count in edge_counts.values() if count > 2)

    return {
        "vertices": vertex_count,
        "triangles": triangle_count,
        "connected_components": connected_components,
        "largest_component_triangles": largest_component_triangles,
        "largest_component_vertices": largest_component_vertices,
        "largest_component_ratio": largest_component_ratio,
        "small_component_count": small_component_count,
        "isolated_vertices": int(isolated_vertices),
        "boundary_edge_count": int(boundary_edge_count),
        "non_manifold_edge_count": int(non_manifold_edge_count),
    }


def compute_displacement_metrics(source_path: Path, target_path: Path) -> dict[str, Any]:
    source_mesh = read_mesh(source_path)
    target_mesh = read_mesh(target_path)
    source_vertices = np.asarray(source_mesh.vertices)
    target_vertices = np.asarray(target_mesh.vertices)
    if source_vertices.shape != target_vertices.shape:
        return {
            "source_vertices": int(source_vertices.shape[0]),
            "target_vertices": int(target_vertices.shape[0]),
            "mean_displacement": None,
            "p95_displacement": None,
            "max_displacement": None,
            "reason": "vertex_count_mismatch",
        }

    displacement = np.linalg.norm(target_vertices - source_vertices, axis=1)
    return {
        "source_vertices": int(source_vertices.shape[0]),
        "target_vertices": int(target_vertices.shape[0]),
        "mean_displacement": float(np.mean(displacement)) if displacement.size else 0.0,
        "p95_displacement": float(np.percentile(displacement, 95)) if displacement.size else 0.0,
        "max_displacement": float(np.max(displacement)) if displacement.size else 0.0,
    }


def find_post_mesh(variant_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in variant_dir.glob("*.ply")
        if path.name.endswith("_post.ply") or path.name == "fuse_post.ply"
    )
    if not candidates:
        raise FileNotFoundError(f"No post-processed mesh found in {variant_dir}")
    return candidates[0]


def find_raw_mesh(variant_dir: Path) -> Path:
    candidates = sorted(path for path in variant_dir.glob("*.ply") if not path.name.endswith("_post.ply"))
    if not candidates:
        raise FileNotFoundError(f"No raw mesh found in {variant_dir}")
    return candidates[0]


def parse_variant_dir_name(variant_dir: Path) -> tuple[str, str, str]:
    name = variant_dir.name
    marker = "_A6_M1_soft_M4_"
    if marker not in name:
        raise ValueError(f"Unexpected S19 variant directory name: {name}")
    sample, mesh_variant = name.split(marker, maxsplit=1)
    model_variant = "A6+M1-soft+M4"
    mesh_variant = {
        "standard": "standard_tsdf",
        "small_trunc": "small_trunc_tsdf",
        "post_boundary": "post_boundary",
    }.get(mesh_variant, mesh_variant)
    return sample, model_variant, mesh_variant


def find_standard_post_mesh(s19_root: Path, sample: str) -> Path | None:
    standard_dir = s19_root / f"{sample}_A6_M1_soft_M4_standard"
    if not standard_dir.exists():
        return None
    try:
        return find_post_mesh(standard_dir)
    except FileNotFoundError:
        return None


def summarize_variant(variant_dir: Path, small_component_triangle_threshold: int) -> dict[str, Any]:
    sample, model_variant, mesh_variant = parse_variant_dir_name(variant_dir)
    metrics_path = variant_dir / "mesh_metrics.json"
    mesh_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    raw_mesh_path = find_raw_mesh(variant_dir)
    post_mesh_path = find_post_mesh(variant_dir)
    post_mesh = read_mesh(post_mesh_path)
    topology = compute_topology_metrics(post_mesh, small_component_triangle_threshold)
    displacement = compute_displacement_metrics(raw_mesh_path, post_mesh_path)
    edge_metrics = mesh_metrics.get("edge_metrics") or {}
    standard_to_post_boundary = None
    if mesh_variant == "post_boundary":
        standard_post_mesh = find_standard_post_mesh(variant_dir.parent, sample)
        if standard_post_mesh is not None:
            standard_to_post_boundary = compute_displacement_metrics(standard_post_mesh, post_mesh_path)
    return {
        "sample": sample,
        "model_variant": model_variant,
        "mesh_variant": mesh_variant,
        "raw_mesh": raw_mesh_path.name,
        "post_mesh": post_mesh_path.name,
        "post_vertices": topology["vertices"],
        "post_triangles": topology["triangles"],
        "connected_components": topology["connected_components"],
        "largest_component_ratio": topology["largest_component_ratio"],
        "small_component_count": topology["small_component_count"],
        "isolated_vertices": topology["isolated_vertices"],
        "boundary_edge_count": topology["boundary_edge_count"],
        "non_manifold_edge_count": topology["non_manifold_edge_count"],
        "boundary_consistency": edge_metrics.get("mean_consistency"),
        "mean_displacement": edge_metrics.get("mean_displacement"),
        "p95_raw_to_post_displacement": displacement.get("p95_displacement"),
        "max_raw_to_post_displacement": displacement.get("max_displacement"),
        "p95_standard_to_post_boundary_displacement": (
            standard_to_post_boundary.get("p95_displacement") if standard_to_post_boundary else None
        ),
        "max_standard_to_post_boundary_displacement": (
            standard_to_post_boundary.get("max_displacement") if standard_to_post_boundary else None
        ),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "model_variant",
        "mesh_variant",
        "post_vertices",
        "post_triangles",
        "connected_components",
        "largest_component_ratio",
        "small_component_count",
        "isolated_vertices",
        "boundary_edge_count",
        "non_manifold_edge_count",
        "boundary_consistency",
        "mean_displacement",
        "p95_raw_to_post_displacement",
        "max_raw_to_post_displacement",
        "p95_standard_to_post_boundary_displacement",
        "max_standard_to_post_boundary_displacement",
        "raw_mesh",
        "post_mesh",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s19-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--small-component-triangle-threshold", default=100, type=int)
    args = parser.parse_args()

    variant_dirs = sorted(path for path in args.s19_root.iterdir() if path.is_dir() and (path / "mesh_metrics.json").exists())
    rows = [summarize_variant(path, args.small_component_triangle_threshold) for path in variant_dirs]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(rows, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
