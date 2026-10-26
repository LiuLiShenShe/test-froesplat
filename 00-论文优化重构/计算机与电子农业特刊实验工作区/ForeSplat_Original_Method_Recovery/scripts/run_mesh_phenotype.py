#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TSDF mesh extraction + exploratory phenotype geometry for the downstream 2x2.

For every trained foreground-object model this script

  1. calls ``render.py --skip_train --skip_test`` with the frozen S19 meshing
     protocol (standard TSDF, voxel 0.02 / depth_trunc 6.0 / sdf_trunc 0.08),
  2. archives ``fuse_post.ply`` + ``mesh_metrics.json`` under ``runs/mesh/<tag>/``
     before the next tag overwrites the model directory,
  3. measures exploratory geometry (COLMAP units) from the largest connected
     component of the archived post mesh,
  4. writes ``metrics/mesh_phenotype.csv`` + ``metrics/mesh_geometry.json``.

Physical phenotypes in cm are NOT produced.  KongQueZhuYu has no verified
real-world distance tying the reconstruction to a metric scale, so every
physical trait is emitted with ``PHYSICAL_SCALE_NOT_AVAILABLE`` -- the same
evidence gate Phase20F applies (``阶段二十F_.../scripts/phenotype_core.py``).
Exploratory extents are in COLMAP units and must not be reported as cm.

    python scripts/run_mesh_phenotype.py                     # all four tags
    python scripts/run_mesh_phenotype.py --tags RAP_FULL
    python scripts/run_mesh_phenotype.py --measure-only      # reuse archived meshes
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/data/fj/F2DMAS")
RECOVERY = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
GS = ROOT / "2d-gaussian-splatting-m2m3-floor40"
GS_PY = GS / "venv/bin/python"
DOWNSTREAM = RECOVERY / "runs/downstream"
MESH_ROOT = RECOVERY / "runs/mesh"
METRICS = RECOVERY / "metrics"
SCENE = "KongQueZhuYu"
ITERATION = 30000

TAGS = ["C_R1_baseline", "C_R1_M2M3GO", "RAP_FULL", "RAP_FULL_M2M3GO"]

MESH_ARGS = [
    "--iteration", str(ITERATION),
    "--skip_train", "--skip_test",
    "--voxel_size", "0.02", "--depth_trunc", "6.0", "--sdf_trunc", "0.08",
    "--num_cluster", "20", "--mesh_res", "256",
    "--meshing_mode", "standard",
]


def model_dir(tag: str) -> Path:
    return DOWNSTREAM / f"{SCENE}_{tag}"


def archive_dir(tag: str) -> Path:
    return MESH_ROOT / tag


def run_mesh(tag: str) -> dict:
    """Run render.py once and archive its outputs."""
    mdl = model_dir(tag)
    out = archive_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(GS_PY), "render.py", "--model_path", str(mdl), *MESH_ARGS]
    print(f"[mesh] {tag}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=GS, capture_output=True, text=True)
    wall = time.time() - t0
    (out / "mesh_stdout.log").write_text(proc.stdout[-20000:], encoding="utf-8")
    (out / "mesh_stderr.log").write_text(proc.stderr[-20000:], encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{tag}: render.py failed rc={proc.returncode}; "
                           f"see {out}/mesh_stderr.log")

    metrics_src = mdl / "mesh_metrics.json"
    post_src = Path(json.loads(metrics_src.read_text())["post_mesh"]["path"])
    raw_src = Path(json.loads(metrics_src.read_text())["raw_mesh"]["path"])
    for src, name in ((metrics_src, "mesh_metrics.json"),
                      (post_src, "mesh_post.ply"),
                      (raw_src, "mesh_raw.ply")):
        dst = out / name
        dst.write_bytes(src.read_bytes())
    record = {
        "tag": tag,
        "model_dir": str(mdl),
        "mesh_cmd": cmd,
        "mesh_wall_sec": round(wall, 1),
        "returncode": proc.returncode,
        "meshing_mode": "standard",
        "voxel_size": 0.02, "depth_trunc": 6.0, "sdf_trunc": 0.08,
        "num_cluster": 20, "mesh_res": 256,
        "archived_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "mesh_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def _camera_up(rotation: np.ndarray) -> np.ndarray:
    """World-space camera "up" for a rotation stored as world->camera rows."""
    return rotation.T @ np.array([0.0, -1.0, 0.0])


def up_axis_from_cameras(cameras: list[dict]) -> np.ndarray:
    ups = np.stack([_camera_up(np.asarray(c["rotation"], dtype=np.float64))
                    for c in cameras])
    mean = ups.mean(axis=0)
    return mean / np.linalg.norm(mean)


def horizontal_basis(vertical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(vertical @ helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    a = np.cross(vertical, helper)
    a /= np.linalg.norm(a)
    b = np.cross(vertical, a)
    return a, b / np.linalg.norm(b)


def measure(tag: str) -> dict:
    """Exploratory geometry of the largest connected component."""
    import open3d as o3d

    art = archive_dir(tag)
    mesh = o3d.io.read_triangle_mesh(str(art / "mesh_post.ply"))
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"{tag}: archived post mesh is empty")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    order = np.argsort(counts)[::-1]
    largest_label = int(order[0])
    tri_mask = labels == largest_label
    sub_verts, sub_tris = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    used = np.zeros(len(sub_verts), dtype=bool)
    used[sub_tris[tri_mask].reshape(-1)] = True
    remap = -np.ones(len(sub_verts), dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    tris = remap[sub_tris[tri_mask]]
    verts = sub_verts[used]
    cameras = json.loads((model_dir(tag) / "cameras.json").read_text())
    up = up_axis_from_cameras(cameras)
    ha, hb = horizontal_basis(up)
    projected = np.stack([verts @ up, verts @ ha, verts @ hb], axis=1)

    bbox = verts.max(axis=0) - verts.min(axis=0)
    center = verts.mean(axis=0)
    centered = verts - center
    cov = centered.T @ centered / len(verts)
    eigvals, eigvecs = np.linalg.eigh(cov)
    directions = eigvecs[:, np.argsort(eigvals)[::-1]]
    pca_extents = np.ptp(centered @ directions, axis=0)

    height = float(np.ptp(projected[:, 0]))
    span_a = float(np.ptp(projected[:, 1]))
    span_b = float(np.ptp(projected[:, 2]))
    # base = lowest 2% along up, used as the height reference point
    base_cut = np.percentile(projected[:, 0], 2.0)
    base_pts = verts[projected[:, 0] <= base_cut]
    base_point = base_pts.mean(axis=0)

    # downsampled projection for the silhouette figure (kept out of the CSV)
    keep = min(50000, len(verts))
    pick = np.linspace(0, len(verts) - 1, keep).astype(np.int64)
    np.savez_compressed(
        METRICS / f"mesh_silhouette_{tag}.npz",
        up=projected[pick, 0], span_a=projected[pick, 1],
        span_b=projected[pick, 2])

    prov = json.loads((model_dir(tag) / "mask_provenance.json").read_text())
    run = json.loads((model_dir(tag) / "run_record.json").read_text())
    return {
        "tag": tag,
        "mask_run_dir": run.get("mask_run_dir"),
        "m2m3_go": run.get("m2m3_go"),
        "mask_manifest_sha256": run.get("mask_manifest_sha256"),
        "mask_camera_count": prov.get("camera_count"),
        "mask_missing_count": prov.get("missing_count"),
        "vertices_post": int(len(mesh.vertices)),
        "triangles_post": int(len(mesh.triangles)),
        "components": int(len(counts)),
        "largest_component_ratio": float(counts[order[0]] / counts.sum()),
        "largest_component_triangles": int(counts[order[0]]),
        "largest_component_vertices": int(len(verts)),
        "small_components_le_1pct": int((counts / counts.sum() < 0.01).sum()),
        "bbox_extent_units": bbox.tolist(),
        "pca_center": center.tolist(),
        "pca_extent_units": pca_extents.tolist(),
        "up_axis_from_cameras": up.tolist(),
        "base_point_units": base_point.tolist(),
        "height_along_up_units": height,
        "canopy_span_a_units": span_a,
        "canopy_span_b_units": span_b,
        "canopy_span_max_units": max(span_a, span_b),
        "physical_scale_cm_per_unit": None,
        "physical_status": "PHYSICAL_SCALE_NOT_AVAILABLE",
        "physical_trait_claim": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=",".join(TAGS))
    ap.add_argument("--measure-only", action="store_true",
                    help="skip render.py, reuse archived meshes")
    ap.add_argument("--force", action="store_true", help="re-run render.py")
    cli = ap.parse_args()
    tags = [t for t in cli.tags.split(",") if t]

    METRICS.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag in tags:
        art = archive_dir(tag)
        have = (art / "mesh_post.ply").exists() and (art / "mesh_metrics.json").exists()
        if not cli.measure_only and (cli.force or not have):
            try:
                run_mesh(tag)
            except Exception as exc:
                print(f"[fail] {tag}: {exc}", flush=True)
                continue
        elif not have:
            print(f"[skip] {tag}: no archived mesh (run without --measure-only)")
            continue
        try:
            row = measure(tag)
        except Exception as exc:
            print(f"[fail] {tag}: measure failed: {exc}", flush=True)
            continue
        rows.append(row)
        print(f"[ok] {tag}: V={row['vertices_post']} comps={row['components']} "
              f"height={row['height_along_up_units']:.3f} u "
              f"span={row['canopy_span_max_units']:.3f} u", flush=True)

    if not rows:
        print("no meshes measured")
        return 1

    fields = ["tag", "mask_run_dir", "m2m3_go", "mask_manifest_sha256",
              "mask_camera_count", "mask_missing_count",
              "vertices_post", "triangles_post", "components",
              "largest_component_ratio", "largest_component_triangles",
              "largest_component_vertices", "small_components_le_1pct",
              "pca_extent_0_units", "pca_extent_1_units", "pca_extent_2_units",
              "height_along_up_units", "canopy_span_a_units",
              "canopy_span_b_units", "canopy_span_max_units",
              "physical_scale_cm_per_unit", "physical_status",
              "physical_trait_claim"]
    csv_rows = []
    for r in rows:
        csv_rows.append({
            "tag": r["tag"],
            "mask_run_dir": r["mask_run_dir"],
            "m2m3_go": r["m2m3_go"],
            "mask_manifest_sha256": r["mask_manifest_sha256"],
            "mask_camera_count": r["mask_camera_count"],
            "mask_missing_count": r["mask_missing_count"],
            "vertices_post": r["vertices_post"],
            "triangles_post": r["triangles_post"],
            "components": r["components"],
            "largest_component_ratio": f"{r['largest_component_ratio']:.4f}",
            "largest_component_triangles": r["largest_component_triangles"],
            "largest_component_vertices": r["largest_component_vertices"],
            "small_components_le_1pct": r["small_components_le_1pct"],
            "pca_extent_0_units": f"{r['pca_extent_units'][0]:.4f}",
            "pca_extent_1_units": f"{r['pca_extent_units'][1]:.4f}",
            "pca_extent_2_units": f"{r['pca_extent_units'][2]:.4f}",
            "height_along_up_units": f"{r['height_along_up_units']:.4f}",
            "canopy_span_a_units": f"{r['canopy_span_a_units']:.4f}",
            "canopy_span_b_units": f"{r['canopy_span_b_units']:.4f}",
            "canopy_span_max_units": f"{r['canopy_span_max_units']:.4f}",
            "physical_scale_cm_per_unit": "",
            "physical_status": r["physical_status"],
            "physical_trait_claim": False,
        })
    out_csv = METRICS / "mesh_phenotype.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    out_json = METRICS / "mesh_geometry.json"
    out_json.write_text(json.dumps({
        "scene": SCENE,
        "iteration": ITERATION,
        "meshing_protocol": {"meshing_mode": "standard", "voxel_size": 0.02,
                             "depth_trunc": 6.0, "sdf_trunc": 0.08,
                             "num_cluster": 20, "mesh_res": 256},
        "units": "COLMAP scene units (not cm)",
        "physical_gate": "PHYSICAL_SCALE_NOT_AVAILABLE: no verified real-world "
                         "distance for this scene; see Phase20F phenotype_core",
        "tags": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {out_csv}")
    print(f"[out] {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
