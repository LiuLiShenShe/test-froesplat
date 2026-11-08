#!/usr/bin/env python3
"""Finalise a plant whose dataset is already exported (verification + report).

For use after an interrupted run (e.g. an OOM killed the verification stage but
the GGGS_Dataset is complete on disk): re-runs only the lightweight steps

* static dataset checks (header-only, no full image loads)
* GGGS readColmapSceneInfo + GaussianModel.create_from_pcd (ply only)
* per-plant visualisations
* RUN_REPORT.md / STATUS.json / batch_summary.csv row

without redoing inference, fusion or the COLMAP export.

Usage::

    CUDA_VISIBLE_DEVICES=1 python tools/finalize_plant.py \
        --plant ChangShouHua2 \
        --manifest runs/full_sequence_batch/batch_manifest.csv \
        --output-root runs/full_sequence_batch \
        --gggs-repo gggs_community
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.run_all_plant_datasets import (  # noqa: E402
    SUMMARY_FIELDS,
    _write_csv,
    _write_json_status,
    _write_run_report,
    read_manifest,
)


def _detect_hardlinks(dataset_dir: str, source_images_dir: str) -> bool:
    """True if dataset images are hard links to the source folder (same inode)."""
    images_dir = os.path.join(dataset_dir, "images")
    if not (os.path.isdir(images_dir) and os.path.isdir(source_images_dir)):
        return False
    name = next((n for n in sorted(os.listdir(images_dir)) if n.lower().endswith(".png")), None)
    if not name:
        return False
    dataset_file = os.path.join(images_dir, name)
    source_file = os.path.join(source_images_dir, name)
    if not os.path.isfile(source_file):
        return False
    try:
        return os.path.samefile(dataset_file, source_file)
    except OSError:
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify + report an already exported plant")
    parser.add_argument("--plant", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gggs-repo", default="gggs_community" if os.path.isdir("gggs_community") else None)
    parser.add_argument("--skip-visualisation", action="store_true")
    args = parser.parse_args(argv)

    rows = read_manifest(args.manifest)
    plant_row = next(r for r in rows if r["plant_id"] == args.plant)
    output_dir = os.path.join(args.output_root, args.plant)
    dataset_dir = os.path.join(output_dir, "GGGS_Dataset")
    for path in (
        os.path.join(dataset_dir, "sparse", "0", "points3D.ply"),
        os.path.join(dataset_dir, "metadata", "fusion_metadata.json"),
        os.path.join(dataset_dir, "metadata", "reconstruction_metadata.json"),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    t0 = time.time()

    fusion_meta = json.load(open(os.path.join(dataset_dir, "metadata", "fusion_metadata.json"), encoding="utf-8"))
    recon_meta = json.load(open(os.path.join(dataset_dir, "metadata", "reconstruction_metadata.json"), encoding="utf-8"))
    fusion_stats = fusion_meta["fusion_stats"]
    attempts = fusion_meta["ladder_attempts"]
    attempt = next(a for a in attempts if not a.get("out_of_memory"))

    # ---- static checks (header only) --------------------------------------
    status = {"colmap_read": "skipped", "gaussian_init": "skipped", "error": None}
    try:
        from foresplat_adapter.verify import verify_dataset_static

        static = verify_dataset_static(dataset_dir)
        status["colmap_read"] = "ok" if static.get("all_checks_passed") else static.get("checks")
    except Exception as exc:  # noqa: BLE001
        status["colmap_read"] = f"failed: {exc}"

    # ---- GGGS read + Gaussian init (ply only, no image loading) ------------
    if args.gggs_repo:
        try:
            from foresplat_adapter.gaussian_init import check_dataset_and_init_gaussians

            init = check_dataset_and_init_gaussians(dataset_dir, args.gggs_repo)
            status.update({k: init[k] for k in ("colmap_read", "ply_points", "camera_count",
                                                "gaussian_init", "gaussian_shapes", "error")})
        finally:
            torch.cuda.empty_cache()

    # ---- visualisations -----------------------------------------------------
    viz_files: list[str] = []
    if not args.skip_visualisation:
        from foresplat_adapter.visualisations import write_all_visualisations

        camera_meta = json.load(open(os.path.join(dataset_dir, "per_frame", "cameras.json"), encoding="utf-8"))
        viz_files = write_all_visualisations(
            os.path.join(output_dir, "visualizations"),
            dataset_dir,
            os.path.join(dataset_dir, "per_frame"),
            camera_meta,
            os.path.join(dataset_dir, "debug", "raw_pointcloud.ply"),
            os.path.join(dataset_dir, "sparse", "0", "points3D.ply"),
        )

    # ---- summary reconstructed from on-disk metadata ------------------------
    mean_conf = None
    obs_path = os.path.join(dataset_dir, "debug", "observations.npz")
    if os.path.isfile(obs_path):
        with np.load(obs_path) as data:
            mean_conf = float(data["source_confidence"].mean())

    summary = {
        "num_frames": recon_meta["num_frames"],
        "processed_frames": recon_meta["num_frames"],
        "num_points3d": recon_meta["num_points3d"],
        "points3d_ply_is_fused": recon_meta.get("points3d_ply_is_fused", True),
        "num_fused_points_in_ply": recon_meta.get("num_fused_points_in_ply",
                                                  fusion_stats["num_fused_points"]),
        "fused_points": recon_meta.get("num_fused_points_in_ply", fusion_stats["num_fused_points"]),
        "preprocess_mode": attempt["preprocess_mode"],
        "image_resolution": attempt["image_resolution"],
        "model_input_hw": attempt.get("model_input_hw"),
        "ladder_attempts": attempts,
        "inference_time_s": attempt.get("inference_time_s"),
        "inference_time": attempt.get("inference_time_s"),
        "peak_vram_allocated_mib": attempt.get("peak_vram_allocated_mib"),
        "peak_vram_reserved_mib": attempt.get("peak_vram_reserved_mib"),
        "raw_max_points": 0,
        "link_sources": _detect_hardlinks(dataset_dir, plant_row["images_dir"]),
        "mean_observation_confidence": mean_conf,
        "fusion": fusion_stats,
        "elapsed_seconds": round(time.time() - t0, 3),
    }

    report = _write_run_report(plant_row, summary, status, output_dir)
    _write_json_status(args.output_root, args.plant, {"status": "done", "elapsed": summary["elapsed_seconds"]})

    # ---- batch_summary.csv row ----------------------------------------------
    summary_path = os.path.join(args.output_root, "batch_summary.csv")
    summary_rows: list[dict] = []
    if os.path.isfile(summary_path):
        with open(summary_path, newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))
    by_plant = {row["plant_id"]: row for row in summary_rows}
    raw_points = int(summary["num_points3d"])
    fused_points = int(summary["num_fused_points_in_ply"])
    by_plant[args.plant] = {
        "plant_id": args.plant,
        "total_frames": plant_row["total_images"],
        "processed_frames": summary["processed_frames"],
        "image_resolution": str(summary["image_resolution"]),
        "preprocess_mode": summary["preprocess_mode"],
        "inference_time": summary["inference_time_s"],
        "peak_vram_allocated_mib": summary["peak_vram_allocated_mib"],
        "peak_vram_reserved_mib": summary["peak_vram_reserved_mib"],
        "raw_points": raw_points,
        "fused_points": fused_points,
        "fusion_ratio": (f"{fused_points / raw_points:.4f}" if raw_points else ""),
        "camera_count": status.get("camera_count", ""),
        "mean_confidence": (f"{mean_conf:.4f}" if mean_conf is not None else ""),
        "ggss_read_status": status.get("colmap_read", ""),
        "gaussian_init_status": status.get("gaussian_init", ""),
        "quarantined_frames": plant_row.get("quarantined_frames", ""),
        "status": "done",
    }
    _write_csv(summary_path, list(by_plant.values()), SUMMARY_FIELDS)

    print(f"[finalize] {args.plant}: report={report}")
    print(f"[finalize] GGGS read: {status.get('colmap_read')} | gauss init: {status.get('gaussian_init')}")
    print(f"[finalize] visualisations: {len(viz_files)} file(s) -> {os.path.join(output_dir, 'visualizations')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())