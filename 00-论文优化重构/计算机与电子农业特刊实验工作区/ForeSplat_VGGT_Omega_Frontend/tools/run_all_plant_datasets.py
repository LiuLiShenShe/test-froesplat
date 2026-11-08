#!/usr/bin/env python3
"""Batch runner: full-sequence VGGT-Omega -> fusion -> GGGS dataset, one plant at a time.

Reads ``batch_manifest.csv`` (build it with ``tools/build_batch_manifest.py``),
processes each plant **sequentially** (one plant in VRAM at a time), and writes

    runs/full_sequence_batch/
        batch_manifest.csv
        batch_summary.csv
        batch_report.md
        <Plant>/
            GGGS_Dataset/      (full views, fused points3D.ply, per_frame/ geometry)
            RUN_REPORT.md
            visualizations/
            STATUS.json        (resume marker)

Usage (GPU 1 only, per the project constraint)::

    CUDA_VISIBLE_DEVICES=1 python tools/run_all_plant_datasets.py \
        --manifest runs/full_sequence_batch/batch_manifest.csv \
        --output-root runs/full_sequence_batch \
        --checkpoint weights/vggt_omega_1b_512.pt \
        --gggs-repo gggs_community \
        --resume

Never overwrites the historical 20-frame results: everything is written under
``--output-root`` (default ``runs/full_sequence_batch``).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foresplat_adapter.fusion import FusionConfig  # noqa: E402
from foresplat_adapter.full_sequence import OOM_LADDER, process_plant  # noqa: E402

SUMMARY_FIELDS = [
    "plant_id", "total_frames", "processed_frames", "image_resolution",
    "preprocess_mode", "inference_time", "peak_vram_allocated_mib",
    "peak_vram_reserved_mib", "raw_points", "fused_points", "fusion_ratio",
    "camera_count", "mean_confidence", "ggss_read_status", "gaussian_init_status",
    "quarantined_frames", "status",
]


def read_manifest(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _status_path(output_root: str, plant_id: str) -> str:
    return os.path.join(output_root, plant_id, "STATUS.json")


def _is_done(output_root: str, plant_id: str) -> bool:
    path = _status_path(output_root, plant_id)
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get("status") == "done"
    except (OSError, json.JSONDecodeError):
        return False


def _write_run_report(plant_row: dict, summary: dict, status: dict, output_dir: str) -> str:
    dataset_dir = os.path.join(output_dir, "GGGS_Dataset")
    fusion = summary.get("fusion", {}) or {}
    attempts = summary.get("ladder_attempts", [])
    lines = [
        f"# RUN REPORT - {plant_row['plant_id']}",
        "",
        f"- Source frames: {plant_row['images_dir']}",
        f"- Total paired frames: {plant_row['total_images']}  |  processed: {summary['processed_frames']}"
        f"  |  quarantined: {plant_row.get('quarantined_frames', 0)}",
        f"- Image resolution: {plant_row['image_resolution']}",
        f"- Preprocess: {summary['preprocess_mode']} @ {summary['image_resolution']}"
        f" (model input {summary.get('model_input_hw')})",
        f"- Inference time: {summary.get('inference_time', summary.get('inference_time_s'))} s",
        f"- Peak VRAM: allocated {summary.get('peak_vram_allocated_mib')} MiB /"
        f" reserved {summary.get('peak_vram_reserved_mib')} MiB",
        "",
        "## Fusion",
        "",
        f"- Candidates (per-view foreground pixels): {fusion.get('num_candidates')}",
        f"- Fused points: {summary.get('fused_points', summary.get('num_fused_points_in_ply'))}"
        f" (target {fusion.get('target_points')},"
        f" core voxel {fusion.get('core_voxel_size')}, edge voxel {fusion.get('edge_voxel_size')})",
        f"- Meaningful observations per fused point: {fusion.get('mean_observations_per_point')}",
        f"- Edge representatives: {fusion.get('edge_points')} (thinned {fusion.get('thinned_edge_points')})",
        f"- Raw COLMAP points (per-view, single observation): {summary.get('num_points3d')}"
        f" (cap {summary.get('raw_max_points')}; 0 = unlimited)",
        "",
        "## GGGS verification",
        "",
        f"- readColmapSceneInfo: {status.get('colmap_read')} (cameras: {status.get('camera_count')},"
        f" ply points: {status.get('ply_points')})",
        f"- GaussianModel.create_from_pcd: {status.get('gaussian_init')}",
    ]
    if status.get("error"):
        lines.append(f"- Error: {status['error']}")
    if attempts:
        lines += ["", "## OOM ladder attempts", ""]
        for attempt in attempts:
            lines.append(
                f"- {attempt['preprocess_mode']}@{attempt['image_resolution']}: "
                f"{'OOM' if attempt['out_of_memory'] else 'ok'}"
                f" ({attempt.get('peak_vram_allocated_mib')} MiB allocated, "
                f"{attempt.get('inference_time_s')} s)"
            )
    lines += [
        "",
        "## Notes",
        "",
        "- All views of the sequence are in `GGGS_Dataset/sparse/0` (no frame sampling).",
        "- `points3D.ply` is the multi-view fused cloud; per-view raw points stay in",
        "  `sparse/0/points3D.bin` (single-observation tracks, exact) and `debug/raw_pointcloud.ply`.",
        "- Provenance: `debug/observations.npz` maps every fused point to its source images,",
        "  pixels, confidences and fusion weights.",
        "- Per-frame depth/confidence/model RGB/model masks are in `GGGS_Dataset/per_frame/`.",
        "- Reprojection error is NOT a geometric accuracy measure (points are depth",
        "  back-projections, not SfM triangulations).",
    ]
    if summary.get("link_sources"):
        lines.append(
            "- `images/` and `masks/` are hard links to the source directory (same inodes); "
            "do not delete or edit the source plant folder without re-exporting."
        )
    report = os.path.join(output_dir, "RUN_REPORT.md")
    with open(report, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return report


def _available_mem_gb() -> float:
    """MemAvailable from /proc/meminfo (GB)."""
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return -1.0


def _wait_for_memory(min_free_gb: float, max_wait_h: float = 24.0, poll_s: float = 60.0) -> None:
    """Block until at least ``min_free_gb`` GB of RAM is available.

    The batch host is shared: another user's job can fill the RAM mid-batch.
    Waiting here (before a plant's preprocessing allocates its working set)
    keeps a headroom instead of triggering a system-wide OOM.
    """
    waited = 0.0
    max_wait_s = max_wait_h * 3600.0
    while True:
        avail = _available_mem_gb()
        if avail >= min_free_gb:
            print(f"[batch] memory check: {avail:.1f} GB available (>= {min_free_gb} GB)", flush=True)
            return
        print(f"[batch] memory low: {avail:.1f} GB < {min_free_gb} GB - waiting "
              f"(waited {waited / 60:.0f} min, giving up after {max_wait_h:.0f} h)", flush=True)
        if waited >= max_wait_s:
            print(f"[batch] memory guard gave up waiting; proceeding with {avail:.1f} GB", flush=True)
            return
        time.sleep(poll_s)
        waited += poll_s


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run full-sequence plant batch")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gggs-repo", default="gggs_community" if os.path.isdir("gggs_community") else None)
    parser.add_argument("--resume", action="store_true", help="Skip plants whose STATUS.json is done")
    parser.add_argument("--plant", action="append", default=None, help="Only process these plant ids")
    parser.add_argument("--order", choices=["largest_first", "manifest"], default="largest_first")
    parser.add_argument("--force-resolution-mode", default=None,
                        help="Pin the inference variant, e.g. 'balanced@512' (skips the OOM ladder)")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--preprocess-mode", default="balanced")
    parser.add_argument("--confidence-percentile", type=float, default=20.0)
    parser.add_argument("--target-points", type=int, default=1_500_000)
    parser.add_argument("--min-points", type=int, default=1_000_000)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--edge-voxel-divisor", type=float, default=2.0)
    parser.add_argument("--edge-weight-boost", type=float, default=2.0)
    parser.add_argument("--normal-radius", type=int, default=2)
    parser.add_argument("--edge-rtol", type=float, default=0.03)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-max-points", type=int, default=1_000_000,
                        help="Cap on the per-view raw COLMAP points3D.bin entries per plant "
                             "(seeded random subset, tracks stay consistent; 0 = unlimited)")
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--skip-visualisation", action="store_true")
    parser.add_argument("--link-images", action="store_true",
                        help="Hard-link source images/masks into the dataset instead of copying "
                             "(saves ~1 GB/plant on slow disks; source must be kept)")
    parser.add_argument("--reload-model-per-plant", action="store_true",
                        help="Reload the 4.5 GB checkpoint for every plant (default: load once and keep "
                             "the weights resident; per-plant tensors are still fully released)")
    parser.add_argument("--min-free-gb", type=float, default=10.0,
                        help="Wait until this much RAM is available before starting each plant "
                             "(shared host safety margin; 0 disables the guard)")
    args = parser.parse_args(argv)

    rows = read_manifest(args.manifest)
    if args.plant:
        wanted = set(args.plant)
        rows = [r for r in rows if r["plant_id"] in wanted]
    if args.order == "largest_first":
        rows = sorted(rows, key=lambda r: int(r["matched_images"]), reverse=True)

    os.makedirs(args.output_root, exist_ok=True)
    summary_path = os.path.join(args.output_root, "batch_summary.csv")
    summary_rows: list[dict] = []
    if os.path.isfile(summary_path):
        with open(summary_path, newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))
    by_plant = {row["plant_id"]: row for row in summary_rows}

    print(f"[batch] {len(rows)} plant(s) | order={args.order} | resume={args.resume}")
    shared_model = None
    for plant_row in rows:
        plant_id = plant_row["plant_id"]
        output_dir = os.path.join(args.output_root, plant_id)
        if args.resume and _is_done(args.output_root, plant_id):
            print(f"[batch] {plant_id}: already done, skipping (resume)")
            continue
        if int(plant_row["matched_images"]) < 2:
            print(f"[batch] {plant_id}: fewer than 2 valid frames, skipping")
            continue

        plant_dir = plant_row["images_dir"]
        if args.min_free_gb > 0:
            _wait_for_memory(args.min_free_gb)
        print(f"\n[batch] === {plant_id} ({plant_row['matched_images']} valid frames) ===")
        t0 = time.time()
        forced = None
        if args.force_resolution_mode:
            mode, res = args.force_resolution_mode.split("@")
            forced = (int(res), mode)

        fusion_cfg = FusionConfig(
            target_points=args.target_points,
            min_points=args.min_points,
            max_points=args.max_points,
            edge_voxel_divisor=args.edge_voxel_divisor,
            edge_weight_boost=args.edge_weight_boost,
            seed=args.seed,
        )
        if shared_model is None and not args.reload_model_per_plant:
            # Load the 4.5 GB checkpoint once per invocation (a re-read from the
            # contended data disk costs tens of minutes).  Per-plant tensors are
            # still fully released between plants.
            from foresplat_adapter.inference import load_model

            print("[batch] loading shared VGGT-Omega model (kept for the whole batch)")
            shared_model = load_model(args.checkpoint, device=args.device)
        try:
            summary = process_plant(
                plant_dir=plant_dir,
                output_root=args.output_root,
                checkpoint=args.checkpoint,
                conf_thres=args.confidence_percentile,
                image_resolution=args.image_resolution,
                preprocess_mode=args.preprocess_mode,
                forced_resolution_mode=forced,
                normal_radius=args.normal_radius,
                edge_rtol=args.edge_rtol,
                fusion_config=fusion_cfg,
                device=args.device,
                seed=args.seed,
                gggs_repo=args.gggs_repo,
                link_sources=args.link_images,
                raw_max_points=args.raw_max_points,
                model=shared_model,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch going, record the failure
            print(f"[batch] {plant_id}: FAILED - {type(exc).__name__}: {exc}")
            summary_rows_row = {
                "plant_id": plant_id,
                "total_frames": plant_row["total_images"],
                "quarantined_frames": plant_row.get("quarantined_frames", ""),
                "status": f"failed: {type(exc).__name__}",
            }
            by_plant[plant_id] = summary_rows_row
            _write_csv(summary_path, list(by_plant.values()), SUMMARY_FIELDS)
            _write_json_status(args.output_root, plant_id, {
                "status": "failed", "error": f"{type(exc).__name__}: {exc}", "elapsed": time.time() - t0,
            })
            if args.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            continue

        dataset_dir = os.path.join(output_dir, "GGGS_Dataset")
        status = {"colmap_read": "skipped", "gaussian_init": "skipped", "error": None}
        if not args.skip_verification:
            try:
                from foresplat_adapter.verify import verify_dataset_static

                static = verify_dataset_static(dataset_dir)
                status["static_checks"] = "ok" if static.get("all_checks_passed") else static.get("checks")
                status["colmap_read"] = "ok" if static.get("all_checks_passed") else static.get("checks")
            except Exception as exc:  # noqa: BLE001
                status["colmap_read"] = f"failed: {exc}"
            if args.gggs_repo:
                from foresplat_adapter.gaussian_init import check_dataset_and_init_gaussians

                init = check_dataset_and_init_gaussians(dataset_dir, args.gggs_repo)
                status.update({k: init[k] for k in ("colmap_read", "ply_points", "camera_count",
                                                    "gaussian_init", "gaussian_shapes", "error")})

        if not args.skip_visualisation:
            try:
                from foresplat_adapter.visualisations import write_all_visualisations

                camera_meta = json.load(open(os.path.join(dataset_dir, "per_frame", "cameras.json"), encoding="utf-8"))
                write_all_visualisations(
                    os.path.join(output_dir, "visualizations"),
                    dataset_dir,
                    os.path.join(dataset_dir, "per_frame"),
                    camera_meta,
                    os.path.join(dataset_dir, "debug", "raw_pointcloud.ply"),
                    os.path.join(dataset_dir, "sparse", "0", "points3D.ply"),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[batch] {plant_id}: visualisations failed - {exc}")

        # Never let a reporting bug kill the whole batch: the dataset on disk is
        # complete, and `tools/finalize_plant.py` can regenerate the report later.
        try:
            _write_run_report(plant_row, summary, status, output_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[batch] {plant_id}: report writing failed - {type(exc).__name__}: {exc}")

        fused = summary.get("fusion", {}) or {}
        raw_points = int(summary.get("num_points3d", 0))
        fused_points = int(summary.get("num_fused_points_in_ply", 0))
        row = {
            "plant_id": plant_id,
            "total_frames": plant_row["total_images"],
            "processed_frames": summary.get("num_frames", ""),
            "image_resolution": f"{summary.get('image_resolution')}",
            "preprocess_mode": summary.get("preprocess_mode", ""),
            "inference_time": summary.get("inference_time_s", ""),
            "peak_vram_allocated_mib": summary.get("peak_vram_allocated_mib", ""),
            "peak_vram_reserved_mib": summary.get("peak_vram_reserved_mib", ""),
            "raw_points": raw_points,
            "fused_points": fused_points,
            "fusion_ratio": (f"{fused_points / raw_points:.4f}" if raw_points else ""),
            "camera_count": status.get("camera_count", ""),
            "mean_confidence": (f"{summary.get('mean_observation_confidence', 0.0):.4f}"),
            "ggss_read_status": status.get("colmap_read", ""),
            "gaussian_init_status": status.get("gaussian_init", ""),
            "quarantined_frames": plant_row.get("quarantined_frames", ""),
            "status": "done",
        }
        by_plant[plant_id] = row
        _write_csv(summary_path, list(by_plant.values()), SUMMARY_FIELDS)
        _write_json_status(args.output_root, plant_id, {"status": "done", "elapsed": time.time() - t0})
        print(f"[batch] {plant_id}: done in {time.time() - t0:.1f} s "
              f"(raw {raw_points}, fused {fused_points})")

    _write_batch_report(args.output_root, list(by_plant.values()))
    print(f"\n[batch] summary: {summary_path}")
    return 0


def _write_json_status(output_root: str, plant_id: str, payload: dict) -> None:
    os.makedirs(os.path.join(output_root, plant_id), exist_ok=True)
    with open(_status_path(output_root, plant_id), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_batch_report(output_root: str, rows: list[dict]) -> None:
    lines = [
        "# Full-sequence batch report",
        "",
        f"Plants processed: {len(rows)}",
        "",
        "| plant | frames | processed | res | mode | infer s | peak VRAM (MiB) | raw pts | fused pts | ratio | cameras | GGGS read | Gauss init |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('plant_id')} | {row.get('total_frames')} | {row.get('processed_frames')} | "
            f"{row.get('image_resolution')} | {row.get('preprocess_mode')} | {row.get('inference_time')} | "
            f"{row.get('peak_vram_allocated_mib')} | {row.get('raw_points')} | {row.get('fused_points')} | "
            f"{row.get('fusion_ratio')} | {row.get('camera_count')} | {row.get('ggss_read_status')} | "
            f"{row.get('gaussian_init_status')} |"
        )
    lines += [
        "",
        "## Method notes",
        "",
        "- One plant = one scene; the **entire** valid sequence runs through a single joint",
        "  VGGT-Omega forward pass (no sampling / striding / windowing).",
        "- On OOM the ladder is `balanced@512 -> max_size@512 -> balanced@448 ->"
        " max_size@448 -> balanced@384 -> max_size@384`; frames are never dropped.",
        "- Fusion is confidence-weighted, normal-aware (voxel hash + spherical normal bins,",
        "  so the two sides of a thin leaf and different leaves are never merged) and",
        "  edge-preserving (finer voxel and larger weight on depth discontinuities).",
        "- `points3D.ply` is the fused cloud; per-view raw points remain in `points3D.bin`",
        "  with their single-observation tracks. Reprojection error is not a geometric",
        "  accuracy metric for depth back-projection points.",
    ]
    with open(os.path.join(output_root, "batch_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())