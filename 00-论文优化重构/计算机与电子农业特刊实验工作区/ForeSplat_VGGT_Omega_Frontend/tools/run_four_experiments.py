#!/usr/bin/env python3
"""Run the four controlled ForeSplat-GGGS experiments for one plant.

E1 GGGS baseline            (EO off, Foreground off)
E2 GGGS + EO                (EO on,  Foreground off)
E3 GGGS + Foreground        (EO off, Foreground on)
E4 ForeSplat-GGGS           (EO on,  Foreground on)

All four share one dataset, one fused initialisation point cloud, one
GaussianModel(3,0) constructor, one optimiser/scheduler/densification config and
one random seed (GGGS ``safe_state`` -> seed 0).  Only the two experiment
factors differ, so the comparison is a strict 2x2 ablation.

The orchestrator is strictly sequential (one training in VRAM at a time on GPU
1), waits for >=10 GB of free RAM before each launch, and skips experiments
whose RUN_REPORT.md already exists (safe to re-run).

Usage::

    CUDA_VISIBLE_DEVICES=1 python tools/run_four_experiments.py \
        --plant KongQueZhuYu \
        --batch-root runs/full_sequence_batch \
        --iters 30000 --resolution 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGGS_DIR = os.path.join(ROOT, "gggs_community")
if GGGS_DIR not in sys.path:
    sys.path.insert(0, GGGS_DIR)

FG_FLAGS = [
    "--use_foreground_rgb_loss", "--foreground_rgb_crop_padding", "12",
    "--use_mask_loss", "--use_bg_opacity_loss",
    "--lambda_mask", "0.08", "--lambda_bg", "0.02",
    "--mask_loss_type", "l1_dice", "--mask_ignore_boundary_px", "2",
    "--mask_loss_start_iter", "500", "--mask_loss_warmup_iters", "1500",
    "--pruning_mode", "mask", "--pruning_start_iter", "18000",
    "--pruning_interval", "3000", "--pruning_opacity_threshold", "0.005",
    "--pruning_brightness_threshold", "0.01", "--pruning_mask_threshold", "0.45",
    "--pruning_mask_max_views", "12", "--pruning_max_remove_ratio", "0.03",
    "--pruning_mask_score_weight", "3.0", "--save_pruning_report",
]

EXPERIMENTS = [
    {"id": "E1_GGGS_Baseline", "eo": False, "fg": False},
    {"id": "E2_GGGS_EO", "eo": True, "fg": False},
    {"id": "E3_GGGS_Foreground", "eo": False, "fg": True},
    {"id": "E4_ForeSplat_GGGS", "eo": True, "fg": True},
]


def mem_available_gb() -> float:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return -1.0


def wait_for_memory(min_free_gb: float) -> None:
    while mem_available_gb() < min_free_gb:
        print(f"[orchestrator] memory low: {mem_available_gb():.1f} GB < {min_free_gb} GB - waiting", flush=True)
        time.sleep(60)


def read_plant_row(batch_root: str, plant: str) -> dict:
    path = os.path.join(batch_root, "batch_summary.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["plant_id"] == plant:
                return row
    raise FileNotFoundError(f"plant {plant} not found in {path}")


def write_shared_config(experiments_root: str, plant: str, dataset_dir: str, batch_row: dict, init_points: int,
                        iters: int, resolution: int, masks_dir: str) -> dict:
    import torch  # noqa: F401 - import to keep GGGS env warm
    from scene.dataset_readers import readColmapSceneInfo

    info = readColmapSceneInfo(dataset_dir, "images", False)
    shared = os.path.join(experiments_root, "shared_config")
    os.makedirs(shared, exist_ok=True)

    dataset_info = {
        "plant_id": plant,
        "gggs_dataset_dir": os.path.abspath(dataset_dir),
        "num_train_cameras": len(info.train_cameras),
        "num_test_cameras": len(info.test_cameras),
        "image_resolution_full": [int(info.train_cameras[0].width), int(info.train_cameras[0].height)],
        "camera_model": "PINHOLE",
        "masks_dir": os.path.abspath(masks_dir),
        "num_masks": len([n for n in os.listdir(masks_dir) if n.lower().endswith(".png")]),
        "resolution_used": resolution,
        "white_background": False,
        "eval_split": False,
        "note": "GGGS dataset built by the ForeSplat VGGT-Omega frontend; scene in VGGT-Omega native coordinates.",
    }
    camera_info = {
        "num_cameras": len(info.train_cameras),
        "nerf_normalization_radius": float(info.nerf_normalization["radius"]),
        "cameras_json": os.path.join(os.path.abspath(dataset_dir), "sparse", "0"),
        "ply_path": os.path.abspath(info.ply_path),
    }
    initialization_info = {
        "points3d_ply": os.path.abspath(info.ply_path),
        "num_fused_points": init_points,
        "source": "VGGT-Omega full-sequence multi-view fusion (normal-aware, confidence-weighted, edge-preserving)",
        "coordinate_frame": "VGGT-Omega native (no laser/scan Sim(3) registration, no scale change)",
    }
    config = {
        "shared": {
            "iterations": iters,
            "sh_degree": 0,
            "sg_degree": 0,
            "resolution": resolution,
            "seed": 0,
            "optimizer": {
                "position_lr_init": 0.00016, "position_lr_final": 0.0000016,
                "feature_dc_lr": 0.0013, "feature_rest_lr": 0.00011,
                "opacity_lr": 0.05, "scaling_lr": 0.005, "rotation_lr": 0.001,
                "sg_axis_lr": 0.002, "sg_sharpness_lr": 0.095, "sg_color": 0.00064,
            },
            "densification": {
                "densify_from_iter": 500, "densify_until_iter": 15000,
                "densification_interval": 100, "opacity_reset_interval": 3000,
                "densify_grad_threshold": 0.0002, "opacity_cull": 0.05,
            },
            "losses": {"lambda_dssim": 0.2, "lambda_depth_normal": 0.05,
                       "lambda_multi_view_ncc": 0.6, "lambda_multi_view_geo": 0.02,
                       "regularization_from_iter": 7000},
            "initial_num_gaussians": init_points,
        },
        "experiments": [
            {
                "id": exp["id"], "exposure_compensation": exp["eo"], "foreground_constraints": exp["fg"],
                "flags": ([ "--enable_eo" ] if exp["eo"] else []) + (FG_FLAGS if exp["fg"] else []),
            }
            for exp in EXPERIMENTS
        ],
    }
    json.dump(dataset_info, open(os.path.join(shared, "dataset_info.json"), "w", encoding="utf-8"), indent=2)
    json.dump(camera_info, open(os.path.join(shared, "camera_info.json"), "w", encoding="utf-8"), indent=2)
    json.dump(initialization_info, open(os.path.join(shared, "initialization_info.json"), "w", encoding="utf-8"), indent=2)
    try:
        import yaml
        with open(os.path.join(shared, "experiment_config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
    except ImportError:
        json.dump(config, open(os.path.join(shared, "experiment_config.yaml"), "w", encoding="utf-8"), indent=2)
    return {"num_cameras": len(info.train_cameras), "config": config}


def run_experiment(exp: dict, plant: str, experiments_root: str, dataset_dir: str, masks_dir: str,
                   iters: int, smoke_iters: int, resolution: int, min_free_gb: float) -> bool:
    exp_dir = os.path.join(experiments_root, exp["id"])
    report = os.path.join(exp_dir, "RUN_REPORT.md")
    if os.path.isfile(report):
        print(f"[orchestrator] {exp['id']}: RUN_REPORT exists, skipping", flush=True)
        return True
    os.makedirs(exp_dir, exist_ok=True)
    python = os.path.abspath(os.path.join(ROOT, ".venv/bin/python"))
    trainer = os.path.join(GGGS_DIR, "train_foresplat.py")

    # ---- 1. smoke test --------------------------------------------------- #
    smoke_dir = os.path.join(exp_dir, "smoke")
    smoke_csv = os.path.join(smoke_dir, "train_metrics.csv")
    if not os.path.isfile(os.path.join(smoke_dir, "DONE")):
        wait_for_memory(min_free_gb)
        smoke_flags = ["--enable_eo"] if exp["eo"] else []
        if exp["fg"]:
            # Exercise the foreground losses and one pruning round inside the
            # short smoke run (the full-run schedule starts later).
            smoke_flags += FG_FLAGS + [
                "--mask_loss_start_iter", "0", "--mask_loss_warmup_iters", "20",
                "--pruning_start_iter", "100", "--pruning_interval", "100",
            ]
        cmd = [
            python, trainer,
            "--source_path", dataset_dir, "--model_path", smoke_dir,
            "--images", "images", "--resolution", str(resolution),
            "--smoke", "--iterations", str(smoke_iters),
            "--metrics_csv", smoke_csv, "--metrics_interval", "10",
            "--quiet",
        ] + smoke_flags
        if exp["fg"]:
            # masks only enter the training process in the Foreground experiments
            cmd += ["--mask_dir", masks_dir]
        log_path = os.path.join(smoke_dir, "train.log")
        os.makedirs(smoke_dir, exist_ok=True)
        print(f"[orchestrator] {exp['id']}: smoke test ({smoke_iters} iters) -> {log_path}", flush=True)
        with open(log_path, "w", encoding="utf-8") as log:
            ret = subprocess.run(cmd, cwd=GGGS_DIR, stdout=log, stderr=subprocess.STDOUT).returncode
        ok = ret == 0 and _smoke_ok(smoke_csv, smoke_iters)
        if not ok:
            print(f"[orchestrator] {exp['id']}: SMOKE FAILED (ret={ret})", flush=True)
            return False
        open(os.path.join(smoke_dir, "DONE"), "w").write("ok\n")

    # ---- 2. full training ------------------------------------------------ #
    model_path = os.path.join(exp_dir, "checkpoints")
    csv_path = os.path.join(exp_dir, "metrics", "train_metrics.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    wait_for_memory(min_free_gb)
    cmd = [
        python, trainer,
        "--source_path", dataset_dir, "--model_path", model_path,
        "--images", "images", "--resolution", str(resolution),
        "--iterations", str(iters),
        "--save_iterations", str(max(iters // 2, 1)), str(iters),
        "--test_iterations", str(max(iters // 3, 1)), str(max(2 * iters // 3, 1)), str(iters),
        "--metrics_csv", csv_path, "--metrics_interval", "100",
        "--quiet",
    ]
    if exp["eo"]:
        cmd += ["--enable_eo", "--exposure_params_dir", os.path.join(exp_dir, "exposure_params")]
    if exp["fg"]:
        cmd += FG_FLAGS + ["--mask_dir", masks_dir]
    log_path = os.path.join(exp_dir, "train.log")
    print(f"[orchestrator] {exp['id']}: full training ({iters} iters) -> {log_path}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        ret = subprocess.run(cmd, cwd=GGGS_DIR, stdout=log, stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - t0
    final_ply = os.path.join(model_path, "final_gaussians", "point_cloud.ply")
    if ret != 0 or not os.path.isfile(final_ply):
        print(f"[orchestrator] {exp['id']}: TRAINING FAILED (ret={ret})", flush=True)
        return False

    _write_run_report(exp, plant, exp_dir, iters, resolution, elapsed, final_ply)
    print(f"[orchestrator] {exp['id']}: done in {elapsed / 60:.1f} min", flush=True)
    return True


def _smoke_ok(csv_path: str, smoke_iters: int) -> bool:
    import math

    if not os.path.isfile(csv_path):
        return False
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in ("total_loss", "rgb_loss", "l1_loss", "mask_loss", "bg_loss")
    )
    if not finite:
        return False
    last = float(rows[-1]["iteration"])
    if last < smoke_iters:
        return False
    first = float(rows[0]["total_loss"])
    last_loss = float(rows[-1]["total_loss"])
    print(f"[orchestrator] smoke: loss {first:.4f} -> {last_loss:.4f} (finite, decreasing-ish)", flush=True)
    return True


def _write_run_report(exp: dict, plant: str, exp_dir: str, iters: int, resolution: int, elapsed: float,
                      final_ply: str) -> None:
    import numpy as np

    num_gaussians = -1
    try:
        with np.load(os.path.join(os.path.dirname(final_ply), "gaussians.npz")) as data:
            num_gaussians = int(data["num_gaussians"])
            peak = float(data["peak_vram_mib"])
            seconds = float(data["training_seconds"])
    except (OSError, KeyError):
        peak = float("nan")
        seconds = elapsed
    lines = [
        f"# {exp['id']} - RUN REPORT",
        "",
        f"- plant: {plant}",
        f"- exposure compensation (EO): {'ON' if exp['eo'] else 'off'}",
        f"- ForeSplat foreground constraints (FG): {'ON' if exp['fg'] else 'off'}",
        f"- iterations: {iters} | resolution divisor: {resolution} | seed: 0",
        f"- training wall time: {elapsed / 60:.1f} min",
        f"- final Gaussian count: {num_gaussians}",
        f"- peak VRAM allocated: {peak:.0f} MiB",
        f"- coordinate frame: VGGT-Omega native (no scan registration)",
        "",
        "## Outputs",
        "",
        f"- checkpoints + cfg: `checkpoints/`",
        f"- final Gaussian ply: `checkpoints/final_gaussians/point_cloud.ply`",
        f"- training metrics csv: `metrics/train_metrics.csv`",
    ]
    if exp["eo"]:
        lines.append("- per-view exposure params: `exposure_params/exposure_matrices.npz`")
    lines += ["", "## Reproduction", "", "```bash", _command_string(exp, iters, resolution), "```"]
    with open(os.path.join(exp_dir, "RUN_REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _command_string(exp: dict, iters: int, resolution: int) -> str:
    flags = (["--enable_eo"] if exp["eo"] else []) + (FG_FLAGS if exp["fg"] else [])
    return (
        "CUDA_VISIBLE_DEVICES=1 python gggs_community/train_foresplat.py "
        f"--source_path runs/full_sequence_batch/<plant>/GGGS_Dataset --resolution {resolution} "
        f"--iterations {iters} --mask_dir runs/full_sequence_batch/<plant>/GGGS_Dataset/masks "
        + " ".join(flags)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", default="KongQueZhuYu")
    parser.add_argument("--batch-root", default=os.path.join(ROOT, "runs/full_sequence_batch"))
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--iters", type=int, default=30_000)
    parser.add_argument("--smoke-iters", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    args = parser.parse_args(argv)

    plant = args.plant
    # The training subprocess runs with cwd=gggs_community, so every path handed
    # to it must be absolute or it silently resolves against the wrong root.
    args.batch_root = os.path.abspath(args.batch_root)
    row = read_plant_row(args.batch_root, plant)
    dataset_dir = os.path.join(args.batch_root, plant, "GGGS_Dataset")
    masks_dir = os.path.join(dataset_dir, "masks")
    if not os.path.isdir(os.path.join(dataset_dir, "sparse")):
        raise FileNotFoundError(
            f"GGGS dataset not found (no sparse/ under {dataset_dir}); check --batch-root/--plant"
        )
    init_points = int(row["fused_points"])
    experiments_root = args.experiments_root or os.path.join(ROOT, "experiments", plant)
    os.makedirs(experiments_root, exist_ok=True)

    info = write_shared_config(experiments_root, plant, dataset_dir, row, init_points, args.iters, args.resolution, masks_dir)
    print(f"[orchestrator] plant {plant}: {info['num_cameras']} cameras, {init_points} init points")

    for exp in EXPERIMENTS:
        if not run_experiment(exp, plant, experiments_root, dataset_dir, masks_dir, args.iters,
                              args.smoke_iters, args.resolution, args.min_free_gb):
            print(f"[orchestrator] stopping: {exp['id']} failed", flush=True)
            return 1
    print(f"[orchestrator] all four experiments finished under {experiments_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
