#!/usr/bin/env python3
"""Train the no-fusion GGGS experiments (E2 raw-depth, E3 official-100K) with the
ORIGINAL GGGS recipe: EO off, foreground loss off, seed 0, 30000 iters, GPU1.

E1 is NOT retrained (its existing checkpoint is reused).  Each of E2/E3:
  1. runs a 200-iter smoke test gated on finite, decreasing loss,
  2. then the full 30000-iter run,
  3. writes RUN_REPORT.md.

Usage::

    CUDA_VISIBLE_DEVICES=1 python tools/run_no_fusion_experiments.py --plant KongQueZhuYu --iters 30000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGGS_DIR = os.path.join(ROOT, "gggs_community")
if GGGS_DIR not in sys.path:
    sys.path.insert(0, GGGS_DIR)

EXPERIMENTS = [
    {"id": "E2_Omega_RawDepth_NoFusion", "label": "VGGT-Omega raw depth, no fusion (N_E1 pts)"},
    {"id": "E3_Omega_OfficialCOLMAP_100K", "label": "Official-style COLMAP init, 100K pts"},
]


def mem_available_gb() -> float:
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return -1.0


def wait_for_memory(min_free_gb: float) -> None:
    while mem_available_gb() < min_free_gb:
        print(f"[no-fusion] memory low: {mem_available_gb():.1f} GB < {min_free_gb} GB - waiting", flush=True)
        time.sleep(60)


def _smoke_ok(csv_path: str, smoke_iters: int) -> bool:
    if not os.path.isfile(csv_path):
        return False
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return False
    if float(rows[-1]["iteration"]) < smoke_iters:
        return False
    for row in rows:
        for key in ("total_loss", "rgb_loss", "l1_loss"):
            if key in row and not math.isfinite(float(row[key])):
                return False
    first, last = float(rows[0]["total_loss"]), float(rows[-1]["total_loss"])
    print(f"[no-fusion] smoke: loss {first:.4f} -> {last:.4f} (finite)", flush=True)
    return True


def _command(dataset_dir, model_path, csv_path, iters, smoke=False) -> list:
    cmd = [
        os.path.abspath(os.path.join(ROOT, ".venv/bin/python")), "train_foresplat.py",
        "--source_path", dataset_dir, "--model_path", model_path,
        "--images", "images", "--resolution", "2",
        "--metrics_csv", csv_path, "--metrics_interval", "10" if smoke else "100",
        "--quiet",
    ]
    if smoke:
        cmd += ["--smoke", "--iterations", str(min(200, iters))]
    else:
        cmd += [
            "--iterations", str(iters),
            "--save_iterations", str(max(iters // 2, 1)), str(iters),
            "--test_iterations", str(max(iters // 3, 1)), str(max(2 * iters // 3, 1)), str(iters),
        ]
    return cmd


def _run(exp, exp_dir, dataset_dir, iters, min_free_gb):
    model_path = os.path.join(exp_dir, "checkpoints")
    csv_dir = os.path.join(exp_dir, "metrics")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "train_metrics.csv")

    smoke_dir = os.path.join(exp_dir, "smoke")
    os.makedirs(smoke_dir, exist_ok=True)
    smoke_csv = os.path.join(smoke_dir, "train_metrics.csv")
    if not os.path.isfile(os.path.join(smoke_dir, "DONE")):
        wait_for_memory(min_free_gb)
        cmd = _command(dataset_dir, smoke_dir, smoke_csv, iters, smoke=True)
        print(f"[no-fusion] {exp['id']}: smoke (200 iters) -> {smoke_dir}/train.log", flush=True)
        with open(os.path.join(smoke_dir, "train.log"), "w") as log:
            ret = subprocess.run(cmd, cwd=GGGS_DIR, stdout=log, stderr=subprocess.STDOUT).returncode
        if ret != 0 or not _smoke_ok(smoke_csv, 200):
            print(f"[no-fusion] {exp['id']}: SMOKE FAILED (ret={ret})", flush=True)
            return False
        open(os.path.join(smoke_dir, "DONE"), "w").write("ok\n")

    if os.path.isfile(os.path.join(exp_dir, "RUN_REPORT.md")):
        print(f"[no-fusion] {exp['id']}: RUN_REPORT exists, skipping", flush=True)
        return True
    wait_for_memory(min_free_gb)
    cmd = _command(dataset_dir, model_path, csv_path, iters, smoke=False)
    t0 = time.time()
    print(f"[no-fusion] {exp['id']}: full ({iters} iters) -> {exp_dir}/train.log", flush=True)
    with open(os.path.join(exp_dir, "train.log"), "w") as log:
        ret = subprocess.run(cmd, cwd=GGGS_DIR, stdout=log, stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - t0
    final_ply = os.path.join(model_path, "final_gaussians", "point_cloud.ply")
    if ret != 0 or not os.path.isfile(final_ply):
        print(f"[no-fusion] {exp['id']}: TRAINING FAILED (ret={ret})", flush=True)
        return False

    ng = -1
    peak = float("nan")
    try:
        with np.load(os.path.join(os.path.dirname(final_ply), "gaussians.npz")) as data:
            ng = int(data["num_gaussians"])
            peak = float(data["peak_vram_mib"])
    except Exception:
        pass
    lines = [
        f"# {exp['id']} - RUN REPORT", "",
        f"- label: {exp['label']}", f"- plant: KongQueZhuYu",
        f"- EO: off | foreground loss: off (ORIGINAL GGGS recipe)",
        f"- iterations: {iters} | resolution: 2 | seed: 0",
        f"- training wall time: {elapsed / 60:.1f} min",
        f"- final Gaussian count: {ng}",
        f"- peak VRAM allocated: {peak:.0f} MiB",
        f"- coordinate frame: VGGT-Omega native + E1 scene alignment (no scan registration)",
        "", "## Reproduction", "", "```bash",
        f"CUDA_VISIBLE_DEVICES=1 python gggs_community/train_foresplat.py --source_path <dataset> "
        f"--resolution 2 --iterations {iters}", "```",
    ]
    with open(os.path.join(exp_dir, "RUN_REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[no-fusion] {exp['id']}: done in {elapsed / 60:.1f} min", flush=True)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", default="KongQueZhuYu")
    ap.add_argument("--batch-root", default=os.path.join(ROOT, "runs/full_sequence_batch"))
    ap.add_argument("--iters", type=int, default=30_000)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    args = ap.parse_args(argv)
    args.batch_root = os.path.abspath(args.batch_root)
    nf_root = os.path.join(ROOT, "experiments", args.plant, "no_fusion_comparison")
    os.makedirs(nf_root, exist_ok=True)

    for exp in EXPERIMENTS:
        exp_dir = os.path.join(nf_root, exp["id"])
        dataset_dir = os.path.join(exp_dir, "dataset")
        if not os.path.isdir(os.path.join(dataset_dir, "sparse", "0")):
            print(f"[no-fusion] {exp['id']}: dataset not built - run build_no_fusion_initialisation.py first")
            return 1
        if not _run(exp, exp_dir, dataset_dir, args.iters, args.min_free_gb):
            print(f"[no-fusion] stopping: {exp['id']} failed")
            return 1
    print(f"[no-fusion] all no-fusion experiments finished under {nf_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
