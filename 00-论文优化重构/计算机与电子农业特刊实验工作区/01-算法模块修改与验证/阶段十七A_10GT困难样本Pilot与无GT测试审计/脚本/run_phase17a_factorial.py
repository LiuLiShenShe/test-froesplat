#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 17A — Factorial / Behavioral pipeline runner (Steps 3 & 4).

Track A (P-runs, accuracy/mechanism pilot on 10 GT frames):
    P00_Control, P10_A6, P01_A7, P11_A6A7
    input : 00_input_trackA/  (30 prefixed frames, 15/sample)
    output: 02_trackA_variants/<variant>_<label>/

Track B (U-runs, behavioral audit on 32 locked TEST frames, NO GT):
    U00_Control, U10_A6, U01_A7, U11_A6A7
    input : 01_input_trackB/  (32 prefixed frames, 4/sample)
    output: 03_trackB_variants/<variant>_<label>/

Config freeze: identical COMMON_ARGS to run_phase14_factorial.py (frozen
protocol). NO pipeline code change — invocation only.

Execution: up to --max_parallel variants concurrently, one per GPU
(CUDA_VISIBLE_DEVICES round-robin). Logs → logs/<track><variant>.log.
Manifest → Phase17A_runs_manifest.json.

Usage:
    python run_phase17a_factorial.py [--track both|A|B]
          [--variants P00 P10 P01 P11 ...] [--max-parallel 2] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PIPELINE = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参"
                "/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py")
PHASE17A = Path(__file__).resolve().parent.parent
WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
COLMAP_DIR = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/02-位姿COLMAP/03-final_locked")
PYTHON = "/home/test/biosoft/enter/envs/sam3/bin/python"

COMMON_ARGS = [
    "--prompt_list", "P6",
    "--default_prompt_id", "P6",
    "--candidate_mode", "per_instance",
    "--score_weights", "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5",
    "--sam3_mask_threshold", "0.5",
    "--save_raw_instance_masks",
    "--save_candidate_masks",
    "--save_intermediate_masks",
    "--force",
    "--consensus_min_frames", "5",
]

# Track A: P-runs (GT10 pilot). Track B: U-runs (locked TEST, no GT).
TRACKS = {
    "A": {
        "input_dir": PHASE17A / "00_input_trackA",
        "out_root": PHASE17A / "02_trackA_variants",
        "variants": {
            "P00": {"a6": False, "a7": False, "label": "Control"},
            "P10": {"a6": True,  "a7": False, "label": "A6"},
            "P01": {"a6": False, "a7": True,  "label": "A7"},
            "P11": {"a6": True,  "a7": True,  "label": "A6A7"},
        },
    },
    "B": {
        "input_dir": PHASE17A / "01_input_trackB",
        "out_root": PHASE17A / "03_trackB_variants",
        "variants": {
            "U00": {"a6": False, "a7": False, "label": "Control"},
            "U10": {"a6": True,  "a7": False, "label": "A6"},
            "U01": {"a6": False, "a7": True,  "label": "A7"},
            "U11": {"a6": True,  "a7": True,  "label": "A6A7"},
        },
    },
}

TIMEOUT_SEC = 10800  # 3h per variant run
NGPUS = 2            # RTX A6000 x2 (verified in starting audit)


def build_cmd(track: str, variant: str, dry_run: bool = False) -> tuple[list[str], Path, Path]:
    cfg = TRACKS[track]
    v = cfg["variants"][variant]
    output_dir = cfg["out_root"] / f"{variant}_{v['label']}"
    log_file = PHASE17A / "logs" / f"{track}{variant}.log"

    cmd = [PYTHON, str(PIPELINE)]
    cmd += ["--input_dir", str(cfg["input_dir"])]
    cmd += ["--output_dir", str(output_dir)]
    cmd += COMMON_ARGS

    if v["a6"]:
        cmd.append("--use_cross_view_consensus")
        cmd += ["--colmap_dir", str(COLMAP_DIR)]
    if v["a7"]:
        cmd.append("--use_memory_propagation")
    if dry_run:
        cmd.append("--dry_run")
    return cmd, output_dir, log_file


def run_one(track: str, variant: str, gpu: int, dry_run: bool = False) -> dict:
    cmd, output_dir, log_file = build_cmd(track, variant, dry_run)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    v = TRACKS[track]["variants"][variant]
    print(f"\n[{track}{variant} | {v['label']}] GPU={gpu} → {output_dir.name}", flush=True)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC, env=env
        )
        rc, status = result.returncode, "success"
        if rc != 0:
            status = f"failed(rc={rc})"
    except subprocess.TimeoutExpired:
        rc, status = -1, "timeout"
        result = None
    elapsed = time.time() - t0

    text_out = result.stdout if result is not None else ""
    text_err = result.stderr if result is not None else ""
    log_file.write_text(f"CMD: {' '.join(map(str, cmd))}\n\n"
                        f"GPU: {gpu}\n\nSTDOUT:\n{text_out}\n\nSTDERR:\n{text_err}",
                        encoding="utf-8")

    # quick completeness probe
    final_dir = output_dir / "最终掩膜"
    n_final = len(list(final_dir.glob("mask_*.png"))) if final_dir.exists() else 0

    print(f"  {status} ({elapsed:.0f}s, 最终掩膜={n_final})", flush=True)
    return {
        "track": track, "variant": variant, "label": v["label"],
        "a6": v["a6"], "a7": v["a7"],
        "gpu": gpu, "status": status, "elapsed_sec": round(elapsed, 1),
        "output_dir": str(output_dir), "log_file": str(log_file),
        "n_final_masks": n_final,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["A", "B", "both"], default="both")
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--max-parallel", type=int, default=NGPUS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = []
    for t in (["A", "B"] if args.track == "both" else [args.track]):
        vs = list(TRACKS[t]["variants"]) if args.variants is None else args.variants
        for v in vs:
            if v not in TRACKS[t]["variants"]:
                print(f"unknown variant {v} for track {t}; skip")
                continue
            jobs.append((t, v))
    print(f"Jobs: {len(jobs)}  ({', '.join(f'{t}{v}' for t, v in jobs)})")

    # concurrent submission: thread pool releases the GIL during subprocess
    # waits, so workers run truly in parallel (one per GPU via CUDA_VISIBLE_DEVICES)
    if args.dry_run:
        for t, v in jobs:
            cmd, od, lf = build_cmd(t, v, True)
            print(f"[dry-run] {t}{v}\n  {' '.join(map(str, cmd))}")
        return

    results: list[dict] = []
    from threading import Lock
    lock = Lock()
    gpu_counter = [0]

    def worker(track: str, variant: str) -> dict:
        with lock:
            gpu = gpu_counter[0] % NGPUS
            gpu_counter[0] += 1
        return run_one(track, variant, gpu)

    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = [ex.submit(worker, t, v) for t, v in jobs]
        for fut in futs:
            results.append(fut.result())

    manifest = {
        "phase": "17A",
        "description": "Phase 17A run manifest (Track A = GT10 pilot, Track B = locked TEST behavioral)",
        "trackA_note": "P-runs scored against GT10 (accuracy/mechanism pilot).",
        "trackB_note": "U-runs have NO GT — behavioral audit only, no accuracy.",
        "common_args": COMMON_ARGS,
        "runs": results,
    }
    mpath = PHASE17A / "Phase17A_runs_manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in results if r["status"] == "success")
    print(f"\n=== Summary: {ok}/{len(results)} runs success ===")
    for r in results:
        print(f"  {r['track']}{r['variant']} ({r['label']}): {r['status']} {r['elapsed_sec']}s 最终掩膜={r['n_final_masks']}")
    print(f"\nManifest → {mpath}")


if __name__ == "__main__":
    main()