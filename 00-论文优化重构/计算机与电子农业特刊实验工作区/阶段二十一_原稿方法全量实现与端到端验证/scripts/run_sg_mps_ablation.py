#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 21 — SG-MPS / ranking-policy ablation on the frozen A1 multi-prompt cache.

Replays the archived P1–P5 candidate masks for the 5 GT frames of KongQueZhuYu
through the *production* S20 script (no SAM3 inference), varying only:

  S0  ranking_policy=base   SG-MPS off         (historical A1 replay reference)
  S1  ranking_policy=r1     SG-MPS off
  S2  ranking_policy=r1     SG-MPS heuristic   (legacy sigmoid gate)
  S3  ranking_policy=r1     SG-MPS gated, tau=0.25
  S4  ranking_policy=r1     SG-MPS gated, tau=0.10
  S5  ranking_policy=r1     SG-MPS gated, tau=0.25, q_contrast in phi

Every run writes its own output dir plus an execution manifest (command, hashes,
git commit, wall time).  F1/IoU are computed against the frozen GT masks.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/data/fj/F2DMAS")
WORK = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区"
PHASE = WORK / "阶段二十一_原稿方法全量实现与端到端验证"
SCRIPTS = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本"
PROD = SCRIPTS / "生成RAP-FSAM3掩膜.py"
CACHE = ROOT / "00-论文优化重构/数据管理/03-分割Mask/05-RAP-FSAM3掩膜/E2E3_KongQueZhuYu_GT5_A1_多提示词选择"
INPUT = ROOT / "00-论文优化重构/数据管理/05-评测结果/S21_KongQueZhuYu_E2_E3/selected_frames"
GT = ROOT / "00-论文优化重构/数据管理/05-评测结果/S21_KongQueZhuYu_E2_E3/gt_masks"

CONFIGS = [
    ("S0_base_no_gate", ["--ranking_policy", "base"]),
    ("S1_r1_no_gate", ["--ranking_policy", "r1"]),
    ("S2_r1_heuristic_gate", ["--ranking_policy", "r1", "--use_semantic_gate",
                              "--semantic_gate_mode", "heuristic"]),
    ("S3_r1_gated_tau025", ["--ranking_policy", "r1", "--use_semantic_gate",
                            "--semantic_gate_mode", "gated_softmax",
                            "--semantic_gate_temperature", "0.25"]),
    ("S4_r1_gated_tau010", ["--ranking_policy", "r1", "--use_semantic_gate",
                            "--semantic_gate_mode", "gated_softmax",
                            "--semantic_gate_temperature", "0.10"]),
    ("S5_r1_gated_tau025_contrast", ["--ranking_policy", "r1", "--use_semantic_gate",
                                     "--semantic_gate_mode", "gated_softmax",
                                     "--semantic_gate_temperature", "0.25",
                                     "--semantic_gate_use_contrast"]),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def f1_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    denom = float(pred.sum() + gt.sum())
    f1 = 2.0 * inter / denom if denom else 1.0
    union = float(np.logical_or(pred, gt).sum())
    iou = inter / union if union else 1.0
    return f1, iou


def main() -> int:
    import argparse as _argparse
    ap = _argparse.ArgumentParser()
    ap.add_argument("--run-name", default="sg_mps_ablation")
    ap.add_argument("--only", default="", help="comma separated config prefixes")
    cli = ap.parse_args()
    only = [item for item in cli.only.split(",") if item]
    run_root = PHASE / "runs" / cli.run_name
    log_root = PHASE / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": "21",
        "dataset": "KongQueZhuYu_GT5",
        "candidate_cache": str(CACHE),
        "candidate_cache_sha256_of_production_script": sha256(PROD),
        "git_commit": git_commit(),
        "python": sys.executable,
        "input_dir": str(INPUT),
        "gt_dir": str(GT),
        "runs": [],
    }
    rows = []
    for name, extra in CONFIGS:
        if only and not any(name.startswith(prefix) for prefix in only):
            continue
        out = run_root / name
        if out.exists() and any(out.iterdir()):
            print(f"[skip] non-empty output exists: {out}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(PROD),
            "--input_dir", str(INPUT),
            "--output_dir", str(out),
            "--sam3_repo", str(ROOT / "第三方源码/SAM3-latest"),
            "--sam3_checkpoint", str(ROOT / "sam3/sam3.pt"),
            "--device", "cpu",
            "--candidate_mode", "per_instance",
            "--prompt_list", "P1,P2,P3,P4,P5",
            "--default_prompt_id", "P2",
            "--use_prompt_ensemble",
            "--prompt_selection_mode", "score_select",
            "--score_weights", "area=1,comp=1,edge=1,temp=1,contrast=1",
            "--candidate_source_dir", str(CACHE),
            "--reuse_existing_candidates",
            "--force",
        ] + extra
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = str(SCRIPTS) + os.pathsep + env.get("PYTHONPATH", "")
        log_path = log_root / f"sg_mps_{name}.txt"
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"command": cmd}, ensure_ascii=False) + "\n")
            handle.flush()
            result = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                    stdout=handle, stderr=subprocess.STDOUT)
        wall = time.monotonic() - started
        status = {"config": name, "command": cmd, "returncode": result.returncode,
                  "wall_sec": wall, "output": str(out), "log": str(log_path)}
        manifest["runs"].append(status)
        print(f"[{name}] rc={result.returncode} wall={wall:.1f}s", flush=True)
        if result.returncode != 0:
            continue
        for gt_path in sorted(GT.glob("mask_*.png")):
            stem = gt_path.stem.replace("mask_", "")
            pred_path = out / "最终掩膜" / f"mask_{stem}.png"
            if not pred_path.exists():
                print(f"  MISSING final mask: {pred_path}")
                continue
            pred = np.asarray(Image.open(pred_path).convert("L")) > 0
            gt = np.asarray(Image.open(gt_path).convert("L")) > 0
            if pred.shape != gt.shape:
                print(f"  SHAPE MISMATCH {stem}: {pred.shape} vs {gt.shape}")
                continue
            f1, iou = f1_iou(pred, gt)
            rows.append({"config": name, "frame": stem, "f1": f1, "iou": iou,
                         "pred_area": float(pred.mean()), "gt_area": float(gt.mean()),
                         "pred_sha256": sha256(pred_path), "gt_sha256": sha256(gt_path)})
    summary = []
    for name, _ in CONFIGS:
        sub = [r for r in rows if r["config"] == name]
        if not sub:
            continue
        summary.append({
            "config": name, "frames": len(sub),
            "f1_mean": float(np.mean([r["f1"] for r in sub])),
            "f1_min": float(np.min([r["f1"] for r in sub])),
            "iou_mean": float(np.mean([r["iou"] for r in sub])),
        })
    manifest["summary"] = summary
    (PHASE / "metrics").mkdir(exist_ok=True)
    (PHASE / "metrics" / "sg_mps_ablation.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    import csv
    with (PHASE / "metrics" / "sg_mps_ablation_per_frame.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=["config", "frame", "f1", "iou", "pred_area",
                                               "gt_area", "pred_sha256", "gt_sha256"])
        writer.writeheader()
        writer.writerows(rows)
    with (PHASE / "metrics" / "sg_mps_ablation_summary.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=["config", "frames", "f1_mean", "f1_min", "iou_mean"])
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
