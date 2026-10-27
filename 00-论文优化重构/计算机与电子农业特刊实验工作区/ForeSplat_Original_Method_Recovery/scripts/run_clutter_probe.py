#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe RAP-FSAM3 on the controlled multi-plant ("flowerbed") scenes.

Question under test (user's deployment scenario): in a garden where several
plants are visible and only the *front-most* one is the target, does the
SG-MPS-gated multi-prompt pipeline pick the right plant better than the
single-prompt P2 baseline?

Design
------
1. One pool run (`pool/`) over all composites with P1..P5 + per_instance +
   `--save_raw_instance_masks`, so every configuration below shares a byte-
   identical candidate pool and differs ONLY in the selection rule.
2. Per scene, three reuse runs (no SAM3 inference, ~10 s each):
      single_P2   --prompt_list P2 --prompt_selection_mode single --ranking_policy r1
      multi_nogate P1..P5, r1, area admissibility gate, NO semantic gate
      multi_gate   multi_nogate + SG-MPS gate (gated_softmax, tau 0.25)
3. Score every prediction against the constructed foreground GT, plus two
   clutter-specific diagnostics:
      leak_into_neighbours  fraction of the *neighbour* plants' pixels that the
                            prediction wrongly claims (the gate should lower it)
      outside_target        fraction of predicted pixels outside the target GT

Everything is measured, nothing is assumed.  If the gate does not help here,
the CSV will show it.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

ROOT = Path("/data/fj/F2DMAS")
RECOVERY = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
SCRIPTS = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本"
PROD = SCRIPTS / "生成RAP-FSAM3掩膜.py"
OUT = Path(os.environ.get("CLUTTER_ROOT", str(RECOVERY / "runs/clutter_scene")))
POOL = OUT / "pool"
INPUTS = OUT / "inputs"
INPUTS_ONE = OUT / "inputs_one"
GT = OUT / "gt"
LOGS = RECOVERY / "logs"
PYTHON = "/home/test/.venvs/sam3/bin/python"

SAM3 = ["--sam3_repo", str(ROOT / "第三方源码/SAM3-latest"),
        "--sam3_checkpoint", str(ROOT / "sam3/sam3.pt"),
        "--device", "cuda", "--force",
        "--candidate_mode", "per_instance",
        "--save_raw_instance_masks"]
MULTI = ["--prompt_list", "P1,P2,P3,P4,P5", "--default_prompt_id", "P2",
         "--use_prompt_ensemble", "--prompt_selection_mode", "score_select"]
SINGLE = ["--prompt_list", "P2", "--default_prompt_id", "P2",
          "--prompt_selection_mode", "single"]
ADM = ["--candidate_area_admissibility", "gate"]
GATE = ["--use_semantic_gate", "--semantic_gate_mode", "gated_softmax",
        "--semantic_gate_temperature", "0.25",
        "--semantic_gate_target_mode", "potted_plant"]

CONFIGS = {
    "single_P2":    SAM3 + SINGLE + ["--ranking_policy", "r1"],
    "multi_nogate": SAM3 + MULTI + ADM + ["--ranking_policy", "r1"],
    "multi_gate":   SAM3 + MULTI + ADM + GATE + ["--ranking_policy", "r1"],
}
POOL_CFG = SAM3 + MULTI + ["--ranking_policy", "r1"]


def sh(cmd: list[str], log: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS) + os.pathsep + env.get("PYTHONPATH", "")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"command": cmd}, ensure_ascii=False) + "\n")
        fh.flush()
        return subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh,
                              stderr=subprocess.STDOUT).returncode


def make_single_inputs(scenes: list[str]) -> None:
    for s in scenes:
        d = INPUTS_ONE / s
        d.mkdir(parents=True, exist_ok=True)
        link = d / f"{s}.jpg"
        target = INPUTS / f"{s}.jpg"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)


def seg_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"precision": p, "recall": r, "iou": iou, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def clutter_proxies(pred: np.ndarray, gt: np.ndarray, scene: str) -> dict:
    lab = label(pred, connectivity=2)
    props = regionprops(lab)
    areas = sorted((p.area for p in props), reverse=True)
    total = int(pred.sum())
    neigh_leak = 0.0
    neigh_area = 0
    for f in sorted(GT.glob(f"neigh_{scene}_*.png")):
        nm = np.array(Image.open(f).convert("L")) > 127
        neigh_area += int(nm.sum())
        neigh_leak += int(np.logical_and(pred, nm).sum())
    # mask out the target's own GT so a prediction correctly covering the target
    # body (which may overlap a neighbour's *visible* region by a few px) is not
    # penalised as leakage
    neigh_leak_frac = neigh_leak / max(neigh_area, 1)
    neigh_leak_masked = int(np.logical_and(pred & ~gt, nm).sum())
    neigh_leak_frac_masked = neigh_leak_masked / max(neigh_area, 1)
    return {
        "components": len(props),
        "largest_frac": (areas[0] / total) if areas and total else 0.0,
        "pred_pixels": total,
        "neighbour_leak_pixels": int(neigh_leak),
        "neighbour_leak_frac": neigh_leak_frac,
        "neighbour_leak_frac_masked": neigh_leak_frac_masked,
    }


def read_selection(report: Path) -> dict:
    if not report.exists():
        return {}
    rows = list(csv.DictReader(report.open(encoding="utf-8")))
    if not rows:
        return {}
    r = rows[0]
    return {"selected_prompt": r.get("selected_prompt", ""),
            "selected_instance": r.get("selected_instance", ""),
            "selection_path": r.get("selection_path", ""),
            "sg_mps_prompt": r.get("sg_mps_prompt", ""),
            "gate_conf": r.get("sg_mps_gate_confidence", ""),
            "cdr_policy": r.get("cdr_policy", ""),
            "sam3_candidates": r.get("sam3_candidates", "")}


def main() -> int:
    score_only = "--score-only" in sys.argv
    scenes = sorted(p.stem for p in INPUTS.glob("*.jpg"))
    if not scenes:
        print("[ERR] run build_clutter_scenes.py first")
        return 2
    make_single_inputs(scenes)
    LOGS.mkdir(parents=True, exist_ok=True)

    # 1. shared candidate pool
    if not score_only and not (POOL / "逐帧报告.csv").exists():
        print("[pool] running shared candidate pool ...", flush=True)
        t0 = time.monotonic()
        rc = sh([PYTHON, str(PROD), "--input_dir", str(INPUTS),
                 "--output_dir", str(POOL)] + POOL_CFG,
                LOGS / "clutter_pool.log")
        print(f"[pool] rc={rc} wall={time.monotonic()-t0:.0f}s", flush=True)
        if rc != 0:
            print("[ERR] pool run failed, see logs/clutter_pool.log")
            return 1
    else:
        print("[pool] reuse existing pool", flush=True)

    # 2. per-scene configurations
    rows = []
    for scene in scenes:
        for name, cfg in CONFIGS.items():
            out_dir = OUT / "runs" / scene / name
            report = out_dir / "逐帧报告.csv"
            if not report.exists():
                rc = sh([PYTHON, str(PROD),
                         "--input_dir", str(INPUTS_ONE / scene),
                         "--output_dir", str(out_dir)] + cfg
                        + ["--reuse_existing_candidates",
                           "--candidate_source_dir", str(POOL)],
                        LOGS / f"clutter_{scene}_{name}.log")
                print(f"[{scene}/{name}] rc={rc}", flush=True)
                if rc != 0:
                    continue
            mask_path = out_dir / "最终掩膜" / f"mask_{scene}.png"
            if not mask_path.exists():
                print(f"[{scene}/{name}] no final mask", flush=True)
                continue
            pred = np.array(Image.open(mask_path).convert("L")) > 127
            gt = np.array(Image.open(GT / f"mask_{scene}.png").convert("L")) > 127
            row = {"scene": scene, "config": name,
                   "fg_ratio_pred": float(pred.mean()),
                   "fg_ratio_gt": float(gt.mean())}
            row.update(seg_metrics(pred, gt))
            row.update(clutter_proxies(pred, gt, scene))
            row.update(read_selection(report))
            rows.append(row)

    if not rows:
        print("[ERR] no results")
        return 1
    fields = list(rows[0].keys())
    csv_path = OUT / "metrics" / "clutter_probe.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # 3. per-config aggregate
    agg = {}
    for name in CONFIGS:
        sel = [r for r in rows if r["config"] == name]
        if not sel:
            continue
        agg[name] = {
            "n": len(sel),
            "f1_mean": float(np.mean([r["f1"] for r in sel])),
            "iou_mean": float(np.mean([r["iou"] for r in sel])),
            "precision_mean": float(np.mean([r["precision"] for r in sel])),
            "recall_mean": float(np.mean([r["recall"] for r in sel])),
            "leak_mean": float(np.mean([r["neighbour_leak_frac"] for r in sel])),
            "leak_masked_mean": float(np.mean([r["neighbour_leak_frac_masked"] for r in sel])),
            "largest_frac_mean": float(np.mean([r["largest_frac"] for r in sel])),
            "components_mean": float(np.mean([r["components"] for r in sel])),
            "prompts": [r.get("selected_prompt", "") for r in sel],
        }
    (OUT / "metrics" / "clutter_probe_summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== summary ===")
    for k, v in agg.items():
        print(f"{k:14s} F1={v['f1_mean']:.4f} IoU={v['iou_mean']:.4f} "
              f"P={v['precision_mean']:.4f} R={v['recall_mean']:.4f} "
              f"leak={v['leak_mean']:.4f} leak_mask={v['leak_masked_mean']:.4f} "
              f"largest={v['largest_frac_mean']:.3f} "
              f"cc={v['components_mean']:.1f} prompts={v['prompts']}")
    print(f"\n[written] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
