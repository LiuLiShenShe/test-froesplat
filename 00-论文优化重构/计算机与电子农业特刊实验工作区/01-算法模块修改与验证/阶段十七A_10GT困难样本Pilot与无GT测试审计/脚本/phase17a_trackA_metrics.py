#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 17A — Step 5: Track A metrics (accuracy + hardness, GT only).

For the 10 PILOT_GT10 frames × 4 P-variants, compute binary mask metrics vs
the human GT (GT_potted_clean_challenge). GT definition continuity:
potted_clean = potted & ~cube (challenge masks are cube_absent → potted_clean).

Per-frame metrics follow the Phase 12 canonical formulas (计算P6指标.py):
  tp/fp/fn/tn, precision, recall, F1, IoU, pred_area, gt_area, area_ratio.
Missing/empty prediction → F1=0 (Phase 12 fail-frame handling).

Material effect (frozen Phase 14/15 thresholds): ΔF1_vs_P00
  > +0.005 improvement | < -0.005 regression | |Δ| ≤ 0.005 neutral.

Hardness characterization: 10-frame P00 F1 vs easy-21 anchor
  (Phase 12: mean 0.9835, min 0.9761 — ceiling). Frames with F1 < 0.976
  count as "below easy ceiling" → hard-case confirmation.

Outputs (04_metrics/):
  trackA_per_frame.csv      — all variants × GT10 frames
  trackA_material.csv       — ΔF1 vs P00 + material class
  trackA_hardness.csv       — V00 F1 per frame + hardness flags
  trackA_summary.json       — means, material counts, hardness summary
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

PHASE17A = Path(__file__).resolve().parent.parent
WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
PHASE16 = WORKSPACE / "阶段十六_HardCase_GT构建与锁定"
GT_DIR = PHASE16 / "GT_potted_clean_challenge"
VARIANTS_DIR = PHASE17A / "02_trackA_variants"
OUT_DIR = PHASE17A / "04_metrics"

VARIANTS = ["P00", "P10", "P01", "P11"]
CONTROL = "P00"
VARIANT_DIRS = {"P00": "P00_Control", "P10": "P10_A6", "P01": "P01_A7", "P11": "P11_A6A7"}

# easy-21 anchor (Phase 12/14 frozen)
EASY_F1_MEAN = 0.9835
EASY_F1_MIN = 0.9761

MATERIAL_DELTA = 0.005


def load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img > 127


def compute_metrics(pred: np.ndarray | None, gt: np.ndarray) -> dict:
    if pred is None or not pred.any():
        return {
            "tp": 0, "fp": 0, "fn": int(gt.sum()), "tn": int((~gt).sum()),
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0,
            "pred_area": 0, "gt_area": int(gt.sum()), "area_ratio": 0.0,
            "pred_empty": True,
        }
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    iou = tp / max(tp + fp + fn, 1)
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 6), "recall": round(recall, 6),
        "f1": round(f1, 6), "iou": round(iou, 6),
        "pred_area": pred_area, "gt_area": gt_area,
        "area_ratio": round(pred_area / max(gt_area, 1), 4),
        "pred_empty": bool(not pred.any()),
    }


def final_mask_path(variant: str, sample: str, frame: str) -> Path:
    d = VARIANTS_DIR / VARIANT_DIRS[variant]
    return d / "最终掩膜" / f"mask_{sample}_{frame}.png"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # GT10 frames from the frozen manifest
    gt_rows = []
    with (PHASE17A / "GT10_manifest/Phase17A_GT10_manifest.csv").open(encoding="utf-8-sig") as f:
        gt_rows = list(csv.DictReader(f))
    frames = [(r["sample"], r["frame"], r["challenge_id"], r["difficulty_score"]) for r in gt_rows]

    rows = []
    for sample, frame, cid, dscore in sorted(frames):
        gt = load_mask(GT_DIR / sample / f"mask_potted_clean_{frame}.png")
        if gt is None:
            print(f"  ⚠ GT missing: {sample}/{frame}")
            continue
        for v in VARIANTS:
            p = final_mask_path(v, sample, frame)
            pred = load_mask(p)
            m = compute_metrics(pred, gt)
            rows.append({
                "challenge_id": cid, "sample": sample, "frame": frame,
                "variant": v,
                "gt_area": m["gt_area"], "difficulty_score": dscore,
                **m,
                "mask_path": str(p),
            })

    # ΔF1 vs control
    by_key = {(r["sample"], r["frame"], r["variant"]): r for r in rows}
    for r in rows:
        c = by_key.get((r["sample"], r["frame"], CONTROL))
        if c is not None:
            delta = r["f1"] - c["f1"]
            r["delta_f1_vs_p00"] = round(delta, 6)
            if delta > MATERIAL_DELTA:
                r["material"] = "improvement"
            elif delta < -MATERIAL_DELTA:
                r["material"] = "regression"
            else:
                r["material"] = "neutral"
        else:
            r["delta_f1_vs_p00"] = ""
            r["material"] = "n/a"

    # write per-frame CSV
    fields = ["challenge_id", "sample", "frame", "variant", "difficulty_score",
              "mask_path", "pred_empty", "tp", "fp", "fn", "tn",
              "precision", "recall", "f1", "iou", "pred_area", "gt_area",
              "area_ratio", "delta_f1_vs_p00", "material"]
    with (OUT_DIR / "trackA_per_frame.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # material counts per variant
    material_rows = []
    for v in VARIANTS:
        vr = [r for r in rows if r["variant"] == v and r["material"] != "n/a"]
        cnt = {"improvement": 0, "regression": 0, "neutral": 0}
        for r in vr:
            cnt[r["material"]] += 1
        d = [r["delta_f1_vs_p00"] for r in vr]
        material_rows.append({
            "variant": v, "n_frames": len(vr),
            "improvement": cnt["improvement"], "regression": cnt["regression"],
            "neutral": cnt["neutral"],
            "delta_min": round(min(d), 6) if d else "",
            "delta_max": round(max(d), 6) if d else "",
            "delta_mean": round(float(np.mean(d)), 6) if d else "",
        })
    with (OUT_DIR / "trackA_material.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(material_rows[0].keys()))
        w.writeheader()
        w.writerows(material_rows)

    # hardness: P00 F1 per frame vs easy anchor
    hardness_rows = []
    for r in rows:
        if r["variant"] != CONTROL:
            continue
        hardness_rows.append({
            "challenge_id": r["challenge_id"], "sample": r["sample"], "frame": r["frame"],
            "f1_p00": r["f1"], "iou_p00": r["iou"],
            "gt_area": r["gt_area"],
            "below_easy_min": int(r["f1"] < EASY_F1_MIN),
            "easy_anchor_min": EASY_F1_MIN,
            "easy_anchor_mean": EASY_F1_MEAN,
        })
    with (OUT_DIR / "trackA_hardness.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hardness_rows[0].keys()))
        w.writeheader()
        w.writerows(hardness_rows)

    # summary JSON
    summary = {}
    for v in VARIANTS:
        vr = [r for r in rows if r["variant"] == v]
        f1s = [r["f1"] for r in vr]
        ious = [r["iou"] for r in vr]
        summary[v] = {
            "f1_mean": round(float(np.mean(f1s)), 6),
            "f1_median": round(float(np.median(f1s)), 6),
            "f1_min": round(float(min(f1s)), 6),
            "f1_max": round(float(max(f1s)), 6),
            "iou_mean": round(float(np.mean(ious)), 6),
            "empty_pred_count": sum(1 for r in vr if r["pred_empty"]),
            "n": len(vr),
        }
    summary["material"] = {r["variant"]: {
        "improvement": r["improvement"], "regression": r["regression"], "neutral": r["neutral"],
    } for r in material_rows}
    p00 = [r["f1"] for r in rows if r["variant"] == CONTROL]
    summary["hardness"] = {
        "below_easy_min_count": sum(1 for f1 in p00 if f1 < EASY_F1_MIN),
        "easy_anchor_f1_min": EASY_F1_MIN,
        "easy_anchor_f1_mean": EASY_F1_MEAN,
        "n_gt10": len(p00),
    }
    summary["note"] = ("Pilot GT10 only (N=10, 2 sequences). Descriptive; "
                       "no significance claim. Track B has no GT and is excluded here.")
    (OUT_DIR / "trackA_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # console
    print("=== Track A metrics (GT10) ===")
    for v in VARIANTS:
        s = summary[v]
        print(f"  {v}: F1 mean={s['f1_mean']} median={s['f1_median']} min={s['f1_min']} max={s['f1_max']} IoU={s['iou_mean']} empty={s['empty_pred_count']}")
    print(f"  material: {json.dumps(summary['material'], ensure_ascii=False)}")
    print(f"  hardness: {json.dumps(summary['hardness'], ensure_ascii=False)}")
    print(f"\nSaved → 04_metrics/ (trackA_per_frame.csv, trackA_material.csv, trackA_hardness.csv, trackA_summary.json)")


if __name__ == "__main__":
    main()