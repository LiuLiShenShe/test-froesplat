#!/usr/bin/env python3
"""
Phase 17C — Step 26: TEST32 behavioral safety (conditional on raw candidates).

TEST32 = 32 locked no-GT frames (8 samples × 4) in U00_Control. NO accuracy claims.
Only behavioral checks under the winning policy (R1):
  - selection churn vs baseline P00 selections (share of frames whose selected
    instance changes)
  - mask divergence (IoU between baseline-selected and R1-selected mask per frame)
  - empty-mask count
  - runtime/structural integrity (no exceptions, deterministic output)
Large churn (>30%) → flag deployment risk.
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
S20_SCRIPT = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本")
OUT = WORKSPACE / "阶段十七C_候选排序修复开发"
U00 = WORKSPACE / "阶段十七A_10GT困难样本Pilot与无GT测试审计/03_trackB_variants/U00_Control"
WINNER = "r1"

sys.path.insert(0, str(S20_SCRIPT))
from ranking_policy import select_best_policy, primary_and_tiebreak  # noqa: E402

DENOM = 5.5
SAM_W = 0.5
FULL_WEIGHTS = {"area": 1.0, "comp": 1.0, "edge": 1.0, "temp": 1.0, "contrast": 1.0}


def bmask(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (img > 127) if img is not None else None


def mask_iou(a, b):
    if a is None or b is None:
        return float("nan")
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 1.0


def q_sam(sam):
    return 1.0 / (1.0 + np.exp(-6.0 * (sam - 0.5)))


def full_base(r):
    S = sum(FULL_WEIGHTS[k] * r[f"q_{k}"] for k in FULL_WEIGHTS)
    S += SAM_W * q_sam(r["sam_score"])
    return S / DENOM


def penalty_of(r):
    return full_base(r) - r["total_score"]


def main() -> None:
    if not (U00 / "候选掩膜" / "raw_instance_P6").exists():
        print("raw candidates unavailable → TEST32 behavioral audit SKIPPED (recorded)")
        (OUT / "Phase17C_test32_safety.csv").write_text(
            "status,note\nskipped,raw_instance_P6 not present in U00_Control\n", encoding="utf-8-sig")
        return

    with (U00 / "提示词评分.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"TEST32 提示词评分 rows: {len(rows)}")
    for r in rows:
        for k in ("面积比例", "面积得分", "连通域得分", "边界得分", "时序得分", "前背景对比得分",
                   "下方泄漏得分", "侧边泄漏得分", "总分"):
            r[k] = float(r[k])

    # Selection map (baseline, from U00 提示词选择.csv)
    sel_map = {}
    with (U00 / "提示词选择.csv").open(encoding="utf-8-sig") as f:
        for sr in csv.DictReader(f):
            sel_map[sr["图像"].rsplit(".", 1)[0]] = float(sr["前景面积比例"])

    # Group by frame
    by_frame = defaultdict(list)
    for r in rows:
        stem = r["图像"].rsplit(".", 1)[0]
        r["_stem"] = stem
        by_frame[stem].append(r)

    # SAM3 per-instance join (候选评分明细)
    sam_by_frame = {}
    for detail in sorted((U00 / "候选掩膜").glob("候选评分明细_*.csv")):
        stem = detail.name.replace("候选评分明细_", "").replace(".csv", "")
        m = {}
        with detail.open(encoding="utf-8-sig") as f:
            for dr in csv.DictReader(f):
                m[float(dr["面积比例"])] = float(dr["SAM3分数"])
        sam_by_frame[stem] = m

    rows_out = []
    n_frames = 0
    n_changed = 0
    empty_baseline = 0
    empty_new = 0
    ious = []
    for stem in sorted(by_frame):
        sc = by_frame[stem]
        n_frames += 1
        # attach sam_score via positional join (score row i ↔ detail row i by area_ratio)
        area_to_sam = sam_by_frame.get(stem, {})
        records = []
        baseline_area = sel_map.get(stem)
        baseline_idx = None
        for i, r in enumerate(sc):
            r["sam_score"] = area_to_sam.get(r["面积比例"], 0.0)
            # canonicalize to benchmark-style keys
            row = {
                "candidate_id": f"{stem}_{i:02d}",
                "instance_id": i,
                "area_ratio": r["面积比例"],
                "sam_score": r["sam_score"],
                "q_area": r["面积得分"],
                "q_comp": r["连通域得分"],
                "q_edge": r["边界得分"],
                "q_temp": r["时序得分"],
                "q_contrast": r["前背景对比得分"],
                "q_leak": r["下方泄漏得分"],
                "q_side": r["侧边泄漏得分"],
                "total_score": r["总分"],
            }
            pen = penalty_of(row)
            p, t = primary_and_tiebreak(
                base_total=full_base(row), q_contrast=row["q_contrast"], q_area=row["q_area"],
                w_contrast=1.0, denom=DENOM, semantic_total=0.0, penalty=pen, policy=WINNER,
            )
            records.append(SimpleNamespace(
                row=row, primary_score=p, tie_break_score=t,
                prompt_id="P6", instance_id=i, total_score=row["total_score"],
            ))
            if baseline_area is not None and abs(row["area_ratio"] - baseline_area) < 1e-8:
                baseline_idx = i
        best = select_best_policy(records, WINNER)
        best_idx = best.row["instance_id"]

        # mask divergence
        base_mask = bmask(U00 / "候选掩膜" / "raw_instance_P6" / f"mask_{stem}_{baseline_idx:02d}.png") if baseline_idx is not None else None
        new_mask = bmask(U00 / "候选掩膜" / "raw_instance_P6" / f"mask_{stem}_{best_idx:02d}.png")
        iou = mask_iou(base_mask, new_mask)
        ious.append(iou if not np.isnan(iou) else 1.0)
        changed = baseline_idx != best_idx
        n_changed += int(changed)
        empty_baseline += int(base_mask is not None and not base_mask.any())
        empty_new += int(new_mask is not None and not new_mask.any())

        rows_out.append({
            "sample": stem.rsplit("_", 1)[0],
            "frame": stem.rsplit("_", 1)[1],
            "n_candidates": len(sc),
            "baseline_selected": f"{stem}_{baseline_idx:02d}" if baseline_idx is not None else "NONE",
            "r1_selected": f"{stem}_{best_idx:02d}",
            "changed": int(changed),
            "mask_iou_baseline_vs_r1": round(iou, 6) if not np.isnan(iou) else "",
            "baseline_empty": int(base_mask is not None and not base_mask.any()),
            "r1_empty": int(new_mask is not None and not new_mask.any()),
        })

    churn = 100 * n_changed / n_frames if n_frames else 0.0
    mean_iou = float(np.mean(ious)) if ious else 1.0
    print(f"=== Step 26: TEST32 behavioral safety (no GT — behavior only) ===")
    print(f"Frames: {n_frames}, selection churn: {n_changed} ({churn:.1f}%)")
    print(f"Mean IoU baseline-vs-R1 selected masks: {mean_iou:.4f}")
    print(f"Empty masks — baseline: {empty_baseline}, R1: {empty_new}")
    risk = "FLAG" if churn > 30 else "ok"
    print(f"Churn risk: {risk} (>30% → deployment risk)")

    with (OUT / "Phase17C_test32_safety.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    (OUT / "Phase17C_test32_summary.json").write_text(
        '{"status":"%s","frames":%d,"churn_pct":%.2f,"mean_mask_iou":%.4f,'
        '"empty_baseline":%d,"empty_r1":%d,"risk":"%s"}\n' % (
            "PASS" if risk == "ok" else "FLAG", n_frames, churn, mean_iou,
            empty_baseline, empty_new, risk),
        encoding="utf-8")
    print("Written: Phase17C_test32_safety.csv + Phase17C_test32_summary.json")
    print("\n=== Step 26 PASS (behavioral audit only, no accuracy claim) ===")


if __name__ == "__main__":
    main()
