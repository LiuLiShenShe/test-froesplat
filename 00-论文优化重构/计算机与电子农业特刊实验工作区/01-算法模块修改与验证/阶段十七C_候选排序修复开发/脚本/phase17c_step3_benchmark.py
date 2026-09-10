#!/usr/bin/env python3
"""
Phase 17C — Step 3: Ranking benchmark (DEV10 + Easy21, 182 candidates).
Reads:
  P00 提示词评分.csv  + 候选评分明细_* (ALL 30 frames = 157 candidates; 10 GT frames flagged)
  V00 提示词评分.csv  + 候选评分明细_* (21 frames, 25 candidates)
Outputs: Phase17C_ranking_benchmark.csv — every candidate with ALL q-components, GT metrics,
selection flags, and has_gt.

Schema (per candidate row):
  source, sample, frame, candidate_id, instance_id, has_gt, area_ratio, sam_score,
  q_area, q_comp, q_edge, q_temp, q_contrast, q_leak, q_side, total_score,
  candidate_F1, candidate_IoU, candidate_Prec, candidate_Rec,
  baseline_selected, oracle_candidate

- baseline_selected: the selected instance per frame, looked up via 提示词选择.csv 前景面积比例
  (area_ratio join, Phase 17B pattern). Computable for ALL frames (selection CSV covers all 30 P00
  frames + all 21 V00 frames).
- oracle_candidate: the instance with max candidate_F1 vs GT — computable ONLY where GT exists
  (10 DEV10 frames + 21 Easy21 frames). Context P00 frames -> 0 (no GT, no pseudo-GT).
- has_gt: True where a GT mask exists (10 DEV10 + 21 Easy21 frames).

Verification: total_rows == 157 + 25 == 182; every frame's selection is found; positional
join score<->detail verified by area_ratio tolerance < 1e-8.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
P00 = WORKSPACE / "阶段十七A_10GT困难样本Pilot与无GT测试审计/02_trackA_variants/P00_Control"
V00 = WORKSPACE / "阶段十四_A6A7真实数据析因消融/V00_Control"
GT10_DIR = WORKSPACE / "阶段十六_HardCase_GT构建与锁定/GT_potted_clean_challenge"
EASY21_GT_DIR = WORKSPACE / "阶段十二_GT_v2_QA与P6正式验收/GT_potted_clean"
OUT = WORKSPACE / "阶段十七C_候选排序修复开发"

DEV_GT_FRAMES = [("BaiZhang", f"{i:04d}") for i in range(33, 38)] + \
                [("DouBanLv2", f"{i:04d}") for i in range(37, 42)]
EASY21_FRAMES = [  # 5 samples, 21 frames
    (s, f"{fr:04d}") for s in ["CaoMei1", "ChangShouHua2", "DouBanLv1", "KongQueZhuYu"]
    for fr in [0, 25, 50, 75, 100]
] + [("XianKeLai1", "0000")]


def bmask(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (img > 127) if img is not None else None


def metrics(pred, gt):
    if gt is None or pred is None or not np.any(gt):
        return {"f1": 0.0, "iou": 0.0, "prec": 0.0, "rec": 0.0}
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(pred.sum()) - tp
    fn = int(gt.sum()) - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"f1": f1, "iou": iou, "prec": p, "rec": r}


def read_score_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_selection_map(path: Path) -> dict[str, float]:
    """{image_stem: 前景面积比例} from 提示词选择.csv."""
    sel = {}
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            sel[r["图像"].rsplit(".", 1)[0]] = float(r["前景面积比例"])
    return sel


def build_benchmark() -> list[dict]:
    rows: list[dict] = []
    n_positional_checked = 0
    n_positional_bad = 0

    for source_label, source_dir, gt_dir, gt_frames in [
        ("DEV10", P00, GT10_DIR, DEV_GT_FRAMES),
        ("Easy21", V00, EASY21_GT_DIR, EASY21_FRAMES),
    ]:
        score_rows = read_score_rows(source_dir / "提示词评分.csv")
        sel_map = read_selection_map(source_dir / "提示词选择.csv")

        # Group score rows by frame, preserving CSV order (which is rank order)
        by_frame: dict[str, list[dict]] = defaultdict(list)
        for r in score_rows:
            stem = r["图像"].rsplit(".", 1)[0]
            by_frame[stem].append(r)

        for stem, sc in sorted(by_frame.items()):
            sample, frame = stem.rsplit("_", 1)
            has_gt = (sample, frame) in gt_frames

            gt = None
            if has_gt:
                gt_path = gt_dir / sample / f"mask_potted_clean_{frame}.png"
                gt = bmask(gt_path)
                assert gt is not None, f"GT mask missing: {gt_path}"

            # Positional join with 候选评分明细 (same rank order as 提示词评分.csv)
            detail_path = source_dir / "候选掩膜" / f"候选评分明细_{stem}.csv"
            details: list[dict] = []
            if detail_path.exists():
                with detail_path.open(encoding="utf-8-sig") as f:
                    details = list(csv.DictReader(f))
            assert len(details) == len(sc), \
                f"{stem}: {len(sc)} score rows vs {len(details)} detail rows"

            # Selected instance via area_ratio join (Phase 17B pattern)
            selected_area = sel_map.get(stem)
            selected_idx: int | None = None
            if selected_area is not None:
                for i, r in enumerate(sc):
                    if abs(float(r["面积比例"]) - selected_area) < 1e-8:
                        selected_idx = i
                        break
            if has_gt:
                assert selected_idx is not None, f"{stem}: selected area_ratio {selected_area} not found"

            # Per-candidate GT metrics
            cand_f1s: list[float] = []
            for i, r in enumerate(sc):
                n_positional_checked += 1
                if abs(float(r["面积比例"]) - float(details[i]["面积比例"])) >= 1e-8:
                    n_positional_bad += 1
                inst_id = int(details[i]["实例编号"])
                raw_path = source_dir / "候选掩膜" / "raw_instance_P6" / f"mask_{stem}_{inst_id:02d}.png"
                pred = bmask(raw_path) if raw_path.exists() else None
                assert pred is not None, f"raw mask missing: {raw_path}"
                if has_gt:
                    m = metrics(pred, gt)
                else:
                    m = {"f1": float("nan"), "iou": float("nan"), "prec": float("nan"), "rec": float("nan")}
                cand_f1s.append(m["f1"])

            # Oracle = argmax F1 (only where GT exists)
            finite = [(i, x) for i, x in enumerate(cand_f1s) if not np.isnan(x)]
            oracle_idx = max(finite, key=lambda t: t[1])[0] if finite else None

            for i, r in enumerate(sc):
                inst_id = int(details[i]["实例编号"])
                sam_score = float(details[i].get("SAM3分数", 0.0) or 0.0)
                raw_path = source_dir / "候选掩膜" / "raw_instance_P6" / f"mask_{stem}_{inst_id:02d}.png"
                pred = bmask(raw_path) if raw_path.exists() else None
                if has_gt:
                    m = metrics(pred, gt)
                else:
                    m = {"f1": float("nan"), "iou": float("nan"), "prec": float("nan"), "rec": float("nan")}
                rows.append({
                    "source": source_label,
                    "sample": sample,
                    "frame": frame,
                    "candidate_id": f"{stem}_{inst_id:02d}",
                    "instance_id": inst_id,
                    "has_gt": int(has_gt),
                    "area_ratio": round(float(r["面积比例"]), 8),
                    "sam_score": round(sam_score, 6),
                    "q_area": float(r["面积得分"]),
                    "q_comp": float(r["连通域得分"]),
                    "q_edge": float(r["边界得分"]),
                    "q_temp": float(r["时序得分"]),
                    "q_contrast": float(r["前背景对比得分"]),
                    "q_leak": float(r["下方泄漏得分"]),
                    "q_side": float(r["侧边泄漏得分"]),
                    "total_score": float(r["总分"]),
                    "candidate_F1": m["f1"],
                    "candidate_IoU": m["iou"],
                    "candidate_Prec": m["prec"],
                    "candidate_Rec": m["rec"],
                    "baseline_selected": int(i == selected_idx) if selected_idx is not None else 0,
                    "oracle_candidate": int(i == oracle_idx) if oracle_idx is not None else 0,
                })

    print(f"Positional join checks: {n_positional_checked}, mismatches: {n_positional_bad}")
    assert n_positional_bad == 0, f"{n_positional_bad} positional mismatches (score<->detail)"
    return rows


def main() -> None:
    rows = build_benchmark()
    dev_rows = [r for r in rows if r["source"] == "DEV10"]
    easy_rows = [r for r in rows if r["source"] == "Easy21"]
    dev_gt_rows = [r for r in dev_rows if r["has_gt"]]
    print(f"DEV10 candidates:  {len(dev_rows)} (30 frames: {len(dev_gt_rows)} GT10 + {len(dev_rows)-len(dev_gt_rows)} context)")
    print(f"Easy21 candidates: {len(easy_rows)} (21 frames)")
    print(f"Total: {len(rows)} candidates (182 expected)")

    # Verifications
    assert len(dev_rows) == 157, f"DEV10 {len(dev_rows)} != 157"
    assert len(dev_gt_rows) == 30, f"DEV10 GT-scope {len(dev_gt_rows)} != 30"
    assert len(easy_rows) == 25, f"Easy21 {len(easy_rows)} != 25"
    assert len(rows) == 182, f"Total {len(rows)} != 182"

    # Exactly one selected per frame with a selection
    by_frame_sel = defaultdict(int)
    by_frame_orc = defaultdict(int)
    for r in rows:
        if r["baseline_selected"]:
            by_frame_sel[(r["source"], r["sample"], r["frame"])] += 1
        if r["oracle_candidate"]:
            by_frame_orc[(r["source"], r["sample"], r["frame"])] += 1
    assert all(v == 1 for v in by_frame_sel.values()), f"frames without exactly 1 selection: {[k for k,v in by_frame_sel.items() if v != 1]}"
    assert all(v == 1 for v in by_frame_orc.values()), f"frames without exactly 1 oracle: {[k for k,v in by_frame_orc.items() if v != 1]}"

    cols = ["source", "sample", "frame", "candidate_id", "instance_id", "has_gt", "area_ratio",
            "sam_score", "q_area", "q_comp", "q_edge", "q_temp", "q_contrast", "q_leak", "q_side",
            "total_score", "candidate_F1", "candidate_IoU", "candidate_Prec", "candidate_Rec",
            "baseline_selected", "oracle_candidate"]
    out = OUT / "Phase17C_ranking_benchmark.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"Written: {out}")

    print("\n--- Per-frame DEV10 (GT) summary ---")
    by_frame = defaultdict(list)
    for r in dev_gt_rows:
        by_frame[(r["sample"], r["frame"])].append(r)
    for k in sorted(by_frame):
        rows_k = sorted(by_frame[k], key=lambda r: r["total_score"], reverse=True)
        sel = next((r for r in rows_k if r["baseline_selected"]), rows_k[0])
        orc = next((r for r in rows_k if r["oracle_candidate"]), rows_k[0])
        print(f"  {k[0]}_{k[1]}: n={len(rows_k)} sel={sel['candidate_id']} F1={sel['candidate_F1']:.4f}  "
              f"oracle={orc['candidate_id']} F1={orc['candidate_F1']:.4f}  sel_is_oracle={sel['candidate_id']==orc['candidate_id']}  "
              f"regret={orc['candidate_F1']-sel['candidate_F1']:.6f}")

    print("\n--- Per-frame Easy21 summary ---")
    by_frame = defaultdict(list)
    for r in easy_rows:
        by_frame[(r["sample"], r["frame"])].append(r)
    for k in sorted(by_frame):
        rows_k = sorted(by_frame[k], key=lambda r: r["total_score"], reverse=True)
        sel = next((r for r in rows_k if r["baseline_selected"]), rows_k[0])
        orc = next((r for r in rows_k if r["oracle_candidate"]), rows_k[0])
        print(f"  {k[0]}_{k[1]}: n={len(rows_k)} sel={sel['candidate_id']} F1={sel['candidate_F1']:.4f}  "
              f"oracle={orc['candidate_id']} F1={orc['candidate_F1']:.4f}  sel_is_oracle={sel['candidate_id']==orc['candidate_id']}")

    print("\n=== Step 3 PASS (182 candidates) ===")


if __name__ == "__main__":
    main()
