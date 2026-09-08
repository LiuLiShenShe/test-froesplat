#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 16 — Deterministic sample-level split + multi-plant density frame selection.

Produces Phase16_selection_manifest_preGT.csv.

核心目标：选出画面里有**多株植物共存**的帧（目标植株周围有邻居植株），
用于评估模型能否在复杂背景中精准分割目标个体。

选帧标准（multi-plant density）：
  - green_ratio：H[30-90] S[30-255] V[30-255] 的像素占比
  - horizontal_peaks：绿色列密度经平滑后的局部峰数（峰多=绿色在画面多处分布=多株植物）
  - score = green_ratio × n_peaks（越高=多株植物密度越大）

每 sample 取 score 最高的 4 帧。

Split 规则不变：
  - 15 non-GT samples → 7 DEV / 8 TEST
  - DEV: BaiZhang, ChangShouHua1, DouBanLv2, WangWenCao1, WanNianQing2, XiangPiShu1, XianKeLai2
  - TEST: CaoMei2, ChangShouHua3, DouBanLv3, HongZhang, WangWenCao2, WanNianQing1, XiangPiShu2, XianKeLai3

Output: 阶段十六_HardCase_GT构建与锁定/挑战集列表/Phase16_selection_manifest_preGT.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
PHASE16 = WORKSPACE / "阶段十六_HardCase_GT构建与锁定"
DATA = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理")
RAW_FRAMES = DATA / "01-输入图像/01-raw_frames"

NON_GT_SAMPLES = [
    "BaiZhang", "CaoMei2", "ChangShouHua1", "ChangShouHua3",
    "DouBanLv2", "DouBanLv3", "HongZhang", "WangWenCao1", "WangWenCao2",
    "WanNianQing1", "WanNianQing2", "XiangPiShu1", "XiangPiShu2",
    "XianKeLai2", "XianKeLai3",
]

# Deterministic, documented assignment (7 DEV / 8 TEST)
SPLIT_ASSIGNMENT: dict[str, str] = {
    "BaiZhang": "DEV",
    "ChangShouHua1": "DEV",
    "DouBanLv2": "DEV",
    "WangWenCao1": "DEV",
    "WanNianQing2": "DEV",
    "XiangPiShu1": "DEV",
    "XianKeLai2": "DEV",
    "CaoMei2": "TEST",
    "ChangShouHua3": "TEST",
    "DouBanLv3": "TEST",
    "HongZhang": "TEST",
    "WangWenCao2": "TEST",
    "WanNianQing1": "TEST",
    "XiangPiShu2": "TEST",
    "XianKeLai3": "TEST",
}

D1 = "D1_neighbor_plant_adhesion"
D11 = "D11_cluttered_background"
D14 = "D14_other"


def measure_multiplant_density(p: Path) -> dict | None:
    """Compute multi-plant density metrics for a single frame.

    Returns dict with green_pct, n_peaks, score (green_pct * n_peaks),
    or None if the frame is unreadable.
    """
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    mask_g = (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 90) & \
             (hsv[:, :, 1] >= 30) & (hsv[:, :, 2] >= 30)
    green_ratio = mask_g.sum() / (h * w)

    # horizontal green density profile → count peaks (= separate plant columns)
    col_density = mask_g.sum(axis=0) / h
    k = max(3, w // 80)
    kernel = np.ones(k) / k
    smoothed = np.convolve(col_density, kernel, mode="same")
    peaks, _ = find_peaks(smoothed, height=0.1, distance=w // 20)
    n_peaks = len(peaks)

    score = green_ratio * 100 * n_peaks  # green% * peaks
    return {"green_pct": green_ratio * 100, "n_peaks": n_peaks, "score": score}


def select_targets(sample: str, max_per_sample: int = 4) -> list[dict]:
    """Select the top frames by multi-plant density for a sample."""
    d = RAW_FRAMES / sample
    if not d.exists():
        return []
    jpgs = sorted(d.glob("*.jpg"))
    total = len(jpgs)
    frames: list[dict] = []
    for p in jpgs:
        m = measure_multiplant_density(p)
        if m is None:
            continue
        frames.append({
            "frame": p.stem,
            "path": str(p),
            **m,
        })
    if not frames:
        return []
    frames.sort(key=lambda r: r["score"], reverse=True)
    return frames[:max_per_sample]


def context_window(frame_idx: int, sample_total: int, margin: int = 5) -> tuple[int, int]:
    lo = max(0, frame_idx - margin)
    hi = min(sample_total - 1, frame_idx + margin)
    return lo, hi


def main() -> None:
    rows: list[dict] = []

    for sample in NON_GT_SAMPLES:
        split = SPLIT_ASSIGNMENT[sample]
        raw = RAW_FRAMES / sample
        jpgs = sorted(raw.glob("*.jpg")) if raw.exists() else []
        sample_total = len(jpgs)

        targets = select_targets(sample, max_per_sample=4)

        if not targets:
            print(f"  [warn] {sample}: 0 targets selected (excluded from GT)")
            continue

        for t in targets:
            fidx = int(t["frame"])
            ctx_lo, ctx_hi = context_window(fidx, sample_total)
            # difficulty label: green% > 20 → dense multi-plant; else moderate
            if t["green_pct"] >= 20:
                prim, sec = D1, D11
            else:
                prim, sec = D11, D14
            cid = f"HARD-{sample}-{t['frame']}"
            rows.append({
                "challenge_id": cid,
                "split": split,
                "sample": sample,
                "sequence": sample,
                "frame": t["frame"],
                "image_path": t["path"],
                "difficulty_rank": "",
                "difficulty_score": f"{t['score']:.1f}",
                "primary_difficulty": prim,
                "secondary_difficulty": sec,
                "selection_reason": "highest multi-plant density (green% x peaks)",
                "is_gt_target": "yes",
                "context_start": f"{ctx_lo:04d}",
                "context_end": f"{ctx_hi:04d}",
            })

    rows.sort(key=lambda r: (r["split"], r["sample"], r["frame"]))

    out_csv = PHASE16 / "挑战集列表/Phase16_selection_manifest_preGT.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["challenge_id", "split", "sample", "sequence", "frame",
              "image_path", "difficulty_rank", "difficulty_score",
              "primary_difficulty", "secondary_difficulty", "selection_reason",
              "is_gt_target", "context_start", "context_end"]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    split_cnt = Counter(r["split"] for r in rows)
    per_sample = Counter(r["sample"] for r in rows)
    print(f"\n=== Selection manifest (multi-plant density) ===")
    print(f"Total targets: {len(rows)}")
    print(f"By split: {dict(split_cnt)}")
    print("\nPer sample:")
    for s in sorted(per_sample):
        split = SPLIT_ASSIGNMENT[s]
        print(f"  [{split}] {s}: {per_sample[s]}")
    print(f"\nSaved → {out_csv}")

    dev = {s for s, sp in SPLIT_ASSIGNMENT.items() if sp == "DEV"}
    test = {s for s, sp in SPLIT_ASSIGNMENT.items() if sp == "TEST"}
    assert dev & test == set(), "sample overlap DEV/TEST!"
    assert len(dev) == 7 and len(test) == 8, f"expected 7/8, got {len(dev)}/{len(test)}"
    assert set(SPLIT_ASSIGNMENT) == set(NON_GT_SAMPLES), "assignment mismatch!"
    print("\nSplit invariant OK: 7 DEV / 8 TEST, DEV∩TEST=∅, all 15 assigned.")


if __name__ == "__main__":
    main()