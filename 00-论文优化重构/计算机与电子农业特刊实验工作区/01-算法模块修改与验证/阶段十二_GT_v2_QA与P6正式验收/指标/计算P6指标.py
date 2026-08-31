#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段十二 §五：P6 正式指标计算。
读取 P6_raw_baseline 最终掩膜与 GT_potted_clean，计算 IoU/F1/Precision/Recall 等。
输出：
  P6_全部21帧.csv
  P6_三失败帧.csv
  P6_回归帧.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

# ── 路径 ──
DELIVER = Path(__file__).resolve().parent.parent
P6_DIR = DELIVER / "P6_raw_baseline"
GT_CLEAN_DIR = DELIVER / "GT_potted_clean"
GT_SRC = Path("/data/fj/F2DMAS/03-GT-区分")
OUT_DIR = DELIVER / "指标"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAIL_FRAMES = {("CaoMei1", "0100"), ("ChangShouHua2", "0100"), ("DouBanLv1", "0000")}


def load_mask(path: Path) -> np.ndarray | None:
    """Load a grayscale mask PNG → bool array. Returns None if file missing."""
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img > 127


def load_gt_cube(sample: str, frame: str) -> np.ndarray | None:
    """Load blue_cube GT mask from the split v2 directory."""
    cube_path = (Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参"
                       "/S20-RAP-FSAM3掩膜生成与验证/GT口径拆分审计/gt_masks_split_v2"
                       f"/{sample}/mask_blue_cube_{frame}.png"))
    return load_mask(cube_path)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute binary mask metrics."""
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
    area_ratio = pred_area / max(gt_area, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "iou": round(iou, 6),
        "pred_area": pred_area,
        "gt_area": gt_area,
        "area_ratio": round(area_ratio, 4),
    }


def find_p6_output(sample: str, frame: str) -> Path | None:
    """Find the final P6 mask in P6_raw_baseline output directory.

    Pipeline structure: 最终掩膜/mask_{sample}_{frame}.png
    (Pass 2 A1s copies the P6 mask when A6/A7/SPNP are all OFF)
    """
    stem = f"{sample}_{frame}"
    base = P6_DIR
    # Primary: 最终掩膜/mask_{stem}.png
    final_dir = base / "最终掩膜"
    if final_dir.exists():
        candidate = final_dir / f"mask_{stem}.png"
        if candidate.exists():
            return candidate
    # Fallback: 选择后掩膜/mask_{stem}.png (Pass 1 output, same as final when no A6/A7)
    selected_dir = base / "选择后掩膜"
    if selected_dir.exists():
        candidate = selected_dir / f"mask_{stem}.png"
        if candidate.exists():
            return candidate
    # Fallback: search in subdirectories
    for subdir in [final_dir, selected_dir]:
        if subdir.exists():
            for p in subdir.glob(f"*{stem}*.png"):
                return p
    return None


def main():
    # Discover all samples/frames from GT_potted_clean
    frames = []
    for sample_dir in sorted(GT_CLEAN_DIR.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample = sample_dir.name
        for mask_path in sorted(sample_dir.glob("mask_potted_clean_*.png")):
            frame = mask_path.stem.replace("mask_potted_clean_", "")
            frames.append((sample, frame))

    print(f"找到 {len(frames)} 帧 GT potted_clean")

    rows = []
    for sample, frame in sorted(frames):
        gt_path = GT_CLEAN_DIR / sample / f"mask_potted_clean_{frame}.png"
        gt = load_mask(gt_path)
        if gt is None:
            print(f"  ⚠ GT 缺失: {sample}/{frame}")
            continue

        p6_path = find_p6_output(sample, frame)
        pred = load_mask(p6_path) if p6_path else None

        if pred is None:
            print(f"  ⚠ P6 输出缺失: {sample}/{frame}")
            rows.append({
                "sample": sample, "frame": frame,
                "p6_path": str(p6_path) if p6_path else "MISSING",
                "pred_empty": True, "f1": 0.0, "iou": 0.0,
                "precision": 0.0, "recall": 0.0,
                "pred_area": 0, "gt_area": int(gt.sum()),
                "area_ratio": 0.0,
                "cube_overlap_px": 0, "cube_overlap_ratio": 0.0,
                "is_fail_frame": int((sample, frame) in FAIL_FRAMES),
            })
            continue

        metrics = compute_metrics(pred, gt)

        # Cube overlap check
        cube_mask = load_gt_cube(sample, frame)
        cube_overlap = int((pred & cube_mask).sum()) if cube_mask is not None else 0
        cube_area = int(cube_mask.sum()) if cube_mask is not None else 0
        cube_overlap_ratio = cube_overlap / max(cube_area, 1)

        row = {
            "sample": sample, "frame": frame,
            "p6_path": str(p6_path),
            "pred_empty": bool(not pred.any()),
            **metrics,
            "cube_overlap_px": cube_overlap,
            "cube_overlap_ratio": round(cube_overlap_ratio, 6),
            "is_fail_frame": int((sample, frame) in FAIL_FRAMES),
        }
        rows.append(row)
        tag = "★" if (sample, frame) in FAIL_FRAMES else " "
        print(f"  {tag} {sample}_{frame}: F1={metrics['f1']:.4f} IoU={metrics['iou']:.4f} "
              f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"area_ratio={metrics['area_ratio']:.3f}")

    # Write full CSV
    fields = ["sample", "frame", "p6_path", "pred_empty",
              "tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou",
              "pred_area", "gt_area", "area_ratio",
              "cube_overlap_px", "cube_overlap_ratio", "is_fail_frame"]
    csv_all = OUT_DIR / "P6_全部21帧.csv"
    with csv_all.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n全部帧 CSV → {csv_all}")

    # Split by fail / regression
    fail_rows = [r for r in rows if r["is_fail_frame"]]
    reg_rows = [r for r in rows if not r["is_fail_frame"]]

    for name, subset in [("P6_三失败帧", fail_rows), ("P6_回归帧", reg_rows)]:
        csv_path = OUT_DIR / f"{name}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(subset)
        print(f"  {name} CSV → {csv_path} ({len(subset)} 帧)")

    # Summary stats
    valid = [r for r in rows if not r["pred_empty"]]
    if valid:
        f1s = [r["f1"] for r in valid]
        ious = [r["iou"] for r in valid]
        cube_overlaps = [r["cube_overlap_ratio"] for r in valid]
        print(f"\n=== 汇总 ({len(valid)} 帧有效) ===")
        print(f"  F1  均值={np.mean(f1s):.4f} 中位={np.median(f1s):.4f} "
              f"最差={min(f1s):.4f} 最好={max(f1s):.4f}")
        print(f"  IoU 均值={np.mean(ious):.4f} 中位={np.median(ious):.4f}")
        print(f"  cube误覆盖 均值={np.mean(cube_overlaps):.6f} 最大={max(cube_overlaps):.6f}")

        # Per-sample
        samples = sorted(set(r["sample"] for r in valid))
        print("\n  按样本:")
        for s in samples:
            sf = [r["f1"] for r in valid if r["sample"] == s]
            print(f"    {s}: F1均值={np.mean(sf):.4f} ({len(sf)}帧)")

    # Write summary JSON
    summary = {
        "total_frames": len(rows),
        "valid_frames": len(valid),
        "empty_frames": len([r for r in rows if r["pred_empty"]]),
        "f1_mean": round(float(np.mean(f1s)), 6) if valid else 0,
        "f1_median": round(float(np.median(f1s)), 6) if valid else 0,
        "f1_min": round(float(min(f1s)), 6) if valid else 0,
        "f1_max": round(float(max(f1s)), 6) if valid else 0,
        "iou_mean": round(float(np.mean(ious)), 6) if valid else 0,
        "cube_overlap_max": round(float(max(cube_overlaps)), 6) if cube_overlaps else 0,
    }
    (OUT_DIR / "P6_汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
