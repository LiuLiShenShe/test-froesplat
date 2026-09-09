#!/usr/bin/env python3
"""
Phase 17B — Step 11: Instance audit visualization for failing frames.
Per-instance grid: [RGB | GT | each raw P6 instance | selected-oracle overlay].
Zero-inference: reads P00 raw instance masks + GT + RGB.
"""
from __future__ import annotations
import csv
from pathlib import Path
import cv2
import numpy as np

PHASE17A = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七A_10GT困难样本Pilot与无GT测试审计")
PHASE16  = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十六_HardCase_GT构建与锁定")
P00      = PHASE17A / "02_trackA_variants/P00_Control"
GT_DIR   = PHASE16 / "GT_potted_clean_challenge"
INPUT_A  = PHASE17A / "00_input_trackA"
OUT      = Path(__file__).resolve().parent.parent
VIZ      = OUT / "visualizations"

FAILING = [("DouBanLv2", "0039"), ("DouBanLv2", "0040"), ("DouBanLv2", "0041")]


def bmask(p):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (img > 127) if img is not None else None


def fm(a, b):
    if a is None or b is None:
        return 0.0, 0.0, 0.0, 0.0
    tp = int(np.logical_and(a, b).sum()); fp = int(a.sum()) - tp; fn = int(b.sum()) - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0; r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def read_instances(sample, frame):
    stem = f"{sample}_{frame}"
    rows = []
    with (P00 / "候选掩膜" / f"候选评分明细_{stem}.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({"instance_id": int(r["实例编号"]),
                         "sam_score": float(r["SAM3分数"]),
                         "area_ratio": float(r["面积比例"])})
    return rows


def read_selected(sample, frame):
    with (P00 / "提示词选择.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["图像"].rsplit(".", 1)[0] == f"{sample}_{frame}":
                fg = r.get("前景面积比例", "")
                return float(fg) if fg not in ("", "0") else None
    return None


def tile(label_imgs, labels, cols=4, cell=(640, 360)):
    """Build a grid of labeled image tiles."""
    n = len(label_imgs)
    rows = (n + cols - 1) // cols
    cell_w, cell_h = cell
    grid = np.full((rows * (cell_h + 34), cols * cell_w, 3), 255, np.uint8)
    for i, (img, lbl) in enumerate(zip(label_imgs, labels)):
        r, c = divmod(i, cols)
        y0, x0 = r * (cell_h + 34), c * cell_w
        img = cv2.resize(img, (cell_w, cell_h))
        grid[y0:y0 + cell_h, x0:x0 + cell_w] = img
        cv2.putText(grid, lbl, (x0 + 6, y0 + cell_h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    return grid


def main() -> None:
    VIZ.mkdir(parents=True, exist_ok=True)

    for sample, frame in FAILING:
        stem = f"{sample}_{frame}"
        rgb = cv2.imread(str(INPUT_A / f"{stem}.jpg"))
        gt = bmask(GT_DIR / sample / f"mask_potted_clean_{frame}.png").astype(np.uint8)
        # GT overlay tile (compute at native res, then downscale)
        gt_ov = rgb.copy()
        for c, col in enumerate((0, 200, 0)):
            gt_ov[..., c] = np.where(gt, 0.55 * gt_ov[..., c] + 0.45 * col, gt_ov[..., c])
        rgb = cv2.resize(rgb, (1280, 720))
        gt_ov = cv2.resize(gt_ov, (1280, 720))
        tiles = [rgb, gt_ov]
        labels = ["RGB", "GT (potted_clean)"]

        insts = read_instances(sample, frame)
        sel_area = read_selected(sample, frame)
        sel_id = next((d["instance_id"] for d in insts if abs(d["area_ratio"] - sel_area) < 1e-6), None) if sel_area else None

        f1_by_id = {}
        instance_tiles = []
        instance_labels = []
        for d in insts:
            raw = bmask(P00 / "候选掩膜" / "raw_instance_P6" / f"mask_{stem}_{d['instance_id']:02d}.png")
            f1 = fm(raw, gt)
            f1_by_id[d["instance_id"]] = f1
            m = cv2.resize(raw.astype(np.uint8), (1280, 720)).astype(bool)
            ov = rgb.copy()
            for c, col in enumerate((0, 0, 220)):
                ov[..., c] = np.where(m, 0.5 * ov[..., c] + 0.5 * col, ov[..., c])
            mark = " [SELECTED]" if d["instance_id"] == sel_id else ""
            star = " [ORACLE]" if f1 == f1_by_id[max(f1_by_id, key=f1_by_id.get)] else ""
            instance_tiles.append(ov)
            instance_labels.append(f"inst_{d['instance_id']:02d} F1={f1:.3f}{mark}{star} sam={d['sam_score']:.2f}")

        # order: RGB, GT, all instance tiles (with SELECTED/ORACLE markers), then wrap-around to fill
        tiles = [rgb, gt_ov] + instance_tiles
        labels_used = ["RGB", "GT (potted_clean)"] + instance_labels
        while len(tiles) < 4:
            tiles.append(cv2.bitwise_not(rgb))
            labels_used.append("")
        grid = tile(tiles, labels_used, cols=len(tiles))
        out_p = VIZ / f"instance_audit_{stem}.jpg"
        cv2.imwrite(str(out_p), grid)
        print(f"{stem}: {len(insts)} instances, selected={sel_id}, oracle={max(f1_by_id, key=f1_by_id.get)}, grid -> {out_p.name}")


if __name__ == "__main__":
    main()