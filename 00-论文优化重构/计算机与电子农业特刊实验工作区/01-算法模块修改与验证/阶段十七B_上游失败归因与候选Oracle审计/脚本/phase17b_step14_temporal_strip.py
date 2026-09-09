#!/usr/bin/env python3
"""
Phase 17B — Step 14: Temporal strip visualization (DouBanLv2 0037-0041).
Shows GT + P6 raw instances + selected/final per frame in a horizontal strip.
Zero-inference: reads P00 masks + GT.
"""
from __future__ import annotations
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

FRAMES = [f"{i:04d}" for i in range(37, 42)]


def bmask(p):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (img > 127) if img is not None else None


def overlay(rgb, mask, color, alpha=0.5):
    ov = rgb.copy()
    for c, col in enumerate(color):
        ov[..., c] = np.where(mask, (1 - alpha) * ov[..., c] + alpha * col, ov[..., c])
    return ov


GEOM = [("RGB+GT+final", (0, 200, 0), (0, 0, 255)),
        ("RGB+final only", None, (0, 0, 255))]


def main() -> None:
    VIZ.mkdir(parents=True, exist_ok=True)
    sample = "DouBanLv2"
    n = len(FRAMES)
    cell = (320, 180)
    cell_w, cell_h = cell
    # 3 rows: GT row, final row, instance-count marker row
    rows = 3
    strip = np.full(((cell_h + 40) * rows, cell_w * n + 8, 3), 255, np.uint8)

    for j, frame in enumerate(FRAMES):
        stem = f"{sample}_{frame}"
        rgb = cv2.imread(str(INPUT_A / f"{stem}.jpg"))
        gt = bmask(GT_DIR / sample / f"mask_potted_clean_{frame}.png").astype(bool)
        fin = bmask(P00 / "最终掩膜" / f"mask_{stem}.png").astype(bool)

        # row 0: RGB + GT(overlay)
        t0 = overlay(cv2.resize(rgb, (cell_w, cell_h)), cv2.resize(gt.astype(np.uint8), (cell_w, cell_h)).astype(bool), (0, 200, 0))
        # row 1: RGB + final(overlay)
        t1 = overlay(cv2.resize(rgb, (cell_w, cell_h)), cv2.resize(fin.astype(np.uint8), (cell_w, cell_h)).astype(bool), (0, 0, 255))
        # row 2: RGB only (context), count in label
        t2 = cv2.resize(rgb, (cell_w, cell_h))

        for r_i, tile in enumerate((t0, t1, t2)):
            y0 = r_i * (cell_h + 40)
            strip[y0:y0 + cell_h, j * cell_w + 4:j * cell_w + 4 + cell_w] = tile
        cv2.putText(strip, stem[-4:], (j * cell_w + 6, cell_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        cv2.putText(strip, f"gt_area={int(gt.sum()/1000)}k", (j * cell_w + 6, cell_h + 30 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        # instance count from P00 候选评分明细
        import csv
        inst_n = 0
        with (P00 / "候选掩膜" / f"候选评分明细_{stem}.csv").open(encoding="utf-8-sig") as f:
            inst_n = sum(1 for _ in csv.DictReader(f))
        cv2.putText(strip, f"#inst={inst_n}", (j * cell_w + 6, cell_h + 30 + 40 + 1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 0, 0), 2)

    out = VIZ / "temporal_strip_DouBanLv2_0037-0041.jpg"
    cv2.imwrite(str(out), strip)
    print(f"temporal strip -> {out.name}")


if __name__ == "__main__":
    main()