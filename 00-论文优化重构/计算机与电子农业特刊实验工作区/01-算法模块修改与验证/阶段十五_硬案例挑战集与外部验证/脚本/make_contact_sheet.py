#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 15 — Contact sheet for difficulty-ranked candidates.

Builds a single montage image (grid of thumbnails labeled with rank/sample/
frame/difficulty) from difficulty_ranking.csv, so the user can visually review
the 60 candidate frames before selecting for annotation.

Usage:
    python make_contact_sheet.py [--csv path] [--out path] [--cols 5]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

THUMB_W, THUMB_H = 384, 216  # ~1/10 of 3840x2160
PAD = 8
LABEL_H = 28


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(Path(
        "/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
        "/01-算法模块修改与验证/阶段十五_硬案例挑战集与外部验证/挑战集列表/difficulty_ranking.csv")))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
    cols = args.cols
    rows_per_page = 12
    page_cells = cols * rows_per_page

    pages = [rows[i:i + page_cells] for i in range(0, len(rows), page_cells)]

    out_pngs = []
    for p, page in enumerate(pages, 1):
        cell_w = THUMB_W + 2 * PAD
        cell_h = THUMB_H + LABEL_H + 2 * PAD
        sheet_w = cols * cell_w
        sheet_h = rows_per_page * cell_h
        sheet = np.full((sheet_h, sheet_w, 3), 30, dtype=np.uint8)

        for i, r in enumerate(page):
            col, row = i % cols, i // cols
            x0 = col * cell_w + PAD
            y0 = row * cell_h + PAD
            img = cv2.imread(r["path"])
            if img is None:
                continue
            thumb = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
            sheet[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = thumb

            label = f"#{r['rank']} {r['sample']}_{r['frame'].split('.')[0]}  D={float(r['diff']):.3f}"
            cv2.rectangle(sheet, (x0, y0 + THUMB_H), (x0 + THUMB_W, y0 + THUMB_H + LABEL_H), (20, 20, 20), -1)
            cv2.putText(sheet, label, (x0 + 4, y0 + THUMB_H + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        out_png = args.out or str(Path(
            "/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
            "/01-算法模块修改与验证/阶段十五_硬案例挑战集与外部验证/挑战集列表") / f"candidates_p{p}.png")
        cv2.imwrite(out_png, sheet)
        out_pngs.append(out_png)
        print(f"page {p}: {len(page)} candidates → {out_png}")

    print(f"\nTotal pages: {len(pages)}")


if __name__ == "__main__":
    main()