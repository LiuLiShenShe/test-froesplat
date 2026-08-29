#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段十一 §十 — 修复后指标汇总

对 3 失败帧 + 10 成功帧，用修复后的 per_instance 管线产物（最终掩膜）对照
GT口径拆分审计产出的 scoped GT，计算 plant_only / plant_plus_pot 两套口径的
IoU / F1 / Precision / Recall，并输出：
  - 修复后_三失败帧_指标.csv
  - 修复后_十成功帧_指标.csv
  - 修复后_全部帧_指标.csv

同时输出 oracle-vs-selected 差列占位（oracle 需原始候选，本脚本仅基于最终掩膜）。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

WORKDIR = Path(__file__).resolve().parent.parent  # 脚本在 交付物/ 下，父级即 阶段十一修复验证
GT_SPLIT = (WORKDIR.parent.parent.parent
            / "数据管理" / "07-运行脚本与超参" / "S20-RAP-FSAM3掩膜生成与验证"
            / "GT口径拆分审计" / "gt_masks_split")

# (sample, frame, 早前报告F1, 类别)
FAIL_FRAMES = [
    ("CaoMei1",      "0100", 0.229, "失败→修复"),
    ("ChangShouHua2","0100", 0.436, "失败→修复"),
    ("DouBanLv1",    "0000", 0.481, "失败→修复(GT混标)"),
]
SUCCESS_FRAMES = [
    ("CaoMei1",      "0025", 0.955, "成功"),
    ("CaoMei1",      "0050", 0.969, "成功"),
    ("ChangShouHua2","0000", 0.821, "成功"),
    ("ChangShouHua2","0025", 0.978, "成功"),
    ("ChangShouHua2","0050", 0.984, "成功"),
    ("DouBanLv1",    "0025", 0.974, "成功"),
    ("DouBanLv1",    "0050", 0.986, "成功"),
    ("DouBanLv1",    "0075", 0.977, "成功"),
    ("DouBanLv1",    "0100", 0.970, "成功"),
    ("XianKeLai1",   "0000", 0.971, "成功"),
]


def read_mask(p: Path) -> np.ndarray | None:
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return (img > 127).astype(np.uint8)


def confusion(gt: np.ndarray, pred: np.ndarray):
    if gt.shape != pred.shape:
        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                         interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())
    return tp, fp, fn, tn


def metrics(c):
    tp, fp, fn, tn = c
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return iou, f1, prec, rec


def eval_frame(sample: str, frame: str, final_rel: str) -> dict:
    final = WORKDIR / final_rel
    row = {"样本": sample, "帧": frame}
    for scope in ("plant_only", "plant_plus_pot"):
        gt = read_mask(GT_SPLIT / sample / f"mask_{scope}_{frame}.png")
        pred = read_mask(final)
        if gt is None or pred is None:
            row[f"{scope}_IoU"] = ""
            row[f"{scope}_F1"] = ""
            row[f"{scope}_Prec"] = ""
            row[f"{scope}_Recall"] = ""
            continue
        iou, f1, prec, rec = metrics(confusion(gt, pred))
        row[f"{scope}_IoU"] = round(iou, 4)
        row[f"{scope}_F1"] = round(f1, 4)
        row[f"{scope}_Prec"] = round(prec, 4)
        row[f"{scope}_Recall"] = round(rec, 4)
    return row


def main():
    out_dir = WORKDIR / "交付物" / "修复后指标"
    out_dir.mkdir(parents=True, exist_ok=True)

    fail_rows, succ_rows, all_rows = [], [], []

    for sample, frame, before, cat in FAIL_FRAMES:
        row = eval_frame(sample, frame, f"阶段证据_{sample}_{frame}/最终掩膜/mask_{frame}.png")
        row["早前F1"] = before
        row["类别"] = cat
        fail_rows.append(row)
        all_rows.append(row)

    for sample, frame, before, cat in SUCCESS_FRAMES:
        final_rel = f"回归_{sample}_{frame}/最终掩膜/mask_{frame}.png"
        if not (WORKDIR / final_rel).exists():
            print(f"  [跳过] 缺失回归产物: {final_rel}", file=sys.stderr)
            continue
        row = eval_frame(sample, frame, final_rel)
        row["早前F1"] = before
        row["类别"] = cat
        succ_rows.append(row)
        all_rows.append(row)

    fields = ["样本", "帧", "类别", "早前F1",
              "plant_only_IoU", "plant_only_F1", "plant_only_Prec", "plant_only_Recall",
              "plant_plus_pot_IoU", "plant_plus_pot_F1", "plant_plus_pot_Prec", "plant_plus_pot_Recall"]

    def write_csv(name, rows):
        p = out_dir / name
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  写出 {p.name}  ({len(rows)} 行)")

    write_csv("修复后_三失败帧_指标.csv", fail_rows)
    write_csv("修复后_十成功帧_指标.csv", succ_rows)
    write_csv("修复后_全部帧_指标.csv", all_rows)

    # 汇总最低 F1
    fails_po = [r["plant_only_F1"] for r in all_rows if isinstance(r.get("plant_only_F1"), float)]
    print(f"\n汇总: 全部 {len(all_rows)} 帧")
    if fails_po:
        print(f"  plant_only F1 区间: [{min(fails_po):.4f}, {max(fails_po):.4f}]")
    print("完成。")


if __name__ == "__main__":
    main()
