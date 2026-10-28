#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score the 210-frame R4bm05 production masks against the 5 KQZ GT frames.

Reuses the exact production metric (segmentation_metrics + 15px leak band) so the
numbers are comparable with metrics/kqz5_R6_vs_splitGT.json.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/score_kqz5_r4bm05.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/data/fj/F2DMAS")
RECOVERY = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
GT = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/GT口径拆分审计/gt_masks_split_v2/KongQueZhuYu"
MASKS = RECOVERY / "runs/KongQueZhuYu_RAP_FULL_210_R4bm05/最终掩膜"
OUT = RECOVERY / "metrics/kqz5_r4bm05_vs_splitGT.json"
FRAMES = ["0000", "0025", "0050", "0075", "0100"]


def seg_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    # 15px boundary band leak ratio
    from scipy import ndimage
    dilated = ndimage.binary_dilation(gt, iterations=15)
    band = dilated & ~gt
    leak = int(np.logical_and(pred, band).sum())
    return {"f1": f1, "iou": iou, "precision": p, "recall": r,
            "leak_ratio": leak / max(int(band.sum()), 1),
            "gt_area": int(gt.sum())}


def main() -> int:
    if not MASKS.exists():
        print(f"[ERR] {MASKS} missing — run R4bm05 first")
        return 2
    frames = {}
    for fr in FRAMES:
        pred_p = MASKS / f"mask_{fr}.png"
        if not pred_p.exists():
            print(f"[skip] {fr}")
            continue
        pred = np.array(Image.open(pred_p).convert("L")) > 127
        gt = {}
        for conv in ["potted_plant", "plant_only", "pot", "blue_cube"]:
            # on-disk names: mask_potted_plant_*, mask_plant_*, mask_pot_*, mask_blue_cube_*
            stem_name = {"plant_only": "plant"}.get(conv, conv)
            # GT split masks are 2160x3840 (original frames) while the KQZ
            # 03-final_locked images (and hence the predictions) are 2120x3791.
            # Resize GT to the prediction grid with NEAREST so the comparison is
            # on the same pixel lattice; recorded in the JSON below.
            g_img = Image.open(GT / f"mask_{stem_name}_{fr}.png").convert("L")
            if g_img.size != (pred.shape[1], pred.shape[0]):
                g_img = g_img.resize((pred.shape[1], pred.shape[0]), Image.NEAREST)
            g = np.array(g_img) > 127
            gt[conv] = seg_metrics(pred, g)
        frames[fr] = {"pred_area": int(pred.sum()),
                      "gt": {k: v for k, v in gt.items()}}
    # aggregate
    agg = {}
    for conv in ["potted_plant", "plant_only", "pot"]:
        fs = [frames[f]["gt"][conv]["f1"] for f in frames]
        agg[conv] = {"f1_mean": float(np.mean(fs)),
                     "f1_min": float(np.min(fs)),
                     "f1_max": float(np.max(fs))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"produced_by": "score_kqz5_r4bm05.py",
                               "metric": "production seg_metrics + 15px leak band",
                               "gt_resize": "GT (2160x3840) resized to prediction grid "
                                            "(2120x3791) with PIL NEAREST",
                               "frames": frames, "aggregate": agg},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[written] {OUT}")
    print("aggregate potted_plant F1:", agg["potted_plant"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
