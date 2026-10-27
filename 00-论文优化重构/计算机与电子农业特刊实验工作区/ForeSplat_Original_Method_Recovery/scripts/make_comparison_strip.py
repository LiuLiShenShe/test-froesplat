#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build human-viewable side-by-side comparison strips for the feasibility ladder.

For a given scene/frame it lays out, left to right:

    original RGB | GT (potted_plant) | R1s baseline | S1s | R4bm05 | R6m05

Each panel is the *mask as a red overlay on the original image* (the same
rendering the pipeline writes to 叠加图/, so what you see is what the pipeline
produced), except the GT panel which is green.

Output goes to runs/dev_ablation_v3/comparison/<scene>_<frame>.jpg at a readable
width, so one file shows the entire comparison without opening six windows.

Usage:
    python scripts/make_comparison_strip.py                      # 6 key frames
    python scripts/make_comparison_strip.py --scene CaoMei1 --frame 0025
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

RECOVERY = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery")
GT_SPLIT = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参/"
                "S20-RAP-FSAM3掩膜生成与验证/GT口径拆分审计/gt_masks_split_v2")
RUNS = RECOVERY / "runs/dev_ablation_v3"
INPUTS = RECOVERY / "dev_inputs"
OUT = RUNS / "comparison"

CONFIGS = [
    ("R1s baseline", "R1s_single_P2_cdr"),
    ("S1s", "S1s_single_P2_spnp_boxes"),
    ("R4bm05", "R4bm05_R3gm05_SPNP_boxes"),
    ("R6m05", "R6m05_full_with_margin"),
]

# best gain / worst loss frames found in metrics/dev_ablation_per_frame.csv
KEY_FRAMES = [
    ("CaoMei1", "0025"),        # +0.0091  best gain
    ("ChangShouHua2", "0050"),  # +0.0068
    ("ChangShouHua2", "0100"),  # +0.0050
    ("DouBanLv1", "0000"),      # +0.0028
    ("DouBanLv1", "0100"),      # -0.0024  loss
    ("ChangShouHua2", "0075"),  # -0.0061  worst loss
]

PANEL_W = 480
HEADER_H = 34


def load_rgb(scene: str, frame: str) -> Image.Image:
    return Image.open(INPUTS / scene / f"{frame}.jpg").convert("RGB")


def panel(img: Image.Image, mask: np.ndarray, color: tuple[int, int, int],
          alpha: float = 0.45) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    if mask is not None:
        m = mask.astype(bool)
        overlay = np.array(color, dtype=np.float32)
        arr[m] = arr[m] * (1 - alpha) + overlay * alpha
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    h = int(out.height * PANEL_W / out.width)
    return out.resize((PANEL_W, h), Image.LANCZOS)


def label(img: Image.Image, text: str) -> Image.Image:
    from PIL import ImageDraw
    canvas = Image.new("RGB", (img.width, img.height + HEADER_H), (24, 24, 24))
    canvas.paste(img, (0, HEADER_H))
    ImageDraw.Draw(canvas).text((8, 10), text, fill=(240, 240, 240))
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--frame")
    args = ap.parse_args()
    frames = [(args.scene, args.frame)] if args.scene and args.frame else KEY_FRAMES
    OUT.mkdir(parents=True, exist_ok=True)

    for scene, frame in frames:
        rgb = load_rgb(scene, frame)
        gt_path = GT_SPLIT / scene / f"mask_potted_plant_{frame}.png"
        gt = (np.array(Image.open(gt_path).convert("L")) > 127) if gt_path.exists() else None
        panels = [label(panel(rgb, gt, (40, 220, 90)), f"{scene}/{frame}  GT")]
        for name, cfg in CONFIGS:
            mp = RUNS / scene / cfg / "最终掩膜" / f"mask_{frame}.png"
            if not mp.exists():
                panels.append(label(panel(rgb, None, (255, 0, 0)), f"{name}\n(missing)"))
                continue
            m = np.array(Image.open(mp).convert("L")) > 127
            p = np.array(Image.open(mp).convert("L")) > 127
            g = gt if gt is not None else np.zeros_like(p)
            tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            panels.append(label(panel(rgb, m, (235, 60, 60)), f"{name}  F1={f1:.4f}"))
        h = max(p.height for p in panels)
        strip = Image.new("RGB", (sum(p.width for p in panels), h), (24, 24, 24))
        x = 0
        for p in panels:
            strip.paste(p, (x, 0)); x += p.width
        out = OUT / f"{scene}_{frame}.jpg"
        strip.save(out, quality=92)
        print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
