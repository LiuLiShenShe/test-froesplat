#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 15 — Difficulty Scout (raw-image-only difficulty attributes).

For each of the 15 non-GT samples, sample candidate frames evenly from the
full 250-frame video, compute raw-RGB difficulty attributes, rank by a
composite difficulty score (equal weights, each normalized to [0,1]),
and select a top list with a per-sample diversity constraint.

Fully §3-compliant: difficulty is derived ONLY from raw pixel statistics.
No segmentation output, no detection, no algorithm result is used.

Usage:
    python scout_difficulty.py [--frames_per_sample 30] [--top_n 60]
                               [--max_per_sample 5] [--out_csv path]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

RAW_DIR = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/01-输入图像/01-raw_frames")

# The 15 samples that have raw frames + COLMAP but NO GT (Phase 15 H2 pool).
NON_GT_SAMPLES = [
    "BaiZhang", "CaoMei2", "ChangShouHua1", "ChangShouHua3",
    "DouBanLv2", "DouBanLv3", "HongZhang",
    "WangWenCao1", "WangWenCao2", "WanNianQing1", "WanNianQing2",
    "XiangPiShu1", "XiangPiShu2", "XianKeLai2", "XianKeLai3",
]

WORK_H = 540  # downscale so shorter side ~= 540 px for attribute computation


def exg(bgr: np.ndarray) -> np.ndarray:
    """Excess Green index map, normalized to ~[-1, 2]. bgr uint8 -> float map."""
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    return 2.0 * g - r - b


def _normalize(values: list[float]) -> dict[float, float]:
    """min-max normalize to [0,1]; returns mapping value->norm (identity if constant)."""
    lo, hi = min(values), max(values)
    rng = hi - lo
    if rng < 1e-9:
        return {v: 0.5 for v in values}
    return {v: (v - lo) / rng for v in values}


def frame_difficulty(frame_path: Path, work_h: int = WORK_H) -> dict[str, float]:
    """Compute raw-image difficulty attributes for one frame.

    All attributes are scene-level pixel statistics (no segmentation).
    Returns:
      laplacian_var : sharpness (lower = blurrier = harder)
      pixel_entropy : texture complexity (higher = more complex = harder)
      exg_sep       : ExG foreground/background separation (lower = harder)
      green_ratio   : ExG>threshold area fraction (extreme low/high = harder)
      dyn_range     : luminance dynamic range (lower = flatter = harder)
      diff_score    : composite (higher = harder)
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        return {}
    h, w = img.shape[:2]
    scale = work_h / float(h)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), work_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness: variance of Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_var = float(lap.var())

    # 2. Texture complexity: shannon entropy of gray histogram (256 bins)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist[hist > 0].astype(np.float64)
    p = hist / hist.sum()
    pixel_entropy = float(-(p * np.log2(p)).sum() / np.log2(256.0))  # normalized [0,1]

    # 3. ExG separation: distance between foreground (green) and background ExG means
    exg_map = exg(img)
    thr = float(np.percentile(exg_map, 85))
    fg_mask = exg_map > thr
    bg_mask = ~fg_mask
    if fg_mask.sum() > 0 and bg_mask.sum() > 0:
        fg_mean = exg_map[fg_mask].mean()
        bg_mean = exg_map[bg_mask].mean()
        exg_sep = float(fg_mean - bg_mean)
    else:
        exg_sep = 0.0

    # 4. Green ratio: fraction of pixels that are clearly green (ExG > 20)
    green_ratio = float((exg_map > 20).mean())

    # 5. Dynamic range: luminance std / range (flatter = less range)
    lum_range = float(gray.max()) - float(gray.min())
    lum_std = float(gray.std())
    dyn_range = float(lum_std / max(lum_range, 1.0))

    return {
        "laplacian_var": laplacian_var,
        "pixel_entropy": pixel_entropy,
        "exg_sep": exg_sep,
        "green_ratio": green_ratio,
        "dyn_range": dyn_range,
    }


def _complexity_component(v: float, lo: float, hi: float) -> float:
    """map [lo,hi] -> [0,1] meaning higher = harder (clamped)."""
    if hi <= lo:
        return 0.5
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 15 raw-image difficulty scout")
    ap.add_argument("--frames_per_sample", type=int, default=30)
    ap.add_argument("--top_n", type=int, default=60)
    ap.add_argument("--max_per_sample", type=int, default=5)
    ap.add_argument("--out_csv", type=str, default=None)
    args = ap.parse_args()

    candidates: list[tuple[str, str, Path]] = []  # (sample, frame, path)
    for sample in NON_GT_SAMPLES:
        sdir = RAW_DIR / sample
        if not sdir.is_dir():
            print(f"  [warn] missing sample dir {sdir}")
            continue
        frames = sorted(p.name for p in sdir.glob("*.jpg"))
        total = len(frames)
        if total == 0:
            print(f"  [warn] no frames in {sdir}")
            continue
        # evenly spaced sampling
        n = min(args.frames_per_sample, total)
        idxs = {round(i * (total - 1) / (n - 1)) for i in range(n)} if n > 1 else {0}
        for i in sorted(idxs):
            candidates.append((sample, frames[i], sdir / frames[i]))

    print(f"Candidate pool: {len(candidates)} frames across "
          f"{len({c[0] for c in candidates})} samples")

    attrs: list[dict] = []
    for k, (sample, frame, path) in enumerate(candidates, 1):
        a = frame_difficulty(path)
        if not a:
            print(f"  [warn] unreadable {path}")
            continue
        a.update({"sample": sample, "frame": frame, "path": str(path)})
        attrs.append(a)
        if k % 100 == 0 or k == len(candidates):
            print(f"  computed {k}/{len(candidates)}")

    if not attrs:
        print("No attributes computed — abort")
        sys.exit(1)

    # --- normalize & composite ---
    lp = _normalize([a["laplacian_var"] for a in attrs])
    en = _normalize([a["pixel_entropy"] for a in attrs])
    ex = _normalize([a["exg_sep"] for a in attrs])
    gr = _normalize([a["green_ratio"] for a in attrs])
    dr = _normalize([a["dyn_range"] for a in attrs])

    for a in attrs:
        # Difficulty direction per attribute (equal weights):
        #  sharpness: LOWER = harder     -> use (1 - norm)
        #  entropy:   HIGHER = harder    -> use norm
        #  exg_sep:   LOWER = harder     -> use (1 - norm)
        #  green_ratio: extreme = harder -> distance from 0.5 (normalized to [0,1] band)
        #  dyn_range: LOWER = harder     -> use (1 - norm)
        ls = 1.0 - lp[a["laplacian_var"]]
        es = en[a["pixel_entropy"]]
        xs = 1.0 - ex[a["exg_sep"]]
        grc = 2.0 * abs(gr[a["green_ratio"]] - 0.5)
        ds = 1.0 - dr[a["dyn_range"]]
        a["diff"] = 0.2 * (ls + es + xs + grc + ds)
        a["c_laplacian"] = ls
        a["c_entropy"] = es
        a["c_exg_sep"] = xs
        a["c_green_ratio"] = grc
        a["c_dyn_range"] = ds

    attrs.sort(key=lambda a: a["diff"], reverse=True)

    # --- diversity-constrained selection: at least 1, at most max_per_sample/sample ---
    selected: list[dict] = []
    counts: dict[str, int] = {}
    for a in attrs:
        s = a["sample"]
        if counts.get(s, 0) >= args.max_per_sample:
            continue
        selected.append(a)
        counts[s] = counts.get(s, 0) + 1
        if len(selected) >= args.top_n:
            break

    print(f"\n=== Difficulty ranking: composite (higher=harder) ===")
    print(f"{'rank':>4} {'sample':<16} {'frame':<8} {'diff':>6} {'lap':>6} {'entr':>6} {'exg':>6} {'green':>6} {'dyn':>6}")
    for i, a in enumerate(selected, 1):
        print(f"{i:>4} {a['sample']:<16} {a['frame']:<8} {a['diff']:>6.3f} "
              f"{a['laplacian_var']:>6.0f} {a['pixel_entropy']:>6.3f} {a['exg_sep']:>6.1f} "
              f"{a['green_ratio']:>6.4f} {a['dyn_range']:>6.3f}")

    print(f"\nSamples represented: {len(counts)}")
    for s, c in sorted(counts.items()):
        print(f"  {s}: {c}")

    out_csv = args.out_csv or str(
        Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
             "/01-算法模块修改与验证/阶段十五_硬案例挑战集与外部验证/挑战集列表/difficulty_ranking.csv"))
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "sample", "frame", "diff", "laplacian_var", "pixel_entropy",
              "exg_sep", "green_ratio", "dyn_range", "c_laplacian", "c_entropy",
              "c_exg_sep", "c_green_ratio", "c_dyn_range", "path"]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, a in enumerate(selected, 1):
            a["rank"] = i
            w.writerow(a)
    print(f"\nSaved {len(selected)} candidates → {out_csv}")


if __name__ == "__main__":
    main()