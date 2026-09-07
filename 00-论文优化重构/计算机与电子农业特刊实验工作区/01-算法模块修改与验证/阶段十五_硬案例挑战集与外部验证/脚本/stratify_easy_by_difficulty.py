#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 15 — Zero-compute difficulty stratification of the 21 easy frames.

Reuses Phase 14/14.1 artifacts (no new inference). For each of the 21 easy
frames, computes raw-image difficulty attributes from the frame_inputs source
(symlinks to 03-GT-区分), assigns a composite difficulty score (identical
formula to scout_difficulty.py), and stratifies into tertiles.

Output: 挑战集列表/21易帧难度分层.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from scout_difficulty import frame_difficulty, _normalize  # noqa: E402

PHASE12 = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
               "/01-算法模块修改与验证/阶段十二_GT_v2_QA与P6正式验收")
FRAME_INPUTS = PHASE12 / "frame_inputs"

EASY_FRAMES = {
    "CaoMei1": ["0000", "0025", "0050", "0075", "0100"],
    "ChangShouHua2": ["0000", "0025", "0050", "0075", "0100"],
    "DouBanLv1": ["0000", "0025", "0050", "0075", "0100"],
    "KongQueZhuYu": ["0000", "0025", "0050", "0075", "0100"],
    "XianKeLai1": ["0000"],
}

# Historical failure frames (H1) — all inside the 21 easy set
H1_FAILURE_FRAMES = {("CaoMei1", "0100"), ("ChangShouHua2", "0100"), ("DouBanLv1", "0000")}


def main() -> None:
    frames: list[dict] = []
    for sample, flist in EASY_FRAMES.items():
        for frame in flist:
            p = FRAME_INPUTS / f"{sample}_{frame}.jpg"
            if not p.exists():
                print(f"  [warn] missing {p}")
                continue
            a = frame_difficulty(p)
            if not a:
                print(f"  [warn] unreadable {p}")
                continue
            a.update({"sample": sample, "frame": frame, "stem": f"{sample}_{frame}",
                      "is_h1_failure": (sample, frame) in H1_FAILURE_FRAMES})
            frames.append(a)

    print(f"Computed difficulty for {len(frames)}/21 easy frames")

    # normalize + composite (same formula as scout)
    lp = _normalize([a["laplacian_var"] for a in frames])
    en = _normalize([a["pixel_entropy"] for a in frames])
    ex = _normalize([a["exg_sep"] for a in frames])
    gr = _normalize([a["green_ratio"] for a in frames])
    dr = _normalize([a["dyn_range"] for a in frames])
    for a in frames:
        ls = 1.0 - lp[a["laplacian_var"]]
        es = en[a["pixel_entropy"]]
        xs = 1.0 - ex[a["exg_sep"]]
        grc = 2.0 * abs(gr[a["green_ratio"]] - 0.5)
        ds = 1.0 - dr[a["dyn_range"]]
        a["diff"] = 0.2 * (ls + es + xs + grc + ds)

    frames.sort(key=lambda a: a["diff"], reverse=True)

    # tertiles
    n = len(frames)
    t = n // 3
    for i, a in enumerate(frames):
        if i < t:
            a["stratum"] = "hard"
        elif i < 2 * t:
            a["stratum"] = "mid"
        else:
            a["stratum"] = "easy"

    out_csv = SCRIPT_DIR.parent / "挑战集列表" / "21易帧难度分层.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["stratum", "rank", "stem", "sample", "frame", "diff",
              "laplacian_var", "pixel_entropy", "exg_sep", "green_ratio",
              "dyn_range", "is_h1_failure"]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, a in enumerate(frames, 1):
            a["rank"] = i
            w.writerow(a)

    print(f"\n=== 21 easy frames stratified by raw-image difficulty ===")
    print(f"{'stratum':<6} {'rank':>4} {'stem':<22} {'diff':>6} {'lap':>7} {'entr':>6} {'exg':>7} {'green':>6} {'H1?':>4}")
    for a in frames:
        h1 = "H1" if a["is_h1_failure"] else ""
        print(f"{a['stratum']:<6} {a['rank']:>4} {a['stem']:<22} {a['diff']:>6.3f} "
              f"{a['laplacian_var']:>7.0f} {a['pixel_entropy']:>6.3f} {a['exg_sep']:>7.1f} "
              f"{a['green_ratio']:>6.4f} {h1:>4}")

    # stratum summary
    from collections import Counter
    cnt = Counter(a["stratum"] for a in frames)
    h1_by_stratum = Counter(a["stratum"] for a in frames if a["is_h1_failure"])
    print(f"\nstratum counts: {dict(cnt)}")
    print(f"H1 failures per stratum: {dict(h1_by_stratum)}")
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()