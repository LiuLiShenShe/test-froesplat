#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 16 — Deterministic sample-level split + candidate frame selection.

Produces Phase16_selection_manifest_preGT.csv.

Rules (all deterministic, no randomness):
  - 15 non-GT samples → 7 DEV / 8 TEST, each sample in exactly one split.
  - WangWenCao1 (COLMAP reg only ~70.6%) → DEV, never on locked TEST.
  - Paired species (ChangShouHua1/3, DouBanLv2/3, WangWenCao1/2,
    WanNianQing1/2, XiangPiShu1/2, XianKeLai2/3) are split 1-and-1 so that
    no species appears exclusively on one side.
  - Singles (BaiZhang, CaoMei2, HongZhang): BaiZhang → DEV to keep totals
    7/8; CaoMei2 and HongZhang → TEST.
  - Frame selection: prefer temporal coverage — for each sample, take its
    difficulty-ranked candidates and pick the hardest frame in each of up to
    4 temporal quartiles (max 4 per sample, fewer if unavailable).
  - Difficulty taxonomy: heuristic assignment from raw-image attributes only
    (model-independent). Annotator confirms during annotation.

Inputs:
  - Phase 15 difficulty_ranking.csv
  - scout_difficulty.py (frame_difficulty, _normalize) for samples with no
    candidates in the ranking (DouBanLv3) or for quartile re-ranking.

Output: 阶段十六_HardCase_GT构建与锁定/挑战集列表/Phase16_selection_manifest_preGT.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

PHASE15_SCRIPTS = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
                       "/01-算法模块修改与验证/阶段十五_硬案例挑战集与外部验证/脚本")
sys.path.insert(1, str(PHASE15_SCRIPTS))

WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
PHASE15 = WORKSPACE / "阶段十五_硬案例挑战集与外部验证"
PHASE16 = WORKSPACE / "阶段十六_HardCase_GT构建与锁定"
DATA = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理")
RAW_FRAMES = DATA / "01-输入图像/01-raw_frames"

from scout_difficulty import frame_difficulty, _normalize  # noqa: E402

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

# Per-attribute heuristic → difficulty taxonomy (raw-image only, model-independent)
D1 = "D1_neighbor_plant_adhesion"
D3 = "D3_foreground_background_confusion"
D6 = "D6_small_target"
D11 = "D11_cluttered_background"
D12 = "D12_low_contrast"
D14 = "D14_other"


def heuristic_taxonomy(a: dict) -> tuple[str, str]:
    """Map raw-image attributes to a primary/secondary difficulty label.

    Deterministic and model-independent. Annotator confirms during annotation.
    """
    gr = a.get("green_ratio", 0.5)
    exg = a.get("exg_sep", 1.0)
    ent = a.get("pixel_entropy", 0.5)
    lap = a.get("laplacian_var", 500.0)

    if gr < 0.05:
        return D6, D12          # very little green → small/sparse target
    if exg < 3.0:
        return D12, D3          # weak green/background separation → low contrast
    if ent > 0.80 and exg < 8.0:
        return D3, D11          # busy texture + weak separation → fg/bg confusion
    if gr > 0.60:
        return D1, D11          # dense green → neighbor adhesion / clutter
    if ent > 0.80:
        return D11, D3          # busy background
    if lap < 150.0:
        return D12, D14          # flat / blurry image
    return D14, ""


def compute_difficulty_for_sample(sample: str) -> list[dict]:
    """Compute composite difficulty for ~30 evenly-spaced frames of a sample."""
    d = RAW_FRAMES / sample
    jpgs = sorted(d.glob("*.jpg"))
    if not jpgs:
        return []
    total = len(jpgs)
    idxs = sorted({round(i * (total - 1) / 29) for i in range(30)})
    frames: list[dict] = []
    for i in idxs:
        p = jpgs[i]
        a = frame_difficulty(p)
        if not a:
            continue
        a["sample"] = sample
        a["frame"] = p.stem
        a["path"] = str(p)
        frames.append(a)
    if not frames:
        return []
    # normalize composite (same formula as scout_difficulty)
    lp = _normalize([f["laplacian_var"] for f in frames])
    en = _normalize([f["pixel_entropy"] for f in frames])
    ex = _normalize([f["exg_sep"] for f in frames])
    gr = _normalize([f["green_ratio"] for f in frames])
    dr = _normalize([f["dyn_range"] for f in frames])
    for f in frames:
        ls = 1.0 - lp[f["laplacian_var"]]
        es = en[f["pixel_entropy"]]
        xs = 1.0 - ex[f["exg_sep"]]
        grc = 2.0 * abs(gr[f["green_ratio"]] - 0.5)
        ds = 1.0 - dr[f["dyn_range"]]
        f["diff"] = 0.2 * (ls + es + xs + grc + ds)
    return frames


def select_targets(sample: str, candidates: list[dict], max_per_sample: int = 4) -> list[dict]:
    """Pick up to max_per_sample targets maximizing temporal coverage.

    Strategy: sort candidates by frame index, split into up to N temporal
    quartiles, take the hardest (max diff) candidate in each quartile.
    """
    if not candidates:
        return []
    cands = sorted(candidates, key=lambda c: int(c["frame"]))
    n = min(max_per_sample, len(cands))
    if n <= 1:
        return sorted(cands, key=lambda c: -c["diff"])[:n]
    selected: list[dict] = []
    boundaries = np.linspace(0, len(cands), n + 1).astype(int)
    seen = set()
    for i in range(n):
        lo, hi = int(boundaries[i]), int(boundaries[i + 1])
        if lo >= hi:
            continue
        quart = cands[lo:hi]
        # pick hardest unused candidate in this quartile
        quart_sorted = sorted(quart, key=lambda c: -c["diff"])
        pick = None
        for c in quart_sorted:
            if c["frame"] not in seen:
                pick = c
                break
        if pick is None:
            pick = quart_sorted[0]
        selected.append(pick)
        seen.add(pick["frame"])
    # fill remainder (if quartiles didn't produce enough) with hardest unused
    for c in sorted(cands, key=lambda c: -c["diff"]):
        if len(selected) >= max_per_sample:
            break
        if c["frame"] not in seen:
            selected.append(c)
            seen.add(c["frame"])
    return selected


def context_window(frame_idx: int, sample_total: int, margin: int = 5) -> tuple[int, int]:
    lo = max(0, frame_idx - margin)
    hi = min(sample_total - 1, frame_idx + margin)
    return lo, hi


def main() -> None:
    # Load Phase 15 ranking
    rank_path = PHASE15 / "挑战集列表/difficulty_ranking.csv"
    ranking: dict[str, dict[str, dict]] = {}  # sample -> frame -> attrs
    with open(rank_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            frame_key = Path(r["frame"]).stem   # strip .jpg suffix
            ranking.setdefault(r["sample"], {})[frame_key] = {
                "diff": float(r["diff"]),
                "laplacian_var": float(r["laplacian_var"]),
                "pixel_entropy": float(r["pixel_entropy"]),
                "exg_sep": float(r["exg_sep"]),
                "green_ratio": float(r["green_ratio"]),
                "dyn_range": float(r["dyn_range"]),
                "path": r["path"],
                "rank": int(r["rank"]),
            }

    rows: list[dict] = []
    excluded: list[dict] = []

    for sample in NON_GT_SAMPLES:
        split = SPLIT_ASSIGNMENT[sample]
        # candidate pool: from ranking, or freshly computed if absent
        cands_ranked = ranking.get(sample, {})
        if cands_ranked:
            candidates = [dict(v, sample=sample, frame=k) for k, v in cands_ranked.items()]
        else:
            computed = compute_difficulty_for_sample(sample)
            print(f"  [info] {sample}: no Phase-15 candidates; computed "
                  f"{len(computed)} new candidate difficulties")
            candidates = computed
            for c in computed:
                c.setdefault("rank", 999)

        # file count for context window clamping
        raw = RAW_FRAMES / sample
        jpgs = sorted(raw.glob("*.jpg")) if raw.exists() else []
        sample_total = len(jpgs)

        targets = select_targets(sample, candidates, max_per_sample=4)

        if not targets:
            excluded.append({"sample": sample, "split": split, "reason": "no readable candidates"})
            print(f"  [warn] {sample}: 0 targets selected (excluded from GT)")
            continue

        for c in targets:
            fidx = int(c["frame"])
            ctx_lo, ctx_hi = context_window(fidx, sample_total)
            prim, sec = heuristic_taxonomy(c)
            cid = f"HARD-{sample}-{c['frame']}"
            rows.append({
                "challenge_id": cid,
                "split": split,
                "sample": sample,
                "sequence": sample,          # sequence == sample here
                "frame": c["frame"],
                "image_path": c.get("path", str(raw / f"{c['frame']}.jpg")),
                "difficulty_rank": c.get("rank", ""),
                "difficulty_score": f"{c['diff']:.4f}",
                "primary_difficulty": prim,
                "secondary_difficulty": sec,
                "selection_reason": "hardest-in-temporal-quartile (raw-image difficulty)",
                "is_gt_target": "yes",
                "context_start": f"{ctx_lo:04d}",
                "context_end": f"{ctx_hi:04d}",
            })

    # deterministic ordering: by split then sample then frame
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

    # summary
    from collections import Counter
    split_cnt = Counter(r["split"] for r in rows)
    per_sample = Counter(r["sample"] for r in rows)
    print(f"\n=== Selection manifest ===")
    print(f"Total targets: {len(rows)}")
    print(f"By split: {dict(split_cnt)}")
    print("\nPer sample:")
    for s in sorted(per_sample):
        split = SPLIT_ASSIGNMENT[s]
        print(f"  [{split}] {s}: {per_sample[s]}")
    if excluded:
        print("\nExcluded samples:")
        for e in excluded:
            print(f"  {e['sample']}: {e['reason']}")
    print(f"\nSaved → {out_csv}")

    # sanity: each sample in exactly one split
    dev = {s for s, sp in SPLIT_ASSIGNMENT.items() if sp == "DEV"}
    test = {s for s, sp in SPLIT_ASSIGNMENT.items() if sp == "TEST"}
    assert dev & test == set(), "sample overlap DEV/TEST!"
    assert len(dev) == 7 and len(test) == 8, f"expected 7/8, got {len(dev)}/{len(test)}"
    assert set(SPLIT_ASSIGNMENT) == set(NON_GT_SAMPLES), "assignment mismatch!"
    print("\nSplit invariant OK: 7 DEV / 8 TEST, DEV∩TEST=∅, all 15 assigned.")


if __name__ == "__main__":
    main()
