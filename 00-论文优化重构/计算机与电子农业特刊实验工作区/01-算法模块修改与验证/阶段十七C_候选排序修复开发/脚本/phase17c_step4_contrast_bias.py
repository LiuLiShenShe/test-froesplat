#!/usr/bin/env python3
"""
Phase 17C — Step 4: Contrast bias analysis (Phase17C_contrast_bias.csv).

Hypothesis under test: q_contrast is systematically biased toward small masks
(whole-plant candidates span low-contrast leaf edges → low q_contrast; tiny
distractors sit in compact high-contrast blobs → q_contrast ≈ 1.0).

Evidence (all zero-inference, from Phase17C_ranking_benchmark.csv):
  - Spearman(area_ratio, q_contrast) over all 182 candidates
  - Same over subsets: DEV10 only, Easy21 only, failure frames (0039/0040/0041),
    oracle candidates only, selected candidates only
  - q_contrast distribution: oracle candidates vs non-oracle candidates (per-GT-frame)
  - q_contrast distribution: selected-correct (regret 0) vs selected-incorrect frames
  - Verdict per protocol: Spearman < −0.3 across full set → "contrast biased toward
    small masks" is SUPPORTED; single-frame examples insufficient alone.
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
OUT = WORKSPACE / "阶段十七C_候选排序修复开发"
BENCH = OUT / "Phase17C_ranking_benchmark.csv"

FAIL_FRAMES = {("DouBanLv2", "0039"), ("DouBanLv2", "0040"), ("DouBanLv2", "0041")}


def load_rows() -> list[dict]:
    with BENCH.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def spearman(x, y) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def main() -> None:
    rows = load_rows()
    for r in rows:
        r["area_ratio"] = float(r["area_ratio"])
        r["q_contrast"] = float(r["q_contrast"])
        r["candidate_F1"] = float(r["candidate_F1"])
        r["has_gt"] = int(r["has_gt"])
        r["baseline_selected"] = int(r["baseline_selected"])
        r["oracle_candidate"] = int(r["oracle_candidate"])

    out_rows = []
    def record(name, rows_subset, note=""):
        ar = [r["area_ratio"] for r in rows_subset]
        qc = [r["q_contrast"] for r in rows_subset]
        rho, p = spearman(ar, qc)
        n = len(rows_subset)
        n_small = sum(1 for a in ar if a < 0.06)
        qc_small = [c for a, c in zip(ar, qc) if a < 0.06]
        qc_large = [c for a, c in zip(ar, qc) if a >= 0.06]
        out_rows.append({
            "subset": name,
            "n": n,
            "spearman_rho": round(rho, 4) if not np.isnan(rho) else "",
            "spearman_p": round(p, 6) if not np.isnan(p) else "",
            "mean_q_contrast": round(float(np.mean(qc)), 4),
            "median_q_contrast": round(float(np.median(qc)), 4),
            "pct_q_contrast_ge_0.8": round(100 * sum(1 for c in qc if c >= 0.8) / n, 1) if n else "",
            "mean_q_contrast_small(a<0.06)": round(float(np.mean(qc_small)), 4) if qc_small else "",
            "mean_q_contrast_large(a>=0.06)": round(float(np.mean(qc_large)), 4) if qc_large else "",
            "note": note,
        })

    all_cands = rows
    dev = [r for r in rows if r["source"] == "DEV10"]
    easy = [r for r in rows if r["source"] == "Easy21"]
    fail = [r for r in dev if (r["sample"], r["frame"]) in FAIL_FRAMES]
    gt_rows = [r for r in rows if r["has_gt"]]
    oracle_cands = [r for r in gt_rows if r["oracle_candidate"]]
    non_oracle_cands = [r for r in gt_rows if not r["oracle_candidate"]]
    selected_cands = [r for r in gt_rows if r["baseline_selected"]]

    record("ALL candidates (DEV10+Easy21, 182)", all_cands)
    record("DEV10 only (157)", dev)
    record("Easy21 only (25)", easy)
    record("DEV10 GT-scope (30)", [r for r in dev if r["has_gt"]])
    record("Failure frames 0039/0040/0041 (14)", fail, "all 3 frames: selected spurious small instance")
    record("Oracle candidates (GT-scope)", oracle_cands, "the correct instance per GT frame")
    record("Non-oracle candidates (GT-scope)", non_oracle_cands, "wrong / partial instances")
    record("Selected candidates (GT-scope)", selected_cands, "what pipeline selected")

    # q_contrast: oracle vs non-oracle (per GT frame) — distribution comparison
    print("=== Step 4: Contrast bias ===")
    print("Spearman(area_ratio, q_contrast):")
    for r in out_rows:
        print(f"  {r['subset']:<45s} n={r['n']:<4d} rho={r['spearman_rho']:<7} p={r['spearman_p']:<9} "
              f"mean_qc={r['mean_q_contrast']:<6} %qc>=0.8={r['pct_q_contrast_ge_0.8']}")

    print("\n=== Oracle vs non-oracle q_contrast (GT-scope) ===")
    for name, grp in [("oracle", oracle_cands), ("non-oracle", non_oracle_cands)]:
        qc = [r["q_contrast"] for r in grp]
        ar = [r["area_ratio"] for r in grp]
        print(f"  {name:<12} n={len(qc)} mean_qc={np.mean(qc):.4f} median_qc={np.median(qc):.4f} "
              f"mean_area={np.mean(ar):.4f}")

    # Verdict: Spearman < -0.3 across full set → SUPPORTED
    rho_all = spearman([r["area_ratio"] for r in all_cands],
                       [r["q_contrast"] for r in all_cands])[0]
    verdict = ("SUPPORTED" if (not np.isnan(rho_all) and rho_all < -0.3)
               else "NOT SUPPORTED (weak/absent negative correlation)")
    print(f"\nVerdict: Spearman(full 182) = {rho_all:.4f} → contrast-bias claim {verdict}")
    for r in out_rows:
        r["verdict"] = verdict if r["subset"].startswith("ALL") else ""

    with (OUT / "Phase17C_contrast_bias.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("Written: Phase17C_contrast_bias.csv")
    print("\n=== Step 4 PASS ===")


if __name__ == "__main__":
    main()
