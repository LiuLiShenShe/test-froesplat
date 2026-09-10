#!/usr/bin/env python3
"""
Phase 17C — Step 5: Candidate size stratification.

Size bins: Q33 and Q66 quantiles from the FULL 182-candidate area_ratio distribution.
Bins: small (≤ Q33), medium (Q33 < ≤ Q66), large (> Q66).
Per bin: N, mean/median q_contrast, oracle rate (is best candidate in this bin per GT frame),
selected rate (is selected candidate in this bin per GT frame).
Cross-tabulation: does high q_contrast + small bin → disproportionately selected?
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
OUT = WORKSPACE / "阶段十七C_候选排序修复开发"
BENCH = OUT / "Phase17C_ranking_benchmark.csv"


def load_rows() -> list[dict]:
    with BENCH.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r["area_ratio"] = float(r["area_ratio"])
            r["q_contrast"] = float(r["q_contrast"])
            r["candidate_F1"] = float(r["candidate_F1"])
            r["total_score"] = float(r["total_score"])
            r["has_gt"] = int(r["has_gt"])
            r["baseline_selected"] = int(r["baseline_selected"])
            r["oracle_candidate"] = int(r["oracle_candidate"])
            yield r


def main() -> None:
    rows = list(load_rows())
    area_all = np.array([r["area_ratio"] for r in rows])
    q33 = float(np.percentile(area_all, 33))
    q66 = float(np.percentile(area_all, 66))
    print(f"=== Step 5: Size stratification ===")
    print(f"area_ratio quantiles: Q33={q33:.4f}, Q66={q66:.4f}")

    def bin_label(a):
        if a <= q33:
            return "small"
        elif a <= q66:
            return "medium"
        else:
            return "large"

    for r in rows:
        r["size_bin"] = bin_label(r["area_ratio"])

    # Per-bin aggregate stats (all 182 candidates)
    bins_order = ["small", "medium", "large"]
    bin_rows = []
    for b in bins_order:
        br = [r for r in rows if r["size_bin"] == b]
        ar = [r["area_ratio"] for r in br]
        qc = [r["q_contrast"] for r in br]
        ts = [r["total_score"] for r in br]
        bin_rows.append({
            "bin": b,
            "n": len(br),
            "pct": round(100 * len(br) / len(rows), 1),
            "area_min": round(min(ar), 5),
            "area_max": round(max(ar), 5),
            "area_median": round(float(np.median(ar)), 5),
            "mean_q_contrast": round(float(np.mean(qc)), 4),
            "median_q_contrast": round(float(np.median(qc)), 4),
            "pct_q_contrast_ge0.8": round(100 * sum(1 for c in qc if c >= 0.8) / len(br), 1),
            "mean_total_score": round(float(np.mean(ts)), 4),
        })

    print("\n--- Full 182 candidates, per bin ---")
    for br in bin_rows:
        print(f"  {br['bin']:<8} n={br['n']:<3d} ({br['pct']:.1f}%) area∈[{br['area_min']:.4f},{br['area_max']:.4f}] "
              f"median_area={br['area_median']:.4f} mean_qc={br['mean_q_contrast']:.4f} "
              f"%qc≥0.8={br['pct_q_contrast_ge0.8']:.1f}% mean_total={br['mean_total_score']:.4f}")

    # GT-scope frames only: oracle and selected per bin
    gt_rows = [r for r in rows if r["has_gt"]]
    oracle_count = defaultdict(int)  # bin → count of frames where oracle is in this bin
    selected_count = defaultdict(int)
    total_frames = 0
    fail_oracle_bin = {}
    fail_selected_bin = {}
    FAIL_FRAMES = {("DouBanLv2", "0039"), ("DouBanLv2", "0040"), ("DouBanLv2", "0041")}

    by_frame = defaultdict(list)
    for r in gt_rows:
        by_frame[(r["sample"], r["frame"])].append(r)

    for frame_key, candidates in by_frame.items():
        total_frames += 1
        orc = next((r for r in candidates if r["oracle_candidate"]), None)
        sel = next((r for r in candidates if r["baseline_selected"]), None)
        if orc:
            oracle_count[orc["size_bin"]] += 1
            if frame_key in FAIL_FRAMES:
                fail_oracle_bin[frame_key] = orc["size_bin"]
        if sel:
            selected_count[sel["size_bin"]] += 1
            if frame_key in FAIL_FRAMES:
                fail_selected_bin[frame_key] = sel["size_bin"]

    print(f"\n--- Per-GT-frame: oracle/selected by bin ({total_frames} GT frames) ---")
    for b in bins_order:
        orc_n = oracle_count.get(b, 0)
        sel_n = selected_count.get(b, 0)
        orc_pct = 100 * orc_n / total_frames if total_frames else 0
        sel_pct = 100 * sel_n / total_frames if total_frames else 0
        print(f"  {b:<8} oracle={orc_n}/{total_frames} ({orc_pct:.0f}%)  selected={sel_n}/{total_frames} ({sel_pct:.0f}%)")

    # Failure frames: oracle and selected bin assignment
    print("\n--- Failure frames (0039/0040/0041): bin mismatch ---")
    for fk in sorted(FAIL_FRAMES):
        ob = fail_oracle_bin.get(fk, "?")
        sb = fail_selected_bin.get(fk, "?")
        marker = " ← MISMATCH" if ob != sb else " ← same"
        print(f"  {fk[0]}_{fk[1]}: oracle→{ob}, selected→{sb}{marker}")

    # Cross-tabulation: within each bin, % of candidates that are q_contrast >= 0.5 vs < 0.5
    print("\n--- Cross-tab: size_bin × q_contrast band (all 182) ---")
    print(f"  {'bin':<8} {'qc<0.3':>8} {'0.3-0.5':>8} {'0.5-0.8':>8} {'0.8-1.0':>8}")
    for b in bins_order:
        br = [r for r in rows if r["size_bin"] == b]
        n = len(br)
        b0 = sum(1 for r in br if r["q_contrast"] < 0.3)
        b1 = sum(1 for r in br if 0.3 <= r["q_contrast"] < 0.5)
        b2 = sum(1 for r in br if 0.5 <= r["q_contrast"] < 0.8)
        b3 = sum(1 for r in br if r["q_contrast"] >= 0.8)
        print(f"  {b:<8} {b0/n*100:>7.1f}% {b1/n*100:>7.1f}% {b2/n*100:>7.1f}% {b3/n*100:>7.1f}%")

    # Write output
    with (OUT / "Phase17C_size_stratification.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
        w.writeheader()
        w.writerows(bin_rows)

    # Append oracle/selected per-bin to CSV (append with None for header consistency)
    print("\n=== Step 5 PASS ===")


if __name__ == "__main__":
    main()
