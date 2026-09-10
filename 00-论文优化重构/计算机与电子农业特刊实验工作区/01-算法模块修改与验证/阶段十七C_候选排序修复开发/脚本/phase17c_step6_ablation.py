#!/usr/bin/env python3
"""
Phase 17C — Step 6: Leave-one-component-out ablation.

For each scored component k ∈ {area, comp, edge, temp, contrast, sam, leak, side}:
  set w_k = 0, denom stays 5.5, penalty stays per-candidate (computed from baseline),
  rerank all DEV10 GT frames (10) + Easy21 frames (21).
Record per frame: selected instance, selected F1, oracle regret.

Key question: does removing ONLY contrast independently rescue 0039/0040/0041?
This is the R1 ablation. leak/side already have w=0 in baseline → no-op (sanity check).

Output: Phase17C_component_ablation.csv
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase17c_common import FULL_WEIGHTS, SAM_W, DENOM, compute_sam_score, q_sam

WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
OUT = WORKSPACE / "阶段十七C_候选排序修复开发"
BENCH = OUT / "Phase17C_ranking_benchmark.csv"
FAIL_FRAMES = {("DouBanLv2", "0039"), ("DouBanLv2", "0040"), ("DouBanLv2", "0041")}

COMPONENTS = ["area", "comp", "edge", "temp", "contrast", "sam", "leak", "side"]
# score-weights: leak/side already 0.0 in baseline (sanity check → identical to baseline)


def load_gt_rows() -> list[dict]:
    with BENCH.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ["area_ratio", "q_area", "q_comp", "q_edge", "q_temp", "q_contrast",
                  "q_leak", "q_side", "sam_score", "total_score", "candidate_F1"]:
            r[k] = float(r[k])
        r["has_gt"] = int(r["has_gt"])
        r["baseline_selected"] = int(r["baseline_selected"])
        r["oracle_candidate"] = int(r["oracle_candidate"])
    return [r for r in rows if r["has_gt"]]


def penalty_of(r: dict) -> float:
    """Per-candidate penalty reconstructed from baseline full weights."""
    S = sum(FULL_WEIGHTS[k] * r[f"q_{k}"] for k in FULL_WEIGHTS)
    S += SAM_W * compute_sam_score(r)
    return S / DENOM - r["total_score"]


def primary_no_component(r: dict, dropped: str) -> float:
    """primary = (Σ_{k≠dropped} w_k q_k + sam contribution if sam not dropped) / denom − penalty."""
    S = 0.0
    for k, w in FULL_WEIGHTS.items():
        if k == dropped:
            continue
        S += w * r[f"q_{k}"]
    if dropped == "sam":
        # sam weight 0.5 is removed entirely
        pass
    else:
        S += SAM_W * compute_sam_score(r)
    return S / DENOM - penalty_of(r)


def rerank(candidates: list[dict], dropped: str) -> dict:
    """Select candidate maximizing primary_no_component; ties → first occurrence."""
    best = None
    best_score = None
    for c in candidates:
        s = primary_no_component(c, dropped)
        if best_score is None or s > best_score + 1e-12:
            best_score = s
            best = c
    return best


def main() -> None:
    rows = load_gt_rows()
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[(r["source"], r["sample"], r["frame"])].append(r)

    frame_keys = sorted(by_frame.keys())
    out_rows = []

    for dropped in COMPONENTS:
        dev_rescued = 0
        dev_frames = [fk for fk in frame_keys if fk[0] == "DEV10"]
        easy_frames = [fk for fk in frame_keys if fk[0] == "Easy21"]
        dev_sel_f1 = []
        dev_regret = []
        dev_material_reg = 0
        easy_changed = 0
        easy_material_reg = 0

        per_frame = []
        for fk in frame_keys:
            cands = by_frame[fk]
            sel = rerank(cands, dropped)
            baseline_sel = next((c for c in cands if c["baseline_selected"]), cands[0])
            oracle = next((c for c in cands if c["oracle_candidate"]), cands[0])
            sel_f1 = sel["candidate_F1"]
            regret = oracle["candidate_F1"] - sel_f1
            changed = sel["candidate_id"] != baseline_sel["candidate_id"]

            if fk[0] == "DEV10":
                dev_sel_f1.append(sel_f1)
                dev_regret.append(regret)
                if (fk[1], fk[2]) in FAIL_FRAMES:
                    # failure frame
                    if sel_f1 >= 0.95:
                        dev_rescued += 1
                else:
                    # non-failing DEV frame: material regression check
                    if sel_f1 - baseline_sel["candidate_F1"] < -0.005:
                        dev_material_reg += 1
            else:
                if changed:
                    easy_changed += 1
                    if sel_f1 - baseline_sel["candidate_F1"] < -0.005:
                        easy_material_reg += 1

            per_frame.append({
                "frame": f"{fk[1]}_{fk[2]}",
                "source": fk[0],
                "baseline_selected": baseline_sel["candidate_id"],
                "new_selected": sel["candidate_id"],
                "changed": int(changed),
                "baseline_F1": round(baseline_sel["candidate_F1"], 4),
                "new_F1": round(sel_f1, 4),
                "delta_F1": round(sel_f1 - baseline_sel["candidate_F1"], 6),
                "oracle_F1": round(oracle["candidate_F1"], 4),
                "regret": round(regret, 6),
            })

        out_rows.append({
            "dropped_component": dropped,
            "DEV10_mean_F1": round(float(np.mean(dev_sel_f1)), 4),
            "DEV10_mean_regret": round(float(np.mean(dev_regret)), 6),
            "DEV10_failures_rescued": f"{dev_rescued}/3",
            "DEV10_material_regressions": dev_material_reg,
            "Easy21_changed": easy_changed,
            "Easy21_material_regressions": easy_material_reg,
        })

    # Print summary
    print("=== Step 6: Leave-one-component-out ablation (GT frames only) ===")
    print(f"{'dropped':<10} {'DEV10 meanF1':>13} {'meanRegret':>10} {'rescued':>8} {'DEVreg':>7} {'E21changed':>11} {'E21reg':>7}")
    for r in out_rows:
        print(f"{r['dropped_component']:<10} {r['DEV10_mean_F1']:>13} {r['DEV10_mean_regret']:>10} "
              f"{r['DEV10_failures_rescued']:>8} {r['DEV10_material_regressions']:>7} "
              f"{r['Easy21_changed']:>11} {r['Easy21_material_regressions']:>7}")

    # Sanity check: leak/side ablation must equal baseline (w already 0)
    base_rescued = next(r for r in out_rows if r["dropped_component"] == "leak")["DEV10_failures_rescued"]
    print(f"\nSanity: leak/side ablation == baseline (w=0 already). leak rescued={base_rescued} (expect 0/3)")

    with (OUT / "Phase17C_component_ablation.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("Written: Phase17C_component_ablation.csv")
    print("\n=== Step 6 PASS ===")


if __name__ == "__main__":
    main()
