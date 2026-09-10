#!/usr/bin/env python3
"""
Phase 17C — Step 8: Policy comparison R0/R1/R2/R3 + decision (decision order §14).

For each policy ∈ {R0, R1, R2, R3}:
  - DEV10 (10 GT frames): per-frame selected instance, selected F1, oracle regret,
    rescued failures (0039/0040/0041, F1 ≥ 0.95), material regressions (ΔF1 < −0.005 on
    the 7 non-failing frames), mean F1, mean regret, top-1 accuracy.
  - Easy21 (21 frames): selection changes, F1 improved/neutral/regressed, material
    regressions (ΔF1 < −0.005), churn rate.
Decision order (§14, strict):
  1. Rescue all three C3 failures (selected F1 ≥ 0.95)
  2. Zero material regression on remaining DEV10 (ΔF1 < −0.005)
  3. Zero material regression on Easy21 (ΔF1 < −0.005)
  4. Lowest oracle regret
  5. Lowest selection churn
  6. Simplest mechanism

Outputs:
  Phase17C_policy_comparison.csv  — per policy × per frame rows + summary rows
  Phase17C_oracle_regret.csv      — per frame oracle regret (baseline R0)
  Phase17C_easy21_safety.csv      — per policy Easy21 deltas
  Phase17C_selection_churn.csv    — per policy churn counts
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase17c_common import (primary_score, tie_break_value, compute_penalty,
                             select_best, DENOM, FULL_WEIGHTS, SAM_W, compute_sam_score)

WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
OUT = WORKSPACE / "阶段十七C_候选排序修复开发"
BENCH = OUT / "Phase17C_ranking_benchmark.csv"
FAIL_FRAMES = {("DouBanLv2", "0039"), ("DouBanLv2", "0040"), ("DouBanLv2", "0041")}
POLICIES = ["R0", "R1", "R2", "R3"]
RESCUE_THRESHOLD = 0.95
MATERIAL_DELTA = -0.005


def load_gt_rows() -> list[dict]:
    with BENCH.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ["area_ratio", "q_area", "q_comp", "q_edge", "q_temp", "q_contrast",
                  "q_leak", "q_side", "sam_score", "total_score", "candidate_F1",
                  "candidate_IoU", "candidate_Prec", "candidate_Rec"]:
            r[k] = float(r[k])
        r["has_gt"] = int(r["has_gt"])
        r["baseline_selected"] = int(r["baseline_selected"])
        r["oracle_candidate"] = int(r["oracle_candidate"])
    return [r for r in rows if r["has_gt"]]


def main() -> None:
    rows = load_gt_rows()
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[(r["source"], r["sample"], r["frame"])].append(r)

    frame_keys = sorted(by_frame.keys())
    policy_rows = []          # per policy × per frame
    summary_rows = []         # per policy summary
    regret_rows = []          # oracle regret per frame (baseline)
    easy_rows_all = []        # per policy Easy21 deltas
    churn_rows = []           # per policy churn

    for policy in POLICIES:
        dev_rescued = 0
        dev_sel_f1 = []
        dev_regrets = []
        dev_material_reg = 0
        dev_top1 = 0
        dev_changed = 0
        easy_changed = 0
        easy_improved = 0
        easy_neutral = 0
        easy_regressed = 0
        easy_material_reg = 0
        dev_frames = [fk for fk in frame_keys if fk[0] == "DEV10"]
        easy_frames = [fk for fk in frame_keys if fk[0] == "Easy21"]

        for fk in frame_keys:
            cands = by_frame[fk]
            sel = select_best(cands, policy)
            baseline_sel = next((c for c in cands if c["baseline_selected"]), cands[0])
            oracle = next((c for c in cands if c["oracle_candidate"]), cands[0])
            sel_f1 = sel["candidate_F1"]
            bsel_f1 = baseline_sel["candidate_F1"]
            regret = oracle["candidate_F1"] - sel_f1
            changed = sel["candidate_id"] != baseline_sel["candidate_id"]

            policy_rows.append({
                "policy": policy,
                "source": fk[0],
                "sample": fk[1],
                "frame": fk[2],
                "baseline_selected": baseline_sel["candidate_id"],
                "new_selected": sel["candidate_id"],
                "changed": int(changed),
                "baseline_F1": round(bsel_f1, 4),
                "new_F1": round(sel_f1, 4),
                "delta_F1": round(sel_f1 - bsel_f1, 6),
                "oracle_F1": round(oracle["candidate_F1"], 4),
                "regret": round(regret, 6),
            })

            if fk[0] == "DEV10":
                dev_sel_f1.append(sel_f1)
                dev_regrets.append(regret)
                if changed:
                    dev_changed += 1
                if (fk[1], fk[2]) in FAIL_FRAMES:
                    if sel_f1 >= RESCUE_THRESHOLD:
                        dev_rescued += 1
                else:
                    if sel_f1 - bsel_f1 < MATERIAL_DELTA:
                        dev_material_reg += 1
                if sel["candidate_id"] == oracle["candidate_id"]:
                    dev_top1 += 1
            else:
                if changed:
                    easy_changed += 1
                    if sel_f1 > bsel_f1 + 1e-6:
                        easy_improved += 1
                    elif abs(sel_f1 - bsel_f1) <= 1e-6:
                        easy_neutral += 1
                    else:
                        easy_regressed += 1
                        if sel_f1 - bsel_f1 < MATERIAL_DELTA:
                            easy_material_reg += 1
                else:
                    easy_neutral += 1

        n_dev = len(dev_frames)
        n_easy = len(easy_frames)
        churn_rate = 100 * dev_changed / n_dev if n_dev else 0.0
        summary_rows.append({
            "policy": policy,
            "DEV10_mean_F1": round(float(np.mean(dev_sel_f1)), 4),
            "DEV10_mean_regret": round(float(np.mean(dev_regrets)), 6),
            "DEV10_top1_accuracy": f"{dev_top1}/{n_dev}",
            "DEV10_changed": f"{dev_changed}/{n_dev}",
            "DEV10_churn_rate_pct": round(churn_rate, 1),
            "failures_rescued": f"{dev_rescued}/3",
            "DEV10_material_regressions": dev_material_reg,
            "Easy21_changed": f"{easy_changed}/{n_easy}",
            "Easy21_churn_rate_pct": round(100 * easy_changed / n_easy, 1) if n_easy else 0.0,
            "Easy21_improved": easy_improved,
            "Easy21_neutral": easy_neutral,
            "Easy21_regressed": easy_regressed,
            "Easy21_material_regressions": easy_material_reg,
        })
        churn_rows.append({
            "policy": policy,
            "dev_changed": dev_changed,
            "easy_changed": easy_changed,
            "dev_churn_rate_pct": round(churn_rate, 1),
            "easy_churn_rate_pct": round(100 * easy_changed / n_easy, 1) if n_easy else 0.0,
        })

        # Easy21 per-frame delta detail
        for fk in easy_frames:
            cands = by_frame[fk]
            sel = select_best(cands, policy)
            baseline_sel = next((c for c in cands if c["baseline_selected"]), cands[0])
            sel_f1 = sel["candidate_F1"]
            bsel_f1 = baseline_sel["candidate_F1"]
            easy_rows_all.append({
                "policy": policy,
                "sample": fk[1],
                "frame": fk[2],
                "baseline_selected": baseline_sel["candidate_id"],
                "new_selected": sel["candidate_id"],
                "changed": int(sel["candidate_id"] != baseline_sel["candidate_id"]),
                "baseline_F1": round(bsel_f1, 4),
                "new_F1": round(sel_f1, 4),
                "delta_F1": round(sel_f1 - bsel_f1, 6),
            })

    # Oracle regret per frame (from R0 = baseline)
    regret_rows = [r for r in policy_rows if r["policy"] == "R0"]

    # Print policy table
    print("=== Step 8: Policy comparison ===")
    print(f"{'policy':<7} {'DEVmeanF1':>10} {'meanRegret':>10} {'top1':>6} {'rescue':>7} {'DEVreg':>7} "
          f"{'E21chg':>7} {'E21reg':>6} {'E21churn':>8}")
    for s in summary_rows:
        print(f"{s['policy']:<7} {s['DEV10_mean_F1']:>10} {s['DEV10_mean_regret']:>10} "
              f"{s['DEV10_top1_accuracy']:>6} {s['failures_rescued']:>7} "
              f"{s['DEV10_material_regressions']:>7} {s['Easy21_changed']:>7} "
              f"{s['Easy21_material_regressions']:>6} {s['Easy21_churn_rate_pct']:>8}")

    print("\n--- Per-frame DEV10 under each policy (new_F1 vs baseline) ---")
    dev_pol = [r for r in policy_rows if r["source"] == "DEV10"]
    by_frame_dev = defaultdict(list)
    for r in dev_pol:
        by_frame_dev[(r["sample"], r["frame"])].append(r)
    for fk in sorted(by_frame_dev):
        row_map = {r["policy"]: r for r in by_frame_dev[fk]}
        b = row_map["R0"]
        r1 = row_map["R1"]; r2 = row_map["R2"]; r3 = row_map["R3"]
        print(f"  {fk[0]}_{fk[1]}: R0 F1={b['new_F1']:.4f}({b['new_selected']}) | "
              f"R1 F1={r1['new_F1']:.4f}({r1['new_selected']}) | "
              f"R2 F1={r2['new_F1']:.4f}({r2['new_selected']}) | "
              f"R3 F1={r3['new_F1']:.4f}({r3['new_selected']})")

    # Decision (decision order §14)
    print("\n--- Decision (decision order §14) ---")
    def rescue_ok(s):
        return s["failures_rescued"] == "3/3"
    def dev_ok(s):
        return s["DEV10_material_regressions"] == 0
    def easy_ok(s):
        return s["Easy21_material_regressions"] == 0
    def mean_regret(s):
        return float(s["DEV10_mean_regret"])
    def churn(s):
        return float(s["Easy21_churn_rate_pct"]) + float(s["DEV10_churn_rate_pct"])

    eligible = [s for s in summary_rows if s["policy"] != "R0"
                and rescue_ok(s) and dev_ok(s) and easy_ok(s)]
    if eligible:
        winner = min(eligible, key=lambda s: (mean_regret(s), churn(s)))
        # Simplest mechanism: R1 (pure removal) before R2 (tie-break) before R3 (product)
        simple_order = {"R1": 0, "R2": 1, "R3": 2}
        eligible_sorted = sorted(eligible, key=lambda s: (mean_regret(s), churn(s), simple_order[s["policy"]]))
        winner = eligible_sorted[0]
        print(f"  Candidates meeting gates 1-3: {[s['policy'] for s in eligible]}")
        print(f"  WINNER: {winner['policy']} (mean_regret={winner['DEV10_mean_regret']}, "
              f"DEVchurn={winner['DEV10_churn_rate_pct']}%, E21churn={winner['Easy21_churn_rate_pct']}%)")
    else:
        print("  NO WINNER — all non-baseline policies fail a gate → conclude "
              "'q_contrast is not the only ranking deficiency' → Phase 17D redesign.")
        winner = None

    # Write outputs
    with (OUT / "Phase17C_policy_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(policy_rows[0].keys()))
        w.writeheader()
        w.writerows(policy_rows)
    with (OUT / "Phase17C_oracle_regret.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(regret_rows[0].keys()))
        w.writeheader()
        w.writerows(regret_rows)
    with (OUT / "Phase17C_easy21_safety.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(easy_rows_all[0].keys()))
        w.writeheader()
        w.writerows(easy_rows_all)
    with (OUT / "Phase17C_selection_churn.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(churn_rows[0].keys()))
        w.writeheader()
        w.writerows(churn_rows)

    # Persist decision into manifest log file
    winner_name = winner["policy"] if winner else "NONE"
    with (OUT / "Phase17C_winner.txt").open("w", encoding="utf-8") as f:
        f.write(f"winner={winner_name}\n")
        f.write("decision_order=rescue3>DEVreg>Easy21reg>lowest_regret>lowest_churn>simplest\n")
    print(f"  Written: Phase17C_policy_comparison.csv, Phase17C_oracle_regret.csv, "
          f"Phase17C_easy21_safety.csv, Phase17C_selection_churn.csv, Phase17C_winner.txt")
    print("\n=== Step 8 PASS ===")


if __name__ == "__main__":
    main()
