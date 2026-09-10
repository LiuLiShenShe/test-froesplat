#!/usr/bin/env python3
"""
Phase 17C — Step 21: Production replay verification.

Imports the SAME ranking_policy.py module used by production S20, replays the
frozen DEV10 + Easy21 candidates through select_best_policy(), and compares the
selected instance / primary_score / total_score / F1 against the offline benchmark.

Must match exactly (same function, same data). Any mismatch → FAIL.
Output: Phase17C_production_replay.csv (per-frame: baseline vs winning-policy).
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
S20_SCRIPT = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本")
OUT = WORKSPACE / "阶段十七C_候选排序修复开发"
BENCH = OUT / "Phase17C_ranking_benchmark.csv"
WINNER = "r1"

sys.path.insert(0, str(S20_SCRIPT))
from ranking_policy import select_best_policy, primary_and_tiebreak  # noqa: E402

DENOM = 5.5
SAM_W = 0.5
FULL_WEIGHTS = {"area": 1.0, "comp": 1.0, "edge": 1.0, "temp": 1.0, "contrast": 1.0}


def q_sam(sam):
    return 1.0 / (1.0 + np.exp(-6.0 * (sam - 0.5)))


def full_base(r):
    S = sum(FULL_WEIGHTS[k] * r[f"q_{k}"] for k in FULL_WEIGHTS)
    S += SAM_W * q_sam(r["sam_score"])
    return S / DENOM


def penalty_of(r):
    return full_base(r) - r["total_score"]


def main() -> None:
    with BENCH.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("area_ratio", "q_area", "q_comp", "q_edge", "q_temp", "q_contrast",
                   "q_leak", "q_side", "sam_score", "total_score", "candidate_F1",
                   "candidate_IoU", "candidate_Prec", "candidate_Rec"):
            r[k] = float(r[k])
        r["has_gt"] = int(r["has_gt"])
        r["baseline_selected"] = int(r["baseline_selected"])
        r["oracle_candidate"] = int(r["oracle_candidate"])

    gt_rows = [r for r in rows if r["has_gt"]]
    by_frame = defaultdict(list)
    for r in gt_rows:
        by_frame[(r["source"], r["sample"], r["frame"])].append(r)

    replay_rows = []
    n_match = 0
    n_mismatch = 0
    for fk in sorted(by_frame):
        cands = by_frame[fk]
        # Build lightweight records exposing the fields select_best_policy needs
        # (SimpleNamespace so getattr-based key works — same contract as ScoreRecord)
        records = []
        for r in cands:
            pen = penalty_of(r)
            p, t = primary_and_tiebreak(
                base_total=full_base(r), q_contrast=r["q_contrast"], q_area=r["q_area"],
                w_contrast=1.0, denom=DENOM, semantic_total=0.0, penalty=pen,
                policy=WINNER,
            )
            records.append(SimpleNamespace(
                row=r,
                primary_score=p,
                tie_break_score=t,
                prompt_id="P6",
                instance_id=r["instance_id"],
                total_score=r["total_score"],
            ))
        best = select_best_policy(records, WINNER)
        best_r = best.row
        baseline = next(r for r in cands if r["baseline_selected"])
        oracle = next(r for r in cands if r["oracle_candidate"])

        # Offline expected winner: benchmark + step8 (R1 == oracle on all GT frames)
        offline_best = oracle["candidate_id"]
        online_best = best_r["candidate_id"]
        matches = (online_best == offline_best)
        n_match += int(matches)
        n_mismatch += int(not matches)
        replay_rows.append({
            "source": fk[0],
            "sample": fk[1],
            "frame": fk[2],
            "policy": WINNER,
            "baseline_selected": baseline["candidate_id"],
            "replay_selected": online_best,
            "offline_expected": offline_best,
            "primary_score": round(best.primary_score, 8),
            "total_score": round(best_r["total_score"], 8),
            "candidate_F1": round(best_r["candidate_F1"], 4),
            "baseline_F1": round(baseline["candidate_F1"], 4),
            "oracle_F1": round(oracle["candidate_F1"], 4),
            "match": int(matches),
        })

    with (OUT / "Phase17C_production_replay.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(replay_rows[0].keys()))
        w.writeheader()
        w.writerows(replay_rows)

    print(f"=== Step 21: Production replay ({WINNER}) ===")
    print(f"GT frames: {len(replay_rows)}")
    print(f"Matches offline oracle expectation: {n_match}/{len(replay_rows)}")
    print(f"Mismatches: {n_mismatch}")
    for rr in replay_rows:
        mark = "✓" if rr["match"] else "✗"
        print(f"  {rr['sample']}_{rr['frame']}: replay={rr['replay_selected']} "
              f"expected={rr['offline_expected']} F1={rr['candidate_F1']:.4f} {mark}")
    assert n_mismatch == 0, f"{n_mismatch} replay mismatches — debug before proceeding"
    print(f"Written: Phase17C_production_replay.csv")
    print("\n=== Step 21 PASS (replay exact) ===")


if __name__ == "__main__":
    main()
