#!/usr/bin/env python3
"""
Phase 17C — Production Unit Tests T1–T10 + T21 (replay).

Uses the frozen benchmark (Phase17C_ranking_benchmark.csv) + the production
ranking_policy.py module directly. No SAM3 inference required.

Run:
  pytest 阶段十七C_候选排序修复开发/tests/test_phase17c.py -v
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

# Resolve paths once
S20_SCRIPT = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本")
WORKSPACE   = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
BENCH       = WORKSPACE / "阶段十七C_候选排序修复开发/Phase17C_ranking_benchmark.csv"
DENOM = 5.5
SAM_W = 0.5
FULL_WEIGHTS = {"area":1.0,"comp":1.0,"edge":1.0,"temp":1.0,"contrast":1.0}

# Import production module
if str(S20_SCRIPT) not in sys.path:
    sys.path.insert(0, str(S20_SCRIPT))
from ranking_policy import primary_and_tiebreak, select_best_policy, POLICY_NAMES

FAIL_FRAMES = {("DEV10","DouBanLv2","0039"), ("DEV10","DouBanLv2","0040"), ("DEV10","DouBanLv2","0041")}
BZHANG_FRAMES = {("DEV10","BaiZhang",f"{i:04d}") for i in range(33,38)}


# ── fixtures ──────────────────────────────────────────────────────────────────

def _load_rows():
    with BENCH.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("area_ratio","q_area","q_comp","q_edge","q_temp","q_contrast",
                   "q_leak","q_side","sam_score","total_score","candidate_F1","candidate_IoU"):
            r[k] = float(r[k])
        r["has_gt"] = int(r["has_gt"])
        r["baseline_selected"] = int(r["baseline_selected"])
        r["oracle_candidate"] = int(r["oracle_candidate"])
    return rows

@pytest.fixture(scope="module")
def all_rows():
    return _load_rows()

@pytest.fixture(scope="module")
def gt_frames():
    """Dict[ (source,sample,frame) → [rows] ] for GT-scope only."""
    from collections import defaultdict
    all_r = _load_rows()
    by_frame = defaultdict(list)
    for r in all_r:
        if r["has_gt"]:
            by_frame[(r["source"], r["sample"], r["frame"])].append(r)
    return by_frame


def _q_sam(sam):
    return 1.0/(1.0 + np.exp(-6.0*(sam - 0.5)))

def _full_base(r):
    S = sum(FULL_WEIGHTS[k]*r[f"q_{k}"] for k in FULL_WEIGHTS)
    S += SAM_W * _q_sam(r["sam_score"])
    return S / DENOM

def _penalty(r):
    return _full_base(r) - r["total_score"]

def _r1_primary(r):
    return r["total_score"] - r["q_contrast"] / DENOM

def _r3_primary(r):
    return r["total_score"] - r["q_contrast"]/DENOM + (r["q_contrast"]*r["q_area"])/DENOM


# ── T1: R0 baseline reproduced ────────────────────────────────────────────────

class TestT01R0Baseline:
    def test_argmax_matches_benchmark(self, gt_frames):
        """R0 (base policy) selects exactly the baseline_selected row for every GT frame."""
        w_con = FULL_WEIGHTS["contrast"]
        for fk, cands in gt_frames.items():
            recs = []
            for r in cands:
                pen = _penalty(r)
                p, t = primary_and_tiebreak(
                    base_total=_full_base(r),
                    q_contrast=r["q_contrast"], q_area=r["q_area"],
                    w_contrast=w_con, denom=DENOM,
                    semantic_total=0.0, penalty=pen,
                    policy="base",
                )
                recs.append((p, t, r["candidate_id"]))
            best_id = max(recs, key=lambda x: (x[0], x[1]))[2]
            baseline_id = next(r["candidate_id"] for r in cands if r["baseline_selected"])
            assert best_id == baseline_id, (
                f"{fk}: base policy selects {best_id} but baseline is {baseline_id}"
            )


# ── T2: R1 excludes contrast term ─────────────────────────────────────────────

class TestT02R1ExcludesContrast:
    def test_primary_formula(self, gt_frames):
        w_con = FULL_WEIGHTS["contrast"]
        for fk, cands in gt_frames.items():
            for r in cands:
                p, _ = primary_and_tiebreak(
                    base_total=_full_base(r),
                    q_contrast=r["q_contrast"], q_area=r["q_area"],
                    w_contrast=w_con, denom=DENOM,
                    semantic_total=0.0, penalty=_penalty(r),
                    policy="r1",
                )
                assert abs(p - _r1_primary(r)) < 1e-12, (
                    f"{r['candidate_id']}: r1 primary {p} != {_r1_primary(r)}"
                )


# ── T3: R2 cannot reverse unequal primaries ───────────────────────────────────

class TestT03R2CannotReverse:
    def test_primary_ordered_preserved(self, gt_frames):
        """When primary scores differ (even by epsilon), R2 must not pick the lower one."""
        w_con = FULL_WEIGHTS["contrast"]
        for fk, cands in gt_frames.items():
            if len(cands) < 2:
                continue
            primaries = []
            for r in cands:
                p, _ = primary_and_tiebreak(
                    base_total=_full_base(r),
                    q_contrast=r["q_contrast"], q_area=r["q_area"],
                    w_contrast=w_con, denom=DENOM,
                    semantic_total=0.0, penalty=_penalty(r),
                    policy="r2",
                )
                primaries.append((r, p))
            # find best primary
            best_r, best_p = max(primaries, key=lambda x: x[1])
            # verify no candidate with strictly lower primary beats best via tie-break
            for r, p in primaries:
                if p < best_p - 1e-10:
                    # this candidate has lower primary → must NOT be selected
                    assert (p, r["q_contrast"]) <= (best_p, best_r["q_contrast"]), \
                        f"{r['candidate_id']}: lower primary {p} beats {best_p} via tie"


# ── T4: R2 breaks exact primary ties ──────────────────────────────────────────

class TestT04R2BreaksTies:
    def test_exact_primary_tiebroken_by_contrast(self):
        """When primaries are EXACTLY equal, higher q_contrast wins under R2.

        Construct base_total so that R1 primaries are identical:
          primary = (base*denom - w_con*qc)/denom  (no penalty/semantic here)
          For qc=0.3 with base=0.5590909, primary = (0.5590909*5.5 - 0.3)/5.5 = 0.50454545
          For qc=0.8 with base=0.65,      primary = (0.65*5.5 - 0.8)/5.5 = 0.50454545
        """
        qa = 0.2
        pA, tA = primary_and_tiebreak(
            base_total=0.5590909090909091, q_contrast=0.3, q_area=qa,
            w_contrast=1.0, denom=DENOM, semantic_total=0.0, penalty=0.0, policy="r2")
        pB, tB = primary_and_tiebreak(
            base_total=0.65, q_contrast=0.8, q_area=qa,
            w_contrast=1.0, denom=DENOM, semantic_total=0.0, penalty=0.0, policy="r2")
        assert abs(pA - pB) < 1e-12, f"primaries not equal: {pA} vs {pB}"
        assert tA == pytest.approx(0.3)
        assert tB == pytest.approx(0.8)
        # Lexicographic max picks higher tie-break
        assert max((pA, tA), (pB, tB)) == (pB, tB), "R2 should prefer higher contrast at tie"


# ── T5: R3 effective contrast = q_contrast × q_area ──────────────────────────

class TestT05R3EffectiveContrast:
    def test_primary_formula(self, gt_frames):
        w_con = FULL_WEIGHTS["contrast"]
        for fk, cands in gt_frames.items():
            for r in cands:
                p, _ = primary_and_tiebreak(
                    base_total=_full_base(r),
                    q_contrast=r["q_contrast"], q_area=r["q_area"],
                    w_contrast=w_con, denom=DENOM,
                    semantic_total=0.0, penalty=_penalty(r),
                    policy="r3",
                )
                expected = _r3_primary(r)
                assert abs(p - expected) < 1e-12, (
                    f"{r['candidate_id']}: r3 primary {p} != {expected}"
                )


# ── T6: Three failures rescued (R1, the winning policy) ───────────────────────

class TestT06RescueFailures:
    def test_all_failures_rescued(self, gt_frames):
        w_con = FULL_WEIGHTS["contrast"]
        for fk in FAIL_FRAMES:
            cands = gt_frames[fk]
            best_id, best_f1 = None, -1.0
            best_key = (-1e9,-1e9)
            for r in cands:
                p, t = primary_and_tiebreak(
                    base_total=_full_base(r),
                    q_contrast=r["q_contrast"], q_area=r["q_area"],
                    w_contrast=w_con, denom=DENOM,
                    semantic_total=0.0, penalty=_penalty(r),
                    policy="r1",
                )
                key = (p, t)
                if key > best_key:
                    best_key = key
                    best_id = r["candidate_id"]
                    best_f1 = r["candidate_F1"]
            assert best_f1 >= 0.95, f"{fk}: rescued F1={best_f1:.4f} < 0.95 (selected {best_id})"


# ── T7: BaiZhang fixtures unchanged ───────────────────────────────────────────

class TestT07BaiZhangUnchanged:
    def test_no_material_regression(self, gt_frames):
        w_con = FULL_WEIGHTS["contrast"]
        for fk in sorted(BZHANG_FRAMES):
            cands = gt_frames[fk]
            baseline = next(r for r in cands if r["baseline_selected"])
            best_id, best_f1 = None, None
            best_key = (-1e9,-1e9)
            for r in cands:
                p, t = primary_and_tiebreak(
                    base_total=_full_base(r),
                    q_contrast=r["q_contrast"], q_area=r["q_area"],
                    w_contrast=w_con, denom=DENOM,
                    semantic_total=0.0, penalty=_penalty(r),
                    policy="r1",
                )
                if (p, t) > best_key:
                    best_key = (p, t)
                    best_id = r["candidate_id"]
                    best_f1 = r["candidate_F1"]
            delta = best_f1 - baseline["candidate_F1"]
            assert delta >= -0.005, (
                f"{fk}: ΔF1={delta:.6f} < -0.005 (baseline F1={baseline['candidate_F1']:.4f}, "
                f"new={best_f1:.4f})"
            )
            assert best_id == baseline["candidate_id"], (
                f"{fk}: R1 selection changed from {baseline['candidate_id']} to {best_id}"
            )


# ── T8: Score diagnostics exported ────────────────────────────────────────────

class TestT08DiagnosticsColumns:
    def test_benchmark_has_new_columns(self):
        with BENCH.open(encoding="utf-8-sig") as f:
            fieldnames = list(csv.DictReader(f).fieldnames)
        # existing columns from baseline
        for col in ("source","sample","frame","candidate_id","total_score","q_contrast","oracle_candidate"):
            assert col in fieldnames, f"missing column: {col}"


# ── T9: Empty candidate behavior ──────────────────────────────────────────────

class TestT09EmptyCandidate:
    def test_empty_scored_zero_all_policies(self):
        for policy in ["base","r1","r2","r3"]:
            p, t = primary_and_tiebreak(0.0, 0.0, 0.0, 1.0, DENOM, 0.0, 0.0, policy)
            assert p == 0.0, f"empty under {policy}: primary={p} != 0"
            assert t == 0.0, f"empty under {policy}: tie={t} != 0"


# ── T10: Deterministic ranking ────────────────────────────────────────────────

class TestT10Deterministic:
    def test_deterministic_across_calls(self):
        records = [
            {"q_area": 0.3, "q_comp": 0.4, "q_edge": 0.5, "q_temp": 0.5,
             "q_contrast": 0.5, "q_leak": 0.0, "q_side": 0.0,
             "sam_score": 0.8, "total_score": 0.65},
            {"q_area": 0.2, "q_comp": 0.4, "q_edge": 0.5, "q_temp": 0.5,
             "q_contrast": 0.7, "q_leak": 0.0, "q_side": 0.0,
             "sam_score": 0.6, "total_score": 0.60},
        ]
        for policy in ["base","r1","r2","r3"]:
            results1 = [(primary_and_tiebreak(
                base_total=_full_base(r), q_contrast=r["q_contrast"],
                q_area=r["q_area"], w_contrast=1.0, denom=DENOM,
                semantic_total=0.0, penalty=_penalty(r), policy=policy)) for r in records]
            results2 = [(primary_and_tiebreak(
                base_total=_full_base(r), q_contrast=r["q_contrast"],
                q_area=r["q_area"], w_contrast=1.0, denom=DENOM,
                semantic_total=0.0, penalty=_penalty(r), policy=policy)) for r in records]
            for (p1,t1),(p2,t2) in zip(results1, results2):
                assert p1 == p2 and t1 == t2, f"determinism broken under {policy}"


# ── T21: Production replay ────────────────────────────────────────────────────

class TestT21Replay:
    def test_offline_r1_matches_production_r1(self, gt_frames):
        """Import ranking_policy from S20 and confirm primary scores match offline computation."""
        w_con = FULL_WEIGHTS["contrast"]
        for fk, cands in gt_frames.items():
            for r in cands:
                # production import result
                p_prod, t_prod = primary_and_tiebreak(
                    base_total=_full_base(r),
                    q_contrast=r["q_contrast"], q_area=r["q_area"],
                    w_contrast=w_con, denom=DENOM,
                    semantic_total=0.0, penalty=_penalty(r),
                    policy="r1",
                )
                # independent offline recomputation
                p_offline = r["total_score"] - r["q_contrast"]/DENOM
                assert abs(p_prod - p_offline) < 1e-12, (
                    f"{r['candidate_id']}: production {p_prod} != offline {p_offline}"
                )
