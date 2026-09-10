#!/usr/bin/env python3
"""
Phase 17C — Shared counterfactual reranking logic for Steps 4-8.

Frozen weight system:
  full weights: area=1, comp=1, edge=1, temp=1, contrast=1, sam=0.5
  denom: 5.5 (frozen, even when contrast removed)
  penalty: per-candidate property (vertical coverage = 0.25 if < 0.30; else 0)
"""
from __future__ import annotations
from typing import Sequence

import numpy as np

DENOM = 5.5
SAM_W = 0.5
EPSILON = 1e-12  # tie-break tolerance for R2 (exact primary equality)

FULL_WEIGHTS = {
    "area": 1.0,
    "comp": 1.0,
    "edge": 1.0,
    "temp": 1.0,
    "contrast": 1.0,
}

# Policy definitions: weight overrides for the five scored components
POLICIES: dict[str, dict[str, float]] = {
    "R0": {},  # baseline — all weights = 1.0
    "R1": {"contrast": 0.0},   # contrast removed from primary
    "R2": {"contrast": 0.0},   # same primary as R1; tie-break handled separately
    "R3": {"contrast": -1.0},  # sentinel — handled specially in primary_score
}

POLICY_TIE_BREAK: dict[str, str] = {
    "R0": "none",
    "R1": "none",
    "R2": "contrast",      # R2 = R1 primary + q_contrast tie-break
    "R3": "none",
}


def q_sam(sam: float) -> float:
    """Sigmoid transform of raw SAM score."""
    return 1.0 / (1.0 + np.exp(-6.0 * (sam - 0.5)))


def compute_sam_score(row: dict) -> float:
    """SAM3分数 (raw float) from benchmark row → q_sam (scored component)."""
    return q_sam(row["sam_score"])


def compute_penalty(row: dict) -> float:
    """Per-candidate penalty from the benchmark CSV's total_score.

    penalty = (full_base_total) − total_score
    Since we proved reconstruction is exact in Step 0, this gives penalty ∈ {0, 0.25, 0.40}.
    We recompute from the row to avoid storing redundant columns.
    """
    S = sum(FULL_WEIGHTS[k] * row[f"q_{k}"] for k in FULL_WEIGHTS)
    S += SAM_W * compute_sam_score(row)
    base = S / DENOM
    return base - row["total_score"]


def primary_score(row: dict, policy: str, penalty: float) -> float:
    """Compute counterfactual primary score for a candidate under a policy.

    R3 uses contrast_effective = q_contrast × q_area, replacing q_contrast in S.
    R0/R1/R2 use weight overrides from POLICIES dict (contrast=0.0 for R1/R2).
    """
    q_s = compute_sam_score(row)
    if policy == "R3":
        # S' = Σ_{k≠contrast} w_k·q_k + 1.0·(q_contrast × q_area) + 0.5·q_sam
        S = 0.0
        for k in FULL_WEIGHTS:
            if k == "contrast":
                S += 1.0 * (row["q_contrast"] * row["q_area"])
            else:
                S += FULL_WEIGHTS[k] * row[f"q_{k}"]
        S += SAM_W * q_s
        return S / DENOM - penalty
    else:
        S = sum(FULL_WEIGHTS[k] * row[f"q_{k}"] for k in FULL_WEIGHTS)
        S += SAM_W * q_s
        if "contrast" in POLICIES[policy]:
            S -= FULL_WEIGHTS["contrast"] * row["q_contrast"]  # remove contrast contribution
        return S / DENOM - penalty


def tie_break_value(row: dict, policy: str) -> float:
    """Tie-break value for R2 (q_contrast); 0.0 for other policies."""
    if POLICY_TIE_BREAK[policy] == "contrast":
        return row["q_contrast"]
    return 0.0


def select_best(rows: Sequence[dict], policy: str) -> dict:
    """Return the row with highest (primary, tie_break) under a given policy.

    Selection: max lexicographic (primary_score, tie_break_value).
    Ties broken by first occurrence (stable), matching pipeline's argmax behavior.
    """
    penalties = [compute_penalty(r) for r in rows]
    best = None
    best_key = None
    for row, pen in zip(rows, penalties):
        ps = primary_score(row, policy, pen)
        tb = tie_break_value(row, policy)
        key = (ps, tb)
        if best_key is None or key > best_key:
            best_key = key
            best = row
    return best
