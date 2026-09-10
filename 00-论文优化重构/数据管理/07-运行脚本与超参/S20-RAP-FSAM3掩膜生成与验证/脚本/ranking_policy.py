#!/usr/bin/env python3
"""
Phase 17C — Candidate ranking policy (pure functions, no side effects).

Single source of truth shared by BOTH the S20 production pipeline and the Phase17C
offline benchmark (production replay exactness by construction).

Policies (§8 decision — winner = R1):
  base  primary = total (= base_total + semantic_total − penalty), tie = 0
        → bitwise identical to historical production selection.
  r1    primary = (base_total·denom − w_contrast·q_contrast)/denom + semantic_total − penalty
        (q_contrast removed from primary ranking), tie = 0.
  r2    same primary as r1; tie_break = q_contrast (lexicographic: only breaks EXACT
        primary ties — cannot reverse unequal primaries).
  r3    primary = (base_total·denom − w_contrast·q_contrast + w_contrast·q_contrast·q_area)/denom
        + semantic_total − penalty (contrast modulated by area plausibility), tie = 0.

Denominator (denom) is the frozen full-weights sum (5.5) — unchanged under all policies.
"""
from __future__ import annotations

# Policy names accepted by --ranking_policy
POLICY_NAMES = ("base", "r1", "r2", "r3")


def primary_and_tiebreak(
    base_total: float,
    q_contrast: float,
    q_area: float,
    w_contrast: float,
    denom: float,
    semantic_total: float,
    penalty: float,
    policy: str,
) -> tuple[float, float]:
    """Return (primary_score, tie_break_score) for one candidate under `policy`.

    All math mirrors score_candidate's numerator: numerator = Σ w_k·q_k + w_sam·q_sam,
    base_total = numerator / denom, total = base_total + semantic_total − penalty.
    Policy only alters the contrast term in the numerator (structural, no weight search).
    """
    if policy == "base":
        return base_total + semantic_total - penalty, 0.0
    if policy in ("r1", "r2"):
        primary = (base_total * denom - w_contrast * q_contrast) / denom \
            + semantic_total - penalty
        tie = q_contrast if policy == "r2" else 0.0
        return primary, tie
    if policy == "r3":
        primary = (base_total * denom - w_contrast * q_contrast
                   + w_contrast * q_contrast * q_area) / denom \
            + semantic_total - penalty
        return primary, 0.0
    raise ValueError(f"Unknown ranking policy: {policy!r}")


def select_best_policy(score_records, policy: str):
    """Return the best record under `policy` from a sequence of score records.

    Records must expose .primary_score and .tie_break_score (set by score_candidate).
    Lexicographic max on (primary_score, tie_break_score); ties broken by first
    occurrence — matches the historical per_instance argmax-on-total behavior for
    the base policy (where primary == total and tie == 0).
    """
    return max(
        score_records,
        key=lambda r: (getattr(r, "primary_score", 0.0), getattr(r, "tie_break_score", 0.0)),
    )
