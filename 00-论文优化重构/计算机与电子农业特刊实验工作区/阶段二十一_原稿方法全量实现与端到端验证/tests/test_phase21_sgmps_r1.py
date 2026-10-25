#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 21 — SG-MPS gated softmax + R1 branch unification unit tests.

These tests exercise the *pure* functions added in Phase 21 and do not require
SAM3, COLMAP or GPU. They verify:

  T1  `parse_weight_vector` rejects malformed descriptor weights.
  T2  `semantic_gate_scores` returns phi/psi descriptor values in [0,1] and
      computes a bounded gate confidence.
  T3  The gated confidence decreases when interference (psi) increases and
      increases when foreground reliability (phi) increases.
  T4  `apply_semantic_gate_softmax` produces a normalised distribution that
      sums to 1, is shift invariant, and sharpens as tau_s decreases.
  T5  Under `heuristic` mode the softmax application is a no-op (historical
      replay safety).
  T6  Under `gated_softmax` the gate adds a bounded bonus that never exceeds
      `semantic_gate_bonus`, and R1 primary ordering is still respected.
  T7  `select_mask` routes `score_select` through the ranking policy: with
      `ranking_policy=r1` a contrast-dominated candidate is not selected.
  T8  `select_mask` routes `weighted_fusion` through the ranking policy and
      produces weights that sum to 1.
  T9  `select_mask` under `ranking_policy=base` reproduces the historical
      `max(total_score)` selection for `score_select` (replay equivalence).
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PIPELINE = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参/"
                "S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py")


def load_pipeline_module():
    mod_name = "生成RAP-FSAM3掩码_phase21"
    spec = importlib.util.spec_from_file_location(mod_name, PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    script_dir = str(PIPELINE.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return load_pipeline_module()


def make_args(**overrides):
    base = dict(
        use_semantic_gate=False,
        semantic_gate_mode="heuristic",
        semantic_phi_weights="0.5,0.3,0.15,0.15,0.0,0.1",
        semantic_psi_weights="0.3,0.3,0.3,0.2",
        semantic_gate_bias=0.5,
        semantic_gate_temperature=0.25,
        semantic_gate_bonus=0.05,
        semantic_gate_use_contrast=False,
        use_prompt_ensemble=False,
        prompt_selection_mode="single",
        default_prompt_id="P6",
        candidate_mode="per_instance",
        ranking_policy="r1",
        area_min_ratio=0.01,
        area_max_ratio=0.80,
        area_target_ratio=0.0,
        component_min_area_ratio=0.0005,
        semantic_green_exg_threshold=12.0,
        semantic_pot_lower_start_ratio=0.58,
        target_box_weight=0.35,
        vertical_coverage_weight=0.18,
        center_prior_weight=0.10,
        pot_overlap_weight=0.30,
        side_distractor_weight=0.25,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def make_context(mod, shape=(40, 40)):
    target = np.zeros(shape, dtype=bool)
    target[8:30, 10:30] = True
    pot = np.zeros(shape, dtype=bool)
    pot[28:34, 12:28] = True
    side = np.zeros(shape, dtype=bool)
    side[:, :4] = True
    side[:, -4:] = True
    return mod.SemanticGateContext(
        target_mask=target, pot_mask=pot, side_mask=side,
        target_box=(10, 8, 30, 30), pot_box=(12, 28, 28, 34), side_boxes=[],
    )


def make_record(mod, **overrides):
    fields = dict(
        image_name="f.jpg", prompt_id="P6", prompt_text="potted", total_score=0.5,
        q_area=0.5, q_comp=0.5, q_edge=0.5, q_temp=0.5, q_contrast=0.5,
        q_leak=0.0, q_side=0.0, area_ratio=0.1, component_count=1,
        boundary_density=0.1, temporal_iou=0.5, contrast=0.5,
        bottom_leak_fraction=0.0, side_leak_fraction=0.0, sam_scores="0.5",
        instance_id=0, empty_flag=False, ranking_policy="r1",
        primary_score=0.5, tie_break_score=0.0, contrast_effective=0.0,
        gate_confidence=0.5, gate_softmax=0.0,
    )
    fields.update(overrides)
    return mod.ScoreRecord(**fields)


# ── T1 ───────────────────────────────────────────────────────────────────────
def test_t1_weight_vector_validation(mod):
    args = make_args()
    assert mod.parse_weight_vector("1,2,3,4,5,6", 6, "x") == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    with pytest.raises(ValueError):
        mod.parse_weight_vector("1,2", 6, "x")


# ── T2 / T3 ──────────────────────────────────────────────────────────────────
def test_t2_descriptor_ranges_and_confidence(mod):
    args = make_args()
    ctx = make_context(mod)
    image = None  # pure geometry path
    good = np.zeros((40, 40), dtype=bool)
    good[10:28, 12:28] = True
    result = mod.semantic_gate_scores(good, ctx, args, image=image, temporal_iou=0.5)
    for key in ("q_plant", "q_area", "q_comp", "q_edge", "q_contrast", "q_temp",
                "q_table", "q_bg", "q_neighbour", "q_outlier"):
        assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]}"
    assert 0.0 <= result["gate_confidence"] <= 1.0
    # phi/psi descriptor fields exist
    assert set(result) >= {"q_plant", "q_outlier"}


def test_t3_confidence_responds_to_interference(mod):
    args = make_args()
    ctx = make_context(mod)
    clean = np.zeros((40, 40), dtype=bool)
    clean[10:26, 12:28] = True
    dirty = clean.copy()
    dirty[:, :3] = True      # side-band interference
    dirty[36:, 12:28] = True  # lower-scene interference
    r_clean = mod.semantic_gate_scores(clean, ctx, args, image=None, temporal_iou=0.5)
    r_dirty = mod.semantic_gate_scores(dirty, ctx, args, image=None, temporal_iou=0.5)
    assert r_dirty["q_neighbour"] > r_clean["q_neighbour"]
    assert r_dirty["q_outlier"] >= r_clean["q_outlier"]
    assert r_dirty["gate_confidence"] < r_clean["gate_confidence"]


# ── T4 ───────────────────────────────────────────────────────────────────────
def test_t4_softmax_normalises_and_sharpens(mod):
    args = make_args(use_semantic_gate=True, semantic_gate_mode="gated_softmax")
    recs = [make_record(mod, prompt_id=f"P{i}", gate_confidence=c)
            for i, c in enumerate([0.2, 0.6, 0.9])]
    mod.apply_semantic_gate_softmax(recs, args)
    total = sum(r.gate_softmax for r in recs)
    assert total == pytest.approx(1.0, abs=1e-9)
    assert recs[2].gate_softmax > recs[1].gate_softmax > recs[0].gate_softmax
    # shift invariance of confidences
    recs2 = [make_record(mod, prompt_id=f"P{i}", gate_confidence=c + 1.0)
             for i, c in enumerate([0.2, 0.6, 0.9])]
    mod.apply_semantic_gate_softmax(recs2, args)
    for a, b in zip(recs, recs2):
        assert a.gate_softmax == pytest.approx(b.gate_softmax, abs=1e-9)
    # lower temperature → sharper
    args_hard = make_args(use_semantic_gate=True, semantic_gate_mode="gated_softmax",
                          semantic_gate_temperature=0.05)
    recs3 = [make_record(mod, prompt_id=f"P{i}", gate_confidence=c)
             for i, c in enumerate([0.2, 0.6, 0.9])]
    mod.apply_semantic_gate_softmax(recs3, args_hard)
    assert recs3[2].gate_softmax > recs[2].gate_softmax


# ── T5 ───────────────────────────────────────────────────────────────────────
def test_t5_heuristic_mode_is_noop(mod):
    args = make_args(use_semantic_gate=True, semantic_gate_mode="heuristic")
    recs = [make_record(mod, prompt_id=f"P{i}", gate_confidence=c, primary_score=0.5)
            for i, c in enumerate([0.2, 0.6, 0.9])]
    mod.apply_semantic_gate_softmax(recs, args)
    for r in recs:
        assert r.gate_softmax == 0.0
        assert r.primary_score == pytest.approx(0.5)


# ── T6 ───────────────────────────────────────────────────────────────────────
def test_t6_prompt_selected_by_gate_confidence(mod):
    """k* = argmax_k a_i^(k) == argmax_k c_i^(k); the ranking policy is untouched."""
    args = make_args(use_semantic_gate=True, semantic_gate_mode="gated_softmax")
    recs = [make_record(mod, prompt_id=f"P{i}", gate_confidence=c, primary_score=p)
            for i, (c, p) in enumerate([(0.9, 0.40), (0.2, 0.60)])]
    before = [r.primary_score for r in recs]
    mod.apply_semantic_gate_softmax(recs, args)
    for r, b in zip(recs, before):
        assert r.primary_score == pytest.approx(b), "the gate must not perturb the R1 score"
    assert mod.semantic_gate_prompt(recs, args) == "P0", "semantic evidence picks the prompt"
    # within the gated prompt the ranking policy still decides the instance
    assert mod.semantic_gate_prompt(recs, make_args(use_semantic_gate=True,
                                                    semantic_gate_mode="heuristic")) is None


def test_t6b_gate_prompt_overrides_geometry_winner(mod):
    """SG-MPS must be able to reject a prompt that only wins on geometry score."""
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_a[2:5, 2:5] = True
    mask_b = np.zeros((10, 10), dtype=bool)
    mask_b[6:9, 6:9] = True
    cands = [
        mod.Candidate(prompt_id="P1", prompt_text="green plant", instance_id=0, mask=mask_a,
                      box=None, sam_score=0.9, mask_threshold=0.5, scores=[0.9],
                      raw_detection_count=1, source_stage="pass1", prompt_mode="per_instance"),
        mod.Candidate(prompt_id="P2", prompt_text="plant w/o pot", instance_id=0, mask=mask_b,
                      box=None, sam_score=0.2, mask_threshold=0.5, scores=[0.2],
                      raw_detection_count=1, source_stage="pass1", prompt_mode="per_instance"),
    ]
    recs = [
        make_record(mod, prompt_id="P1", instance_id=0, total_score=0.80,
                    primary_score=0.80, gate_confidence=0.10),
        make_record(mod, prompt_id="P2", instance_id=0, total_score=0.40,
                    primary_score=0.40, gate_confidence=0.85),
    ]
    args = make_args(use_prompt_ensemble=True, prompt_selection_mode="score_select",
                     ranking_policy="r1", default_prompt_id="P2",
                     use_semantic_gate=True, semantic_gate_mode="gated_softmax")
    off = make_args(use_prompt_ensemble=True, prompt_selection_mode="score_select",
                    ranking_policy="r1", default_prompt_id="P2")
    mask_off, pid_off, _, _ = mod.select_mask(
        Path("f.jpg"), cands, recs, {"P1": "a", "P2": "b"}, off)
    assert pid_off == "P1", "without the gate the geometry score wins"
    mask_on, pid_on, _, _ = mod.select_mask(
        Path("f.jpg"), cands, recs, {"P1": "a", "P2": "b"}, args)
    assert pid_on == "P2", "SG-MPS selects the semantically compatible prompt"
    assert np.array_equal(mask_on, mask_b)


# ── T7 / T9 ──────────────────────────────────────────────────────────────────
def _score_select_setup(mod, use_contrast_high=False):
    """Two instances where the contrast-rich one has a higher total_score."""
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_a[2:5, 2:5] = True
    mask_b = np.zeros((10, 10), dtype=bool)
    mask_b[6:9, 6:9] = True
    cands = [
        mod.Candidate(prompt_id="P6", prompt_text="potted", instance_id=1, mask=mask_a,
                      box=(2, 2, 5, 5), sam_score=0.9, mask_threshold=0.5, scores=[0.9],
                      raw_detection_count=2, source_stage="pass1", prompt_mode="per_instance"),
        mod.Candidate(prompt_id="P6", prompt_text="potted", instance_id=2, mask=mask_b,
                      box=(6, 6, 9, 9), sam_score=0.7, mask_threshold=0.5, scores=[0.7],
                      raw_detection_count=2, source_stage="pass1", prompt_mode="per_instance"),
    ]
    # instance 1 wins on base_total, instance 2 only wins because of q_contrast
    recs = [
        make_record(mod, prompt_id="P6", instance_id=1, total_score=0.60, primary_score=0.60,
                    q_contrast=0.10, ranking_policy="r1"),
        make_record(mod, prompt_id="P6", instance_id=2, total_score=0.70, primary_score=0.30,
                    q_contrast=0.95, ranking_policy="r1"),
    ]
    return cands, recs


def test_t7_r1_reaches_score_select(mod):
    cands, recs = _score_select_setup(mod)
    args = make_args(use_prompt_ensemble=True, prompt_selection_mode="score_select",
                     ranking_policy="r1", default_prompt_id="P6")
    mask, pid, score, _ = mod.select_mask(
        Path("f.jpg"), cands, recs, {"P6": "potted"}, args)
    assert np.array_equal(mask, cands[0].mask), "R1 must select the contrast-free candidate"
    assert score == pytest.approx(0.60)


def test_t9_base_replays_historical_score_select(mod):
    cands, recs = _score_select_setup(mod)
    recs = [
        make_record(mod, prompt_id="P6", instance_id=1, total_score=0.60, primary_score=0.60,
                    q_contrast=0.10, ranking_policy="base"),
        make_record(mod, prompt_id="P6", instance_id=2, total_score=0.70, primary_score=0.70,
                    q_contrast=0.95, ranking_policy="base"),
    ]
    args = make_args(use_prompt_ensemble=True, prompt_selection_mode="score_select",
                     ranking_policy="base", default_prompt_id="P6")
    mask, _, score, _ = mod.select_mask(
        Path("f.jpg"), cands, recs, {"P6": "potted"}, args)
    assert np.array_equal(mask, cands[1].mask), "base must keep historical max(total_score)"
    assert score == pytest.approx(0.70)


# ── T8 ───────────────────────────────────────────────────────────────────────
def test_t8_weighted_fusion_uses_primary(mod):
    m1 = np.zeros((6, 6), dtype=bool)
    m1[0:3, 0:3] = True
    m2 = np.zeros((6, 6), dtype=bool)
    m2[3:6, 3:6] = True
    cands = [
        mod.Candidate(prompt_id="P6", prompt_text="a", instance_id=0, mask=m1, box=None,
                      sam_score=0.9, mask_threshold=0.5, scores=[0.9], raw_detection_count=1,
                      source_stage="pass1", prompt_mode="per_instance"),
        mod.Candidate(prompt_id="P2", prompt_text="b", instance_id=0, mask=m2, box=None,
                      sam_score=0.1, mask_threshold=0.5, scores=[0.1], raw_detection_count=1,
                      source_stage="pass1", prompt_mode="per_instance"),
    ]
    recs = [
        make_record(mod, prompt_id="P6", instance_id=0, total_score=0.40, primary_score=0.80),
        make_record(mod, prompt_id="P2", instance_id=0, total_score=0.90, primary_score=0.10),
    ]
    args = make_args(use_prompt_ensemble=True, prompt_selection_mode="weighted_fusion",
                     fusion_threshold=0.5, default_prompt_id="P6")
    mask, pid, score, _ = mod.select_mask(
        Path("f.jpg"), cands, recs, {"P6": "a", "P2": "b"}, args)
    assert pid == "weighted_fusion"
    # P6 dominates under primary; P2's region must be masked out
    assert mask[0, 0], "P6 region must survive fusion"
    assert not mask[5, 5], "P2 region must be suppressed under R1 weighting"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
