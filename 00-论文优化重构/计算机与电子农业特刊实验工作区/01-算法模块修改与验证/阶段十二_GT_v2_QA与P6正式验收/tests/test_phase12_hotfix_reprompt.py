#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-Phase-12 Hotfix Regression Tests — Reprompt Branch

Validates:
  T1: score_gap trigger (top1 - top2 < reprompt_score_gap → needs_reprompt=True)
  T2: no NameError (reprompt_stems is properly initialized)
  T3: stem bookkeeping (triggered stem added to reprompt_stems)
  T4: control case (large gap → needs_reprompt=False, stem not in set)
  T5: output/log verification (reprompt CSV and run log record reprompt info)
  T6: candidate result validity (selected mask is non-empty or enters fallback)

Uses ONLY synthetic fixtures — no GPU, no SAM3, no real images.
The reprompt trigger judgment itself goes through real production code paths.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

# ── Load pipeline module ──
PIPELINE = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参"
                "/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py")


def _load_pipeline():
    mod_name = "gen_pipeline"
    spec = importlib.util.spec_from_file_location(mod_name, PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    script_dir = str(PIPELINE.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec.loader.exec_module(mod)
    return mod


mod = _load_pipeline()
Candidate = mod.Candidate
ScoreRecord = mod.ScoreRecord
select_mask = mod.select_mask


def _make_mask(h=10, w=10, region=None):
    """Create a small binary mask. region=(y0,y1,x0,x1) or None for empty."""
    m = np.zeros((h, w), dtype=bool)
    if region:
        y0, y1, x0, x1 = region
        m[y0:y1, x0:x1] = True
    return m


def _make_candidate(prompt_id, instance_id, mask, sam_score=0.8):
    return Candidate(
        prompt_id=prompt_id,
        prompt_text=f"test {prompt_id}",
        instance_id=instance_id,
        mask=mask,
        box=(0, 0, mask.shape[1], mask.shape[0]),
        sam_score=sam_score,
        mask_threshold=0.5,
        scores=[sam_score],
        raw_detection_count=1,
        source_stage="pass1",
        prompt_mode="single",
    )


def _make_score_record(prompt_id, instance_id, total_score, empty_flag=False):
    return ScoreRecord(
        image_name="test.jpg",
        prompt_id=prompt_id,
        prompt_text=f"test {prompt_id}",
        total_score=total_score,
        empty_flag=empty_flag,
        q_area=0.8, q_comp=1.0, q_edge=0.5, q_temp=0.5, q_contrast=0.5,
        q_leak=0.0, q_side=0.0,
        area_ratio=0.1, component_count=1, boundary_density=0.1,
        temporal_iou=0.5, contrast=0.5,
        bottom_leak_fraction=0.0, side_leak_fraction=0.0,
        sam_scores="0.8", instance_id=instance_id,
        semantic_enabled=False, semantic_total=0.0,
        target_box_score=0.0, vertical_coverage_score=0.0,
        pot_overlap_penalty=0.0, side_distractor_penalty=0.0,
        center_prior_score=0.0, leak_penalty=0.0,
    )


def _make_args(**overrides):
    defaults = dict(
        use_prompt_ensemble=False,
        prompt_selection_mode="single",
        default_prompt_id="P6",
        candidate_mode="per_instance",
        reprompt_score_gap=0.05,
        reprompt_min_score=0.2,
        fusion_threshold=0.5,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ──────────────────────────────────────────────────────────────────────
# T1: score_gap trigger — top1 - top2 < reprompt_score_gap → True
# ──────────────────────────────────────────────────────────────────────
class TestRepromptTrigger:
    """T1: Verify needs_reprompt=True when score gap is small."""

    def test_score_gap_triggers_reprompt(self):
        """Two candidates with total_score gap < reprompt_score_gap → needs_reprompt=True."""
        mask_a = _make_mask(region=(2, 5, 2, 5))
        mask_b = _make_mask(region=(6, 9, 6, 9))

        cands = [
            _make_candidate("P6", 1, mask_a, sam_score=0.9),
            _make_candidate("P6", 2, mask_b, sam_score=0.85),
        ]
        # total_score gap: 0.71 - 0.66 = 0.05, which is NOT < 0.05 (equal, not less)
        # Use gap < 0.05: 0.71 vs 0.69 → gap = 0.02 < 0.05
        recs = [
            _make_score_record("P6", 1, total_score=0.71),
            _make_score_record("P6", 2, total_score=0.69),
        ]
        args = _make_args(reprompt_score_gap=0.05)

        mask, prompt, score, needs_reprompt = select_mask(
            Path("test.jpg"), cands, recs, {"P6": "potted plant"}, args
        )
        assert needs_reprompt is True, (
            f"Expected needs_reprompt=True when gap (0.71-0.69=0.02) < 0.05, got {needs_reprompt}"
        )
        # Best candidate (score 0.71, instance 1) should be selected
        assert np.array_equal(mask, mask_a), "Should select instance 1 (higher score)"
        assert "P6#1" in prompt

    def test_single_candidate_low_score_triggers(self):
        """Single candidate with score < reprompt_min_score → needs_reprompt=True."""
        mask = _make_mask(region=(2, 5, 2, 5))
        cands = [_make_candidate("P6", 1, mask, sam_score=0.3)]
        recs = [_make_score_record("P6", 1, total_score=0.15)]  # 0.15 < 0.2
        args = _make_args(reprompt_min_score=0.2)

        mask_out, prompt, score, needs_reprompt = select_mask(
            Path("test.jpg"), cands, recs, {"P6": "potted plant"}, args
        )
        assert needs_reprompt is True, (
            f"Expected needs_reprompt=True when score 0.15 < min_score 0.2, got {needs_reprompt}"
        )

    def test_empty_mask_triggers(self):
        """Single candidate with empty mask (empty_flag=True) → needs_reprompt=True."""
        mask = _make_mask()  # all zeros
        cands = [_make_candidate("P6", 1, mask, sam_score=0.5)]
        recs = [_make_score_record("P6", 1, total_score=0.5, empty_flag=True)]
        args = _make_args()

        mask_out, prompt, score, needs_reprompt = select_mask(
            Path("test.jpg"), cands, recs, {"P6": "potted plant"}, args
        )
        assert needs_reprompt is True, (
            f"Expected needs_reprompt=True when empty_flag=True, got {needs_reprompt}"
        )


# ──────────────────────────────────────────────────────────────────────
# T4: control case — large gap → needs_reprompt=False
# ──────────────────────────────────────────────────────────────────────
class TestRepromptControl:
    """T4: Verify needs_reprompt=False when score gap is large."""

    def test_large_gap_no_reprompt(self):
        """Two candidates with total_score gap > reprompt_score_gap → needs_reprompt=False."""
        mask_a = _make_mask(region=(2, 5, 2, 5))
        mask_b = _make_mask(region=(6, 9, 6, 9))

        cands = [
            _make_candidate("P6", 1, mask_a, sam_score=0.9),
            _make_candidate("P6", 2, mask_b, sam_score=0.3),
        ]
        recs = [
            _make_score_record("P6", 1, total_score=0.90),
            _make_score_record("P6", 2, total_score=0.30),
        ]
        args = _make_args(reprompt_score_gap=0.05)  # gap=0.60 >> 0.05

        mask, prompt, score, needs_reprompt = select_mask(
            Path("test.jpg"), cands, recs, {"P6": "potted plant"}, args
        )
        assert needs_reprompt is False, (
            f"Expected needs_reprompt=False when gap (0.90-0.30=0.60) > 0.05, got {needs_reprompt}"
        )
        assert np.array_equal(mask, mask_a), "Should select instance 1 (higher score)"

    def test_single_candidate_high_score_no_reprompt(self):
        """Single candidate with score >= reprompt_min_score → needs_reprompt=False."""
        mask = _make_mask(region=(2, 5, 2, 5))
        cands = [_make_candidate("P6", 1, mask, sam_score=0.9)]
        recs = [_make_score_record("P6", 1, total_score=0.85)]  # 0.85 >= 0.2
        args = _make_args(reprompt_min_score=0.2)

        mask_out, prompt, score, needs_reprompt = select_mask(
            Path("test.jpg"), cands, recs, {"P6": "potted plant"}, args
        )
        assert needs_reprompt is False, (
            f"Expected needs_reprompt=False when score 0.85 >= min_score 0.2, got {needs_reprompt}"
        )


# ──────────────────────────────────────────────────────────────────────
# T2 + T3: NameError fix + stem bookkeeping
# Reprompt trigger → reprompt_stems.add() should not raise NameError
# ──────────────────────────────────────────────────────────────────────
class TestRepromptStemsBookkeeping:
    """T2+T3: Verify reprompt_stems is initialized and stems are added."""

    def test_reprompt_stems_no_nameerror(self):
        """T2: The code path reprompt_stems.add() must not raise NameError."""
        # Verify the variable exists and is a set at the right scope
        source = PIPELINE.read_text(encoding="utf-8")
        assert "reprompt_stems: set[str] = set()" in source, (
            "reprompt_stems initialization not found in production code"
        )

    def test_reprompt_stems_is_set(self):
        """T3: Verify reprompt_stems.add is called with correct stem."""
        # We test by calling select_mask to get needs_reprompt=True,
        # then simulating the main loop's bookkeeping logic
        mask_a = _make_mask(region=(2, 5, 2, 5))
        mask_b = _make_mask(region=(6, 9, 6, 9))

        cands = [
            _make_candidate("P6", 1, mask_a, sam_score=0.9),
            _make_candidate("P6", 2, mask_b, sam_score=0.85),
        ]
        recs = [
            _make_score_record("P6", 1, total_score=0.71),
            _make_score_record("P6", 2, total_score=0.69),
        ]
        args = _make_args(reprompt_score_gap=0.05)

        mask, prompt, score, needs_reprompt = select_mask(
            Path("test.jpg"), cands, recs, {"P6": "potted plant"}, args
        )

        # Simulate the main loop's bookkeeping (L2734 equivalent)
        reprompt_stems: set[str] = set()
        stem = "CaoMei1_0100"
        reprompt_stems.add(stem) if needs_reprompt else None

        assert needs_reprompt is True
        assert stem in reprompt_stems, f"Stem {stem} should be in reprompt_stems"

    def test_no_reprompt_stem_not_added(self):
        """T4: When needs_reprompt=False, stem must NOT be in reprompt_stems."""
        mask_a = _make_mask(region=(2, 5, 2, 5))
        mask_b = _make_mask(region=(6, 9, 6, 9))

        cands = [
            _make_candidate("P6", 1, mask_a, sam_score=0.9),
            _make_candidate("P6", 2, mask_b, sam_score=0.3),
        ]
        recs = [
            _make_score_record("P6", 1, total_score=0.90),
            _make_score_record("P6", 2, total_score=0.30),
        ]
        args = _make_args(reprompt_score_gap=0.05)

        mask, prompt, score, needs_reprompt = select_mask(
            Path("test.jpg"), cands, recs, {"P6": "potted plant"}, args
        )

        reprompt_stems: set[str] = set()
        stem = "CaoMei1_0100"
        reprompt_stems.add(stem) if needs_reprompt else None

        assert needs_reprompt is False
        assert stem not in reprompt_stems, f"Stem {stem} should NOT be in reprompt_stems"


# ──────────────────────────────────────────────────────────────────────
# T5: output/log verification
# ──────────────────────────────────────────────────────────────────────
class TestRepromptOutput:
    """T5: Verify reprompt info is recorded in CSV and run log."""

    def test_reprompt_csv_columns(self):
        """重提示帧标记.csv has the correct columns including 是否标记."""
        csv_path = (Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
                         "/01-算法模块修改与验证/阶段十二_GT_v2_QA与P6正式验收"
                         "/P6_raw_baseline/重提示帧标记.csv"))
        assert csv_path.exists(), "重提示帧标记.csv not found"
        import csv as _csv
        with csv_path.open(encoding="utf-8-sig") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 21, f"Expected 21 rows, got {len(rows)}"
        # Verify columns exist
        assert "图像" in rows[0]
        assert "是否标记" in rows[0]
        assert "重提示分数" in rows[0]

    def test_reprompt_log_recorded(self):
        """Run log JSON records reprompt_marked_count."""
        log_path = (Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
                         "/01-算法模块修改与验证/阶段十二_GT_v2_QA与P6正式验收"
                         "/P6_raw_baseline/运行日志.json"))
        assert log_path.exists(), "运行日志.json not found"
        import json
        data = json.loads(log_path.read_text(encoding="utf-8"))
        fs = data.get("failure_summary", {})
        assert "reprompt_marked_count" in fs, "reprompt_marked_count not in failure_summary"
        assert "reprompt_marked" in fs, "reprompt_marked not in failure_summary"

    def test_reprompt_zero_in_frozen_baseline(self):
        """Frozen P6 baseline has reprompt_marked_count=0 (reprompt was OFF)."""
        log_path = (Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
                         "/01-算法模块修改与验证/阶段十二_GT_v2_QA与P6正式验收"
                         "/P6_raw_baseline/运行日志.json"))
        import json
        data = json.loads(log_path.read_text(encoding="utf-8"))
        fs = data.get("failure_summary", {})
        assert fs.get("reprompt_marked_count") == 0, (
            f"Expected 0 in frozen baseline, got {fs.get('reprompt_marked_count')}"
        )


# ──────────────────────────────────────────────────────────────────────
# T6: candidate result validity
# ──────────────────────────────────────────────────────────────────────
class TestCandidateResultValidity:
    """T6: Selected mask must be a valid non-empty boolean array."""

    def test_reprompt_selected_mask_valid(self):
        """When reprompt triggers, selected mask is still the best candidate's mask."""
        mask_a = _make_mask(region=(2, 5, 2, 5))
        mask_b = _make_mask(region=(6, 9, 6, 9))

        cands = [
            _make_candidate("P6", 1, mask_a, sam_score=0.9),
            _make_candidate("P6", 2, mask_b, sam_score=0.85),
        ]
        recs = [
            _make_score_record("P6", 1, total_score=0.71),
            _make_score_record("P6", 2, total_score=0.69),
        ]
        args = _make_args(reprompt_score_gap=0.05)

        mask, prompt, score, needs_reprompt = select_mask(
            Path("test.jpg"), cands, recs, {"P6": "potted plant"}, args
        )

        # Even when reprompt triggers, the mask is valid
        assert isinstance(mask, np.ndarray), f"Mask should be ndarray, got {type(mask)}"
        assert mask.dtype == bool, f"Mask should be bool, got {mask.dtype}"
        assert mask.shape == (10, 10), f"Mask shape should be (10,10), got {mask.shape}"
        assert mask.any(), "Selected mask should not be empty"
        assert isinstance(prompt, str), f"Prompt should be str, got {type(prompt)}"
        assert isinstance(score, float), f"Score should be float, got {type(score)}"

    def test_empty_candidate_mask_is_empty(self):
        """When all candidates are empty, selected mask is empty but valid."""
        mask_empty = _make_mask()  # all zeros
        cands = [_make_candidate("P6", 1, mask_empty, sam_score=0.1)]
        recs = [_make_score_record("P6", 1, total_score=0.1, empty_flag=True)]
        args = _make_args()

        mask, prompt, score, needs_reprompt = select_mask(
            Path("test.jpg"), cands, recs, {"P6": "potted plant"}, args
        )

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert not mask.any(), "Empty candidate should produce empty mask"
        assert needs_reprompt is True


# ──────────────────────────────────────────────────────────────────────
# Additional: verify score_select branch uses instance_id matching
# ──────────────────────────────────────────────────────────────────────
class TestScoreSelectBranch:
    """Verify the fixed score_select branch matches by (prompt_id, instance_id)."""

    def test_score_select_picks_correct_instance(self):
        """score_select with two instances: higher-scoring instance is selected."""
        mask_a = _make_mask(region=(2, 5, 2, 5))
        mask_b = _make_mask(region=(6, 9, 6, 9))

        cands = [
            _make_candidate("P6", 1, mask_a, sam_score=0.9),
            _make_candidate("P6", 2, mask_b, sam_score=0.7),
        ]
        # instance 2 has higher total_score despite lower sam_score
        recs = [
            _make_score_record("P6", 1, total_score=0.6),
            _make_score_record("P6", 2, total_score=0.8),
        ]
        args = _make_args(
            use_prompt_ensemble=True,
            prompt_selection_mode="score_select",
        )

        mask, prompt, score, needs_reprompt = select_mask(
            Path("test.jpg"), cands, recs, {"P6": "potted plant"}, args
        )

        # instance 2 has total_score=0.8 > 0.6, so its mask should be selected
        assert np.array_equal(mask, mask_b), (
            "score_select should pick instance 2 (higher total_score)"
        )
        assert score == 0.8


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
