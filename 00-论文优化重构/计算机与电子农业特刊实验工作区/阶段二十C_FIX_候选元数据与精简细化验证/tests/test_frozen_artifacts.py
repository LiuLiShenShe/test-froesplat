from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from fix_common import PROD, WORK, read_csv, sha256


def test_ranking_policy_is_byte_identical_to_frozen_source():
    protected = json.loads((WORK / "PROTECTED_SOURCE_HASHES.json").read_text())
    assert sha256(PROD / "ranking_policy.py") == protected["ranking_policy_sha256"]


def test_all_archived_scores_replay_without_selection_changes():
    audit = json.loads((WORK / "SCORE_REPLAY_AUDIT.json").read_text())
    assert audit["candidates"] == audit["score_equivalent_candidates"] == 55
    assert audit["maximum_historical_total_error"] < audit["tolerance"]
    assert audit["restored_selection_changes"] == 0
    certificate = json.loads((WORK / "ROUNDING_CERTIFICATE_AUDIT.json").read_text())
    assert certificate["certified"] == certificate["pairs"] == 62
    assert certificate["minimum_worst_case_margin"] > 0


def test_hard_cases_still_select_original_target_instances():
    rows = read_csv(WORK / "R0_R1_SELECTION_REPLAY.csv")
    cases = {row["frame"]: row for row in rows if row["sample"] == "DouBanLv2" and row["run"] == "restored_R1"}
    for frame, instance in [("0039", "01"), ("0040", "00"), ("0041", "03")]:
        assert cases[frame]["selected_candidate_id"] == f"DouBanLv2_{frame}_{instance}"
        assert float(cases[frame]["f1"]) > 0.98
        assert float(cases[frame]["oracle_regret"]) == 0


def test_v4_v5_only_change_ranking_factor():
    for dataset in ["Easy21", "GT10"]:
        configs = [json.loads((WORK / "V4_V5_CONFIGS" / f"{dataset}_{group}.json").read_text()) for group in ["V4", "V5"]]
        allowed = {"group", "ranking_policy", "output_dir", "command"}
        assert all(configs[0][key] == configs[1][key] for key in configs[0] if key not in allowed)
        for config in configs:
            assert config["use_spnp_refinement"] and not config["use_residual_repair"] and not config["use_corrective_geometry"]
            assert config["colmap_dir"] is None
            assert config["score_weights"] == "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5"
            assert config["mask_threshold"] == 0.5


def test_no_registration_is_fabricated_for_missing_observations():
    rows = [row for row in read_csv(WORK / "COLMAP_MODEL_IMAGE_AUDIT.csv")
            if row["sample"] == "DouBanLv2" and row["frame"] in ["0037", "0039", "0041"]]
    assert len(rows) == 6
    assert all(row["registered"] == "0" and row["decision"] == "GEOMETRY_NOT_AVAILABLE" for row in rows)


def test_test32_is_pending_and_complete_without_accuracy_claims():
    rows = read_csv(WORK / "TEST32_INPUT_AUDIT.csv")
    assert len(rows) == 32
    assert all(row["input_identity_match"] == "1" and row["candidate_complete"] == "1" for row in rows)
    assert all(row["annotation_found"] == "0" and row["plant_label_status"] == "NOT_MANUALLY_CONFIRMED" for row in rows)


def test_summaries_keep_aggregation_separate_when_available():
    path = WORK / "V4_V5_METRICS.csv"
    if not path.exists():
        pytest.skip("GPU evaluation has not finished")
    rows = [row for row in read_csv(path) if row["group"] in ["V0", "V1", "V4", "V5"]]
    assert len(rows) == 16
    assert {row["aggregation"] for row in rows} == {"sample_equal_mean", "frame_equal_mean"}
    assert all(row["status"] == "COMPLETE" for row in rows)
