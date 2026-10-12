from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from direct_common import METRICS, OLD, PROD, WORK, frames, local_path, metric_function, production_module, read_csv, read_mask, sha256
from evaluate_test32 import overlap
from rasterize_contours import rasterize


@pytest.mark.parametrize("rasterizer", ["opencv", "pillow"])
def test_real32_raster_masks_reproduce(rasterizer):
    rows = [row for row in read_csv(WORK / "RASTERIZED_MASK_MANIFEST.csv") if row["rasterizer"] == rasterizer]
    assert len(rows) == 32
    for row in rows:
        source = local_path(row["source_json"])
        annotation = json.loads(source.read_text())
        expected, mapping, counts = rasterize(annotation, rasterizer)
        assert np.array_equal(expected, read_mask(local_path(row["mask_path"])))
        assert sha256(source) == row["source_json_sha256"]
        saved = json.loads(local_path(row["mapping_path"]).read_text())
        assert saved["evaluation_shapes"] == mapping
        assert saved["all_source_shapes"] == annotation["shapes"]


@pytest.mark.parametrize("rasterizer", ["opencv", "pillow"])
def test_all32_both_method_metrics_recompute(rasterizer):
    evaluate = metric_function()
    manifest = {(row["frame_id"], row["rasterizer"]): row for row in read_csv(WORK / "RASTERIZED_MASK_MANIFEST.csv")}
    rows = [row for row in read_csv(WORK / "R0_R1_ALL_RASTER_RESULTS.csv") if row["rasterizer"] == rasterizer]
    assert len(rows) == 64
    cache = {}
    for row in rows:
        gt = manifest[(row["frame_id"], rasterizer)]
        key = (gt["mask_sha256"], row["output_mask_sha256"])
        if key not in cache:
            cache[key] = evaluate(read_mask(local_path(gt["mask_path"])), read_mask(local_path(row["output_mask_path"])))
        for key, value in cache[key].items():
            assert float(row[key]) == pytest.approx(value, rel=0, abs=1e-12)


def test_failures_and_non_switches_not_removed():
    rows = read_csv(WORK / "R0_R1_TEST32_RESULTS.csv")
    assert len(rows) == 32
    assert sum(row["selection_changed"] == "1" for row in rows) == 2
    failure = next(row for row in rows if row["frame_id"] == "WanNianQing1_0223")
    assert float(failure["r0_f1"]) == float(failure["r1_f1"]) == 0
    assert failure["r1_selected_id"] == "WanNianQing1_0223_00"
    assert failure["oracle_candidate_id"] == "WanNianQing1_0223_01"
    assert float(failure["oracle_f1"]) > .98
    assert sum(float(row["r0_f1"]) < .5 for row in rows) == 3
    assert sum(float(row["r1_f1"]) < .5 for row in rows) == 1
    assert sum(float(row["delta_f1"]) < -1e-12 for row in rows) == 0


def test_original13frames_kept_and_component_warning_distinguished():
    rows = read_csv(WORK / "RASTERIZATION_SENSITIVITY.csv")
    assert len(rows) == 32
    assert sum(row["geometry_warning"] == "GEOMETRY_AMBIGUITY" for row in rows) == 13
    assert sum(int(row["main_gt_shape_ambiguous"]) for row in rows) == 10


def test_runtime_only_policy_changed():
    baseline = json.loads((WORK / "logs/full_base_runtime_parameters.json").read_text())
    treatment = json.loads((WORK / "logs/full_r1_runtime_parameters.json").read_text())
    assert {key for key in baseline if baseline[key] != treatment[key]} == {"ranking_policy", "output_dir"}
    assert sha256(PROD / "ranking_policy.py") == "c80442a2161c0560bc7bf53a7714b6d9da5a996e2a824716a0b0dc2092071905"


def test_every_actual_oracle_candidate_score():
    manifest = {(row["frame_id"], row["rasterizer"]): row for row in read_csv(WORK / "RASTERIZED_MASK_MANIFEST.csv")}
    candidates = {row["candidate_id"]: row for row in read_csv(OLD / "TEST32_CANDIDATE_PROVENANCE.csv")}
    rows = read_csv(WORK / "ORACLE_ALL_CANDIDATES.csv")
    assert len(rows) == 142
    for row in rows:
        ground = read_mask(local_path(manifest[(row["frame_id"], row["rasterizer"])]["mask_path"]))
        prediction = read_mask(local_path(candidates[row["candidate_id"]]["mask_path"]))
        f1, iou = overlap(ground, prediction)
        assert float(row["f1"]) == pytest.approx(f1, rel=0, abs=1e-12)
        assert float(row["iou_fg"]) == pytest.approx(iou, rel=0, abs=1e-12)


def test_difficulty_counts_from_existing_cells():
    manual = read_csv(OLD / "DIFFICULTY_HUMAN_VERIFICATION.csv")
    expected = {code: sum(row[code] == "YES" for row in manual) for code in ("D1", "D3", "D6", "D11", "D12")}
    assert expected == {"D1": 12, "D3": 16, "D6": 15, "D11": 16, "D12": 12}
    for row in read_csv(WORK / "DIFFICULTY_RESULTS.csv"):
        if row["difficulty"] in expected:
            assert int(row["frames"]) == expected[row["difficulty"]]


def test_complete_real_figures_and_fixed_pilot_split():
    figures = read_csv(WORK / "figures/FIGURE_SOURCE_INDEX.csv")
    assert {row["frame_id"] for row in figures} == {frame["frame_id"] for frame in frames()}
    for row in figures:
        assert sha256(local_path(row["figure_path"])) == row["figure_sha256"]
    pilot = read_csv(WORK / "PHASE20E_VIEW_MANIFEST.csv")
    assert len(pilot) == 532
    for row in pilot:
        assert row["frozen_split"] == ("validation" if int(row["split_order_index"]) % 8 == 0 else "train")
    assert sum(int(row["full_view_P6_candidate_verified"]) for row in pilot) == 6


def test_per_sequence_statistics_independently_recompute():
    results = read_csv(WORK / "R0_R1_ALL_RASTER_RESULTS.csv")
    saved = read_csv(WORK / "R0_R1_TEST32_PER_SEQUENCE.csv")
    assert len(saved) == 288
    for row in saved:
        baseline = [item for item in results if item["rasterizer"] == row["rasterizer"] and item["sample"] == row["sample"] and item["method"] == "R0"]
        treatment = [next(item for item in results if item["rasterizer"] == row["rasterizer"] and item["frame_id"] == first["frame_id"] and item["method"] == "R1") for first in baseline]
        assert len(baseline) == int(row["frames"]) == 4
        for metric in METRICS:
            first = np.asarray([float(item[metric]) for item in baseline])
            second = np.asarray([float(item[metric]) for item in treatment])
            values = first if row["method"] == "R0" else second if row["method"] == "R1" else second - first
            aggregation = row["aggregation"]
            if aggregation.endswith("mean"):
                expected = np.mean(values)
            elif aggregation == "frame_median":
                expected = np.median(values)
            elif aggregation == "frame_sd_population_ddof0":
                expected = np.std(values, ddof=0)
            elif aggregation == "sequence_mean_median":
                expected = np.mean(values)
            elif aggregation == "sequence_mean_sd_population_ddof0":
                expected = 0.0
            else:
                raise AssertionError(aggregation)
            assert float(row[metric]) == pytest.approx(expected, rel=0, abs=1e-12)


def test_overlapping_difficulty_statistics_independently_recompute():
    tables = {rasterizer: read_csv(WORK / filename) for rasterizer, filename in (("opencv", "R0_R1_TEST32_RESULTS.csv"), ("pillow", "R0_R1_TEST32_SECONDARY_RESULTS.csv"))}
    for row in read_csv(WORK / "DIFFICULTY_RESULTS.csv"):
        identifiers = set(row["frame_ids"].split(";"))
        selected = [item for item in tables[row["rasterizer"]] if item["frame_id"] in identifiers]
        assert len(selected) == int(row["frames"])
        assert len({item["sample"] for item in selected}) == int(row["sequences"])
        for prefix in ("r0", "r1", "delta"):
            for metric in METRICS:
                if row["aggregation"] == "sample_equal_mean":
                    values = [np.mean([float(item[f"{prefix}_{metric}"]) for item in selected if item["sample"] == sample]) for sample in sorted({item["sample"] for item in selected})]
                else:
                    values = [float(item[f"{prefix}_{metric}"]) for item in selected]
                assert float(row[f"{prefix}_{metric}"]) == pytest.approx(np.mean(values), rel=0, abs=1e-12)
        assert int(row["regressed_frames"]) == sum(float(item["delta_f1"]) < -1e-12 for item in selected)


def test_cross_raster_overlap_and_area_differences_recompute():
    manifest = {(row["frame_id"], row["rasterizer"]): row for row in read_csv(WORK / "RASTERIZED_MASK_MANIFEST.csv")}
    for row in read_csv(WORK / "RASTERIZATION_SENSITIVITY.csv"):
        first, second = [read_mask(local_path(manifest[(row["frame_id"], rasterizer)]["mask_path"])) for rasterizer in ("opencv", "pillow")]
        _, iou = overlap(first, second)
        assert float(row["raster_mask_iou"]) == pytest.approx(iou, rel=0, abs=1e-12)
        assert int(row["area_difference_px"]) == int(second.sum()) - int(first.sum())
        assert int(row["symmetric_difference_px"]) == int((first ^ second).sum())


def test_all_frozen_score_decompositions_against_production_ranking():
    production_module()
    ranking = sys.modules["ranking_policy"]
    provenance = {row["candidate_id"]: row for row in read_csv(OLD / "TEST32_CANDIDATE_PROVENANCE.csv")}
    saved = read_csv(WORK / "FROZEN_SCORE_DECOMPOSITION.csv")
    assert len(saved) == 142
    assert len({(row["policy"], row["candidate_id"]) for row in saved}) == 142
    for row in saved:
        source = provenance[row["candidate_id"]]
        assert row["source_candidate_sha256"] == source["mask_sha256"]
        assert row["source_detail_sha256"] == source["source_detail_sha256"]
        assert row["original_box"] == source["box"]
        assert float(row["sam_score_archived"]) == float(source["sam_score_archived_text"])
        confidence = 1 / (1 + math.exp(-6 * (float(row["sam_score_archived"]) - 0.5)))
        assert float(row["q_sam"]) == pytest.approx(confidence, rel=0, abs=1e-12)
        total = sum(float(row[f"weighted_{component}"]) for component in ("area", "comp", "edge", "temp", "contrast", "sam")) / 5.5
        assert float(row["base_total_before_risk"]) == pytest.approx(total, rel=0, abs=1e-12)
        primary, tie = ranking.primary_and_tiebreak(total, float(row["weighted_contrast"]), float(row["weighted_area"]), 1.0, 5.5, float(row["semantic_total"]), float(row["risk_penalty"]), row["policy"])
        assert float(row["primary_score"]) == pytest.approx(primary, rel=0, abs=1e-12)
        assert float(row["tie_break_score"]) == pytest.approx(tie, rel=0, abs=1e-12)
