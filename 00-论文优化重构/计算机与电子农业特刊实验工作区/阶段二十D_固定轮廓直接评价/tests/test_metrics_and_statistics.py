from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from direct_common import METRICS, PHASEC, ROOT, load_module, metric_function
from evaluate_test32 import direction, overlap, pair_rows, summarize
from run_frozen import OFF, WORK, command_for, frozen_parameters


@pytest.fixture(scope="module")
def evaluate():
    return metric_function()


@pytest.mark.parametrize("empty", [False, True])
def test_confusion_and_empty_conventions(evaluate, empty):
    ground = np.zeros((32, 32), dtype=bool)
    prediction = np.zeros_like(ground)
    if not empty:
        ground[8:24, 8:24] = True
        prediction[10:24, 8:22] = True
    values = evaluate(ground, prediction)
    assert values["tp"] + values["fp"] + values["fn"] + values["tn"] == ground.size
    denominator = 2 * values["tp"] + values["fp"] + values["fn"]
    assert values["f1"] == (2 * values["tp"] / denominator if denominator else 0)
    assert values["leakage_energy"] == values["fp"] / ground.size
    if empty:
        assert values["boundary_f1"] == 1 and values["hd95_px"] == 0


def test_original_metric_implementation_is_used(evaluate):
    legacy = load_module("direct_original_legacy_metric_test", ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/03-扩展实验后续执行/实验二_四样本提示词鲁棒性汇总.py")
    ground = np.zeros((48, 48), dtype=bool)
    prediction = np.zeros_like(ground)
    ground[5:25, 7:21] = True
    prediction[7:27, 7:21] = True
    values = evaluate(ground, prediction)
    assert values["hd95_px"] == legacy.hd95(ground, prediction)
    assert values["boundary_f1"] == legacy.boundary_f1(ground, prediction, 3)
    for key, value in legacy.pixel_metrics(*legacy.confusion(ground, prediction)).items():
        assert values[key] == value


def test_single_empty_boundary_convention(evaluate):
    ground = np.zeros((32, 32), dtype=bool)
    ground[2:8, 2:8] = True
    values = evaluate(ground, np.zeros_like(ground))
    assert values["hd95_px"] == np.hypot(32, 32)
    assert values["boundary_f1"] == 0


def statistical_rows():
    return [{"frame_id": f"{sample}_{index}", "sample": sample, "method": method, "rasterizer": "opencv",
             "geometry_warning": "NUMERIC_GEOMETRY_ONLY", "empty_mask": 0, "severe_f1_lt_0_5": 0,
             **{metric: value + (0.1 if method == "R1" else 0) for metric in METRICS}}
            for method in ("R0", "R1") for sample, index, value in (("a", 0, 0.1), ("a", 1, 0.3), ("b", 0, 0.9))]


def test_sample_and_frame_equal_are_distinct():
    rows = [row for row in statistical_rows() if row["method"] == "R0"]
    summary = {row["aggregation"]: row for row in summarize(rows, "opencv", "fixture", "R0")}
    assert summary["sample_equal_mean"]["f1"] == pytest.approx(0.55)
    assert summary["frame_equal_mean"]["f1"] == pytest.approx(1.3 / 3)
    assert summary["frame_sd_population_ddof0"]["f1"] == pytest.approx(np.std([0.1, 0.3, 0.9], ddof=0))


def test_paired_difference_not_unpaired_subset():
    pairs = pair_rows(statistical_rows())
    assert len(pairs) == 3
    assert all(row["f1"] == pytest.approx(0.1) for row in pairs)
    assert direction(-0.01) == "REGRESSED"
    assert direction(0.01) == "IMPROVED"
    assert direction(1e-14) == "UNCHANGED"


def test_policy_is_only_configuration_change():
    baseline = frozen_parameters("base", WORK / "fixture_inputs", WORK / "fixture_outputs")
    treatment = frozen_parameters("r1", WORK / "fixture_inputs", WORK / "fixture_outputs")
    assert {key for key in baseline if baseline[key] != treatment[key]} == {"ranking_policy"}
    assert not any(treatment[key] for key in OFF)
    assert treatment["score_weights"] == "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5"
    assert "--candidate_metadata_strict" in command_for(treatment)
    assert "--reuse_existing_candidates" in command_for(treatment)


def test_oracle_confusion_exactness():
    ground = np.zeros((8, 8), dtype=bool)
    ground[2:6, 2:6] = True
    assert overlap(ground, ground) == (1.0, 1.0)
    assert overlap(ground, ~ground) == (0.0, 0.0)
