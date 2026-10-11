from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十C_R1与完整细化链联合验证"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_input_manifest_is_ready_and_complete() -> None:
    rows = read_csv(WORK / "configs/PHASE20C_INPUT_MANIFEST.csv")
    assert len(rows) == 31
    assert all(row["status"] == "READY" for row in rows)
    assert all(row["coordinate_system_ok"] == "True" for row in rows)


def test_frozen_matrix_has_only_two_factors() -> None:
    rows = read_csv(WORK / "configs/PHASE20C_CONFIG_MATRIX.csv")
    assert len(rows) == 8
    for dataset in ("Easy21", "GT10"):
        subset = [row for row in rows if row["dataset"] == dataset]
        assert {row["ranking_policy"] for row in subset} == {"base", "r1"}
        assert {row["spnp"] for row in subset} == {"OFF", "ON"}
        for field in ("prompt", "candidate_mode", "prompt_selection", "prompt_ensemble", "a6", "a7", "candidate_source"):
            assert len({row[field] for row in subset}) == 1


def test_all_configs_keep_a6_a7_off() -> None:
    for path in sorted((WORK / "configs").glob("*.json")):
        if "_V" not in path.name:
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["a6_cross_view_consensus"] is False
        assert config["a7_memory_propagation"] is False
        assert config["candidate_mode"] == "per_instance"
        assert config["prompt_list"] == "P6"


def test_easy21_full_chain_has_real_stage_outputs() -> None:
    metrics = read_csv(WORK / "metrics/summary.csv")
    by_group = {(row["dataset"], row["group"]): row for row in metrics}
    for group in ("V0", "V1", "V2", "V3"):
        assert by_group[("Easy21", group)]["status"] == "COMPLETE"
        assert by_group[("Easy21", group)]["n_frames"] == "21"
    modules = read_csv(WORK / "metrics/module_execution.csv")
    v3_spnp = [row for row in modules if row["dataset"] == "Easy21" and row["group"] == "V3" and row["module"] == "spnp"]
    v3_gcfc = [row for row in modules if row["dataset"] == "Easy21" and row["group"] == "V3" and row["module"] == "gcfc"]
    assert len(v3_spnp) == 21 and all(row["enabled"] == "1" and row["accepted_or_changed"] == "1" for row in v3_spnp)
    assert len(v3_gcfc) == 21 and all(row["enabled"] == "1" and row["accepted_or_changed"] == "1" for row in v3_gcfc)


def test_gt10_v1_rescues_three_core_frames() -> None:
    rows = read_csv(WORK / "metrics/selection_replay.csv")
    selected = {
        row["frame"]: row
        for row in rows
        if row["dataset"] == "GT10" and row["group"] == "V1" and row["sample_id"] == "DouBanLv2" and row["frame"] in {"0039", "0040", "0041"}
    }
    assert set(selected) == {"0039", "0040", "0041"}
    assert all(row["historical_selection_match"] == "1" for row in selected.values())
    assert all(float(row["selected_f1"]) > 0.95 for row in selected.values())


def test_gt10_full_refinement_is_explicitly_blocked() -> None:
    metrics = read_csv(WORK / "metrics/summary.csv")
    by_group = {(row["dataset"], row["group"]): row for row in metrics}
    assert by_group[("GT10", "V2")]["status"] == "BLOCKED"
    assert by_group[("GT10", "V3")]["status"] == "BLOCKED"
    audit = read_csv(WORK / "configs/COLMAP_OBSERVATION_AUDIT.csv")
    missing = {row["stem"] for row in audit if row["dataset"] == "GT10" and row["observation_present"] == "0"}
    assert missing == {"DouBanLv2_0037", "DouBanLv2_0039", "DouBanLv2_0041"}
