import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase20e_common import GS, SAMPLES, WORK, read_csv


def completed(path):
    if not path.exists():
        pytest.skip("Final stage validation not completed")
    return json.loads(path.read_text())


def test_all_fifteen_gates_have_real_evidence():
    result = completed(WORK / "logs/G1_G15_VALIDATION.json")
    assert result["status"] == "PASS"
    assert result["checks"] == 45
    assert not result["failed"]
    rows = read_csv(WORK / "reports/ACCEPTANCE_GATES_G1_G15.csv")
    for sample in SAMPLES:
        gates = [row for row in rows if row["sample"] == sample]
        assert {row["gate"] for row in gates} == {f"G{number}" for number in range(1, 16)}
        assert all(row["status"] == "PASS" and row["evidence"] for row in gates)


def test_delivery_does_not_create_training_artifacts():
    delivery = completed(WORK / "logs/DELIVERY_VERIFICATION.json")
    assert delivery["status"] == "PASS"
    assert delivery["registered_views"] == 532
    assert delivery["dataset_views"] == 1596
    assert delivery["protected_sources"] >= 7914
    assert delivery["optimizer_steps"] == delivery["checkpoints_created"] == 0
    assert not delivery["formal_2dgs_training"]
    assert not delivery["mesh_reconstruction"]
    assert not delivery["outside_stage_changes"]


def test_predeclared_registered_split_survives_raw_identity_correction():
    completed(WORK / "reports/FINAL_ACCEPTANCE.json")
    rows = read_csv(WORK / "reports/PREDECLARED_SPLIT_COMPARISON.csv")
    assert len(rows) == 532
    assert all(row["predeclared_split_unchanged"] == "True" for row in rows)
    caomei = [row for row in rows if row["sample"] == "CaoMei2"]
    assert all(row["prior_raw_identity_corrected"] == "True" for row in caomei)


def test_readable_old_rgba_is_not_falsely_certified_as_missing_mask_recovery():
    result = completed(WORK / "logs/LEGACY_A_RECOVERY_RECHECK.json")
    assert not result["historical_A_restored"]
    rows = json.loads((WORK / "reports/LEGACY_RGBA_RECOVERY_PROOF.json").read_text())
    recovered = [row for row in rows if "historical_missing_alpha_recovery_certified" in row]
    assert len(recovered) == 4
    assert all(not row["historical_missing_alpha_recovery_certified"] and row["different_pixels"] > 0 for row in recovered)


def test_corrupt_input_raw_identity_uses_real_colmap_feature_evidence():
    proof = completed(WORK / "reports/WanNianQing1_0136_SIFT_KEYPOINT_IDENTITY.json")
    assert proof["status"] == "PASS"
    assert proof["raw_RGB_camera_identity_verified"]
    assert not proof["historical_undistorted_RGB_same_version"]
    assert not proof["formal_training"] and not proof["new_camera_registration"]
    positive = proof["results"][0]
    assert positive["source_raw_frame"] == "0136"
    assert positive["target_registered_image_id"] == 133
    assert positive["tracked_sparse_observations"] > 0
    assert positive["tracked_fraction_le_1px"] > 0.5
    assert all(row["target_fraction_le_1px"] < .05 for row in proof["results"][1:])


@pytest.mark.parametrize("sample", SAMPLES)
def test_explicit_training_configs_only_differ_in_data_output_paths(sample):
    build = completed(WORK / f"reports/{sample}_BUILD.json")
    configs = [json.loads((WORK / f"reports/{sample}_{group}_TRAIN_CONFIG.json").read_text()) for group in build["groups"]]
    parameters = [{key: value for key, value in config["all_parameters"].items() if key not in ("source_path", "model_path")} for config in configs]
    assert parameters[0] == parameters[1] == parameters[2]
    for config in configs:
        assert config["seed"] == 0
        assert config["optimization"]["iterations"] == 30000
        assert config["CUDA_VISIBLE_DEVICES"] == "1"
        assert config["background"] == [0, 0, 0]
        assert config["model"]["resolution"] == 2


def test_next_stage_wrapper_defaults_to_no_optimizer():
    completed(WORK / "reports/WanNianQing1_PREFLIGHT.json")
    result = subprocess.run([str(GS / "venv/bin/python"), str(WORK / "scripts/run_phase20e1.py"),
                             "--sample", "WanNianQing1", "--group", "C_R1"], capture_output=True, text=True, check=True)
    assert "DRY INSPECTION ONLY; NO OPTIMIZER" in result.stdout
    assert not result.stderr
