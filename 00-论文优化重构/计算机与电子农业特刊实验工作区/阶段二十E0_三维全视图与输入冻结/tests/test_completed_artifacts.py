import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase20e_common import WORK, binary_mask, read_csv, sha256


@pytest.mark.parametrize("sample,registered", [("CaoMei2", 106), ("WanNianQing1", 213), ("HongZhang", 213)])
def test_completed_candidates_have_complete_identity(sample, registered):
    path = WORK / f"reports/{sample}_FULL_CANDIDATES.csv"
    if not path.exists():
        pytest.skip("Scene inference not completed")
    rows = read_csv(path)
    assert len({row["frame_id"] for row in rows}) == registered
    assert len({row["candidate_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["prompt_id"] == "P6"
        assert float(row["mask_threshold"]) == 0.5
        assert row["metadata_status"] == "VERIFIED"
        assert sha256(row["mask_path"]) == row["mask_sha256"]
        assert row["rgb_sha256"]
        assert json.loads(row["box"]) is not None or row["empty_fallback"] == "True"
        if row["set_type"] == "NEW_FULL_VIEW_CANDIDATE_SET":
            assert row["score_precision"] == "FULL_AVAILABLE_TENSOR_PRECISION"
            assert row["score_tensor_dtype"]


@pytest.mark.parametrize("sample", ["CaoMei2", "WanNianQing1", "HongZhang"])
def test_ranking_uses_one_cache_and_selected_masks(sample):
    path = WORK / f"reports/{sample}_SELECTION.csv"
    if not path.exists():
        pytest.skip("Ranking not completed")
    rows = read_csv(path)
    candidates = {row["candidate_id"]: row for row in read_csv(WORK / f"reports/{sample}_FULL_CANDIDATES.csv")}
    masks = {(row["frame_id"], row["policy"]): row for row in read_csv(WORK / f"reports/{sample}_MASKS.csv")}
    for row in rows:
        actual = {}
        for policy in ("R0", "R1"):
            source = candidates[row[f"{policy}_candidate_id"]]
            output = masks[row["frame_id"], policy]
            assert np.array_equal(binary_mask(source["mask_path"]), binary_mask(output["mask_path"]))
            assert source["mask_sha256"] == output["candidate_sha256"]
            actual[policy] = binary_mask(output["mask_path"])
        changed = actual["R0"] != actual["R1"]
        assert float(row["changed_pixel_ratio"]) == changed.mean()
        assert (row["pixelwise_identity"] == "True") == (not changed.any())


@pytest.mark.parametrize("sample", ["CaoMei2", "WanNianQing1", "HongZhang"])
def test_complete_matched_datasets_and_true_gpu_loader(sample):
    path = WORK / f"reports/{sample}_PREFLIGHT.json"
    if not path.exists():
        pytest.skip("Scene loader not completed")
    preflight = json.loads(path.read_text())
    build = json.loads((WORK / f"reports/{sample}_BUILD.json").read_text())
    assert preflight["status"] == build["status"] == "PASS"
    assert not preflight["formal_training"]
    assert len(preflight["groups"]) == 3
    for group in preflight["groups"]:
        assert group["loaded_views"] == build["views"]
        assert group["optimizer_steps"] == 0
        assert not group["optimizer_created"]
        assert not group["checkpoint_created"]
        assert group["one_real_forward"]
    initial = [group["initialization_sha256"] for group in preflight["groups"]]
    assert initial[0] == initial[1] == initial[2]
    rows = read_csv(WORK / f"reports/{sample}_DATASETS.csv")
    groups = build["groups"]
    assert len(rows) == build["views"] * 3
    for field in ("registered_image_id", "registered_name", "source_rgb_sha256", "common_rgb_sha256", "split", "sparse_bundle", "train_width", "train_height"):
        by_group = [{row["frame_id"]: row[field] for row in rows if row["group"] == group} for group in groups]
        assert by_group[0] == by_group[1] == by_group[2]


@pytest.mark.parametrize("sample", ["CaoMei2", "WanNianQing1", "HongZhang"])
def test_mapping_is_official_camera_based_and_measured(sample):
    path = WORK / f"reports/{sample}_BUILD.json"
    if not path.exists():
        pytest.skip("Mapping not completed")
    build = json.loads(path.read_text())
    assert build["sparse_checks"]["conversion_max_px"] <= 1e-6
    assert build["sparse_checks"]["real_sparse_projections"] > 0
    references = [row for row in read_csv(WORK / f"reports/{sample}_RGB_COMPARISONS.csv") if row["reference"].startswith("NATIVE_")]
    assert len(references) == 2
    assert all(float(row["mean_abs_color_error"]) <= 0.1 and int(row["max_abs_color_error"]) <= 1 for row in references)
