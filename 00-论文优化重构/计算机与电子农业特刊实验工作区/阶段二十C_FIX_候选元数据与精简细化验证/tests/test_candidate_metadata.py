from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
PRODUCTION = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本"
sys.path.insert(0, str(PRODUCTION))
SPEC = importlib.util.spec_from_file_location("phase20c_fix_production_tests", PRODUCTION / "生成RAP-FSAM3掩膜.py")
production = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = production
SPEC.loader.exec_module(production)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def cache(tmp_path):
    image = tmp_path / "sample_0039.jpg"
    Image.new("RGB", (12, 10), "green").save(image)
    source = tmp_path / "cache"
    masks = source / "候选掩膜/raw_instance_P6"
    masks.mkdir(parents=True)
    entries = []
    for instance, score, order in [(10, 0.93, 0), (2, 0.37, 1)]:
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[instance % 3:8, 2:9] = 255
        path = masks / f"mask_{image.stem}_{instance:02d}.png"
        Image.fromarray(mask).save(path)
        entries.append({
            "prompt_id": "P6", "instance_id": instance, "sam_score": score,
            "box": [-1, 2, 10, 9], "mask_threshold": 0.5,
            "candidate_order": order, "mask_sha256": digest(path),
            "scores": [0.93, 0.37], "raw_detection_count": 2,
            "prompt_mode": "per_instance",
        })
    metadata = {"schema_version": 1, "image_name": image.name,
                "image_sha256": digest(image), "candidates": list(reversed(entries))}
    metadata_path = source / "候选掩膜" / f"候选元数据_{image.stem}.json"
    metadata_path.write_text(json.dumps(metadata))
    return image, source, metadata_path, metadata


def load(cache, strict=True):
    image, source, _, _ = cache
    return production.load_existing_candidates(image, ["P6"], {"P6": "whole potted plant including the pot"}, source,
                                               candidate_metadata_strict=strict)


def test_identity_join_not_directory_or_metadata_row_order(cache):
    candidates = load(cache)
    assert [(candidate.instance_id, candidate.sam_score) for candidate in candidates] == [(10, 0.93), (2, 0.37)]
    assert all(candidate.box == (-1, 2, 10, 9) for candidate in candidates)
    assert all(candidate.metadata_status == "VERIFIED" for candidate in candidates)
    assert candidates[0].scores == [0.93, 0.37]


def test_missing_metadata_strict_stops(cache):
    cache[2].unlink()
    with pytest.raises(ValueError, match="METADATA_UNVERIFIED"):
        load(cache)


def test_legacy_missing_metadata_warns(cache):
    cache[2].unlink()
    with pytest.warns(RuntimeWarning, match="METADATA_UNVERIFIED"):
        candidates = load(cache, strict=False)
    assert {candidate.metadata_status for candidate in candidates} == {"METADATA_UNVERIFIED"}
    assert all(candidate.sam_score == 0.0 for candidate in candidates)


@pytest.mark.parametrize("fault", ["duplicate_id", "duplicate_order", "wrong_rgb", "wrong_mask", "missing_score", "nan_score", "wrong_image", "missing_box", "missing_threshold", "missing_order"])
def test_invalid_metadata_is_never_silently_zero_filled(cache, fault):
    _, _, path, metadata = cache
    entry = metadata["candidates"][0]
    if fault == "duplicate_id":
        metadata["candidates"].append(dict(entry))
    elif fault == "duplicate_order":
        metadata["candidates"][1]["candidate_order"] = entry["candidate_order"]
    elif fault == "wrong_rgb":
        metadata["image_sha256"] = "0" * 64
    elif fault == "wrong_mask":
        entry["mask_sha256"] = "0" * 64
    elif fault == "nan_score":
        entry["sam_score"] = float("nan")
    elif fault == "wrong_image":
        metadata["image_name"] = "other_0039.jpg"
    else:
        entry.pop({"missing_score": "sam_score", "missing_box": "box", "missing_threshold": "mask_threshold", "missing_order": "candidate_order"}[fault])
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load(cache, strict=False)


def test_incomplete_instance_set_strict_stops(cache):
    _, _, path, metadata = cache
    metadata["candidates"].pop()
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="METADATA_UNVERIFIED"):
        load(cache)


def test_explicit_external_metadata_directory(cache, tmp_path):
    image, source, path, _ = cache
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    path.rename(metadata_dir / path.name)
    candidates = production.load_existing_candidates(image, ["P6"], {"P6": "text"}, source,
                                                     candidate_metadata_dir=metadata_dir, candidate_metadata_strict=True)
    assert candidates[0].sam_score == 0.93


def test_missing_raw_instance_strict_stops(cache):
    image, source, _, _ = cache
    (source / "候选掩膜/raw_instance_P6" / f"mask_{image.stem}_10.png").unlink()
    with pytest.raises(ValueError, match="incomplete candidate identity"):
        load(cache)


def test_exact_tie_retains_archived_candidate_order(cache):
    candidates = load(cache)
    records = [type("Record", (), {"primary_score": 0.5, "tie_break_score": 0.0, "instance_id": item.instance_id})()
               for item in candidates]
    assert production.select_best_policy(records, "r1").instance_id == 10


def test_historical_csv_uses_instance_identity_and_is_not_hash_verified(cache):
    import csv
    image, source, path, _ = cache
    path.unlink()
    historical = source / "候选掩膜" / f"候选评分明细_{image.stem}.csv"
    rows = [{"图像": image.name, "提示词编号": "P6", "实例编号": instance,
             "SAM3分数": score, "外接框": "1,2,9,8", "掩膜阈值": 0.5, "候选模式": "per_instance"}
            for instance, score in [(10, 0.8), (2, 0.4)]]
    with historical.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.warns(RuntimeWarning, match="METADATA_UNVERIFIED"):
        candidates = load(cache, strict=False)
    assert [(item.instance_id, item.sam_score) for item in candidates] == [(2, 0.4), (10, 0.8)]
    with pytest.raises(ValueError, match="METADATA_UNVERIFIED"):
        load(cache)


def test_legacy_union_json_without_box_warns_instead_of_breaking_cli(cache):
    image, source, path, metadata = cache
    entry = metadata["candidates"][0]
    entry["prompt_mode"] = "legacy_union"
    entry["box"] = None
    path.write_text(json.dumps(metadata))
    with pytest.warns(RuntimeWarning, match="METADATA_UNVERIFIED"):
        candidates = load(cache, strict=False)
    assert candidates[1].box is None
    with pytest.raises(ValueError, match="METADATA_UNVERIFIED"):
        load(cache)


def test_legacy_union_masks_and_csv_remain_usable_with_warning(tmp_path):
    import csv
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (12, 10)).save(image)
    target = tmp_path / "候选掩膜" / production.prompt_dir_name("P6") / "mask_frame.png"
    target.parent.mkdir(parents=True)
    Image.new("L", (12, 10), 255).save(target)
    row = {"图像": image.name, "提示词编号": "P6", "实例编号": 0, "SAM3分数": 0.0,
           "外接框": "", "掩膜阈值": 0.5, "候选模式": "legacy_union"}
    with (tmp_path / "候选掩膜/候选评分明细_frame.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.warns(RuntimeWarning, match="METADATA_UNVERIFIED"):
        candidates = production.load_existing_candidates(image, ["P6"], {"P6": "text"}, tmp_path)
    assert len(candidates) == 1 and candidates[0].prompt_mode == "legacy_union"
    assert candidates[0].mask.all()


def test_unverified_provenance_cannot_be_promoted_by_new_hashes(cache):
    _, _, path, metadata = cache
    metadata["candidates"][0]["metadata_status"] = "METADATA_UNVERIFIED"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="METADATA_UNVERIFIED"):
        load(cache)


def test_archived_score_precision_is_preserved(cache):
    _, _, path, metadata = cache
    for entry in metadata["candidates"]:
        entry["score_precision"] = "ARCHIVED_6_DECIMALS"
    path.write_text(json.dumps(metadata))
    assert all(candidate.score_precision == "ARCHIVED_6_DECIMALS" for candidate in load(cache))
