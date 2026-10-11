from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from audit_inputs import verify_candidate
from phase20d_common import unique_index, write_csv, write_json
import validate_annotations as gate


def human_review():
    return {"approval": "APPROVED", "annotator": "Synthetic Human A", "reviewer": "Synthetic Human A",
            "second_reviewer": "Synthetic Human B", "annotator_is_human": "YES", "reviewer_is_human": "YES",
            "second_reviewer_is_human": "YES", "manual_annotation_confirmed": "YES",
            "target_identity_confirmed": "YES", "plant_confirmed": "YES", "pot_confirmed": "YES",
            "difficulty_manually_confirmed": "YES", "annotation_origin": "MANUAL_POLYGONS_ONLY",
            "target_identity_description": "Synthetic fixture only, not a real-image observation",
            "reviewed_at": "2026-10-11T10:00:00+08:00", "second_reviewed_at": "2026-10-11T11:00:00+08:00",
            "blue_cube_present": "NO", "human_difficulty_labels": "none",
            "difficulty_evidence_json": '{"none":"Synthetic unit-test evidence only"}'}


def test_real_human_attestation_schema_without_automating_actual_review():
    gate.validate_human_review(human_review())


@pytest.mark.parametrize("field,value", [("approval", "PENDING"), ("annotator", ""),
    ("second_reviewer", "synthetic human a"), ("second_reviewer", "AI"),
    ("reviewer_is_human", ""), ("manual_annotation_confirmed", ""),
    ("annotation_origin", "SAM3"), ("target_identity_confirmed", ""),
    ("plant_confirmed", ""), ("pot_confirmed", ""), ("blue_cube_present", ""),
    ("difficulty_manually_confirmed", ""), ("reviewed_at", ""),
    ("human_difficulty_labels", "D1/D11"), ("human_difficulty_labels", "D1;none"),
    ("difficulty_evidence_json", "{}"), ("difficulty_evidence_json", '{"none":""}')])
def test_incomplete_or_automatic_approval_is_rejected(field, value):
    review = human_review()
    review[field] = value
    with pytest.raises(ValueError):
        gate.validate_human_review(review)


def test_multilabel_requires_specific_evidence_for_every_label():
    review = human_review()
    review["human_difficulty_labels"] = "D1;D11"
    review["difficulty_evidence_json"] = '{"D1":"Fixture overlap","D11":"Fixture clutter"}'
    gate.validate_human_review(review)
    review["difficulty_evidence_json"] = '{"D1":"Fixture overlap"}'
    with pytest.raises(ValueError, match="exactly"):
        gate.validate_human_review(review)


def annotation_fixture():
    return {"imageWidth": 20, "imageHeight": 20, "imagePath": "synthetic_0001.jpg", "shapes": [
        {"label": "potted_plant", "shape_type": "polygon", "points": [[1, 1], [18, 1], [18, 18], [1, 18]]},
        {"label": "plant", "shape_type": "polygon", "points": [[2, 2], [10, 2], [10, 10], [2, 10]]},
        {"label": "pot", "shape_type": "polygon", "points": [[2, 12], [10, 12], [10, 17], [2, 17]]}]}


def verify_fixture(tmp_path, annotation, review_updates=None):
    path = tmp_path / "synthetic_annotation.json"
    path.write_text(json.dumps(annotation))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = {"frame_id": "synthetic_0001", "frame": "0001", "width": 20, "height": 20,
             "rgb_sha256": "synthetic_rgb_digest"}
    review = {**human_review(), "annotation_sha256": digest, "rgb_sha256": frame["rgb_sha256"]}
    if review_updates:
        review.update(review_updates)
    gate.validate_geometry(annotation, frame, review, path)


def test_original_coordinate_polygons_pass(tmp_path):
    verify_fixture(tmp_path, annotation_fixture())


@pytest.mark.parametrize("invalid", ["missing_plant", "linestrip", "grid", "outside", "degenerate", "identity", "cube"])
def test_invalid_manual_geometry_is_blocked(tmp_path, invalid):
    annotation = annotation_fixture()
    updates = None
    if invalid == "missing_plant":
        annotation["shapes"] = [shape for shape in annotation["shapes"] if shape["label"] != "plant"]
    elif invalid == "linestrip":
        annotation["shapes"][0]["shape_type"] = "linestrip"
    elif invalid == "grid":
        annotation["imageWidth"] = 21
    elif invalid == "outside":
        annotation["shapes"][1]["points"][0] = [19, 19]
    elif invalid == "degenerate":
        annotation["shapes"][0]["points"] = [[2, 2], [2, 2], [3, 3]]
    elif invalid == "identity":
        annotation["imagePath"] = "another_frame.jpg"
    elif invalid == "cube":
        updates = {"blue_cube_present": "YES"}
    with pytest.raises(ValueError):
        verify_fixture(tmp_path, annotation, updates)


def test_annotation_changes_invalidate_approval(tmp_path):
    with pytest.raises(ValueError, match="SHA256"):
        verify_fixture(tmp_path, annotation_fixture(), {"annotation_sha256": "old_unapproved_hash"})


def test_duplicate_reviews_cannot_hide_missing_frames():
    with pytest.raises(ValueError, match="Duplicate"):
        unique_index([{"challenge_id": "same"}, {"challenge_id": "same"}], "challenge_id")


def test_output_creation_never_overwrites_review(tmp_path):
    path = tmp_path / "review.csv"
    rows = [{"approval": "PENDING"}]
    write_csv(path, rows)
    with pytest.raises(FileExistsError):
        write_csv(path, [{"approval": "APPROVED"}])
    json_path = tmp_path / "state.json"
    write_json(json_path, {"status": "PENDING"})
    with pytest.raises(FileExistsError):
        write_json(json_path, {"status": "APPROVED"})


def test_explicit_instance_identity_not_diagnostic_rank():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:10, 2:10] = 255
    detail = {"图像": "synthetic_0001.jpg", "提示词编号": "P6", "实例编号": "0", "排名": "9",
              "面积比例": str(64 / 400), "SAM3分数": "0.960938", "外接框": "-5,-1,20,20",
              "候选模式": "per_instance", "掩膜阈值": "0.5"}
    scores = [{"图像": detail["图像"], "提示词编号": "P6", "面积比例": detail["面积比例"],
               "SAM3原始分数": "0.960938"}]
    joined = verify_candidate(detail, scores, mask, 20, 20, 1)
    assert joined["instance_id"] == 0
    assert joined["box"] == [-5, -1, 20, 20]
    assert joined["sam_score"] == 0.960938
    with pytest.raises(ValueError, match="ambiguous"):
        verify_candidate(detail, scores + scores, mask, 20, 20, 1)


def test_difficulty_form_cannot_auto_copy_heuristic_labels():
    review = human_review()
    frame = {"rgb_sha256": "frozen_rgb"}
    row = {"status": "PENDING_HUMAN_VISUAL_REVIEW"}
    with pytest.raises(ValueError, match="not approved"):
        gate.validate_difficulty(row, review, frame)


def test_pending_full_coverage_never_enters_geometry_or_metrics(tmp_path, monkeypatch):
    frames = [{"challenge_id": f"synthetic-{number:02d}", "frame_id": f"synthetic_{number:04d}",
               "input_status": "VERIFIED", "annotation_json": f"annotation_package/annotations/synthetic_{number:04d}.json"}
              for number in range(32)]
    reviews = [{"challenge_id": frame["challenge_id"], "frame_id": frame["frame_id"],
                "annotation_json": frame["annotation_json"], "approval": "PENDING"} for frame in frames]
    difficulty = [{"challenge_id": frame["challenge_id"], "status": "PENDING_HUMAN_VISUAL_REVIEW"} for frame in frames]
    write_csv(tmp_path / "TEST32_INPUT_MANIFEST.csv", frames)
    write_csv(tmp_path / "TEST32_ANNOTATION_REVIEW.csv", reviews)
    write_csv(tmp_path / "DIFFICULTY_HUMAN_VERIFICATION.csv", difficulty)
    monkeypatch.setattr(gate, "verify_frozen_inputs", lambda stage, values: None)
    def forbidden_geometry(*args):
        raise AssertionError("Missing human annotations must not enter geometry or metrics")
    monkeypatch.setattr(gate, "validate_geometry", forbidden_geometry)
    result = gate.preflight(tmp_path)
    assert result["evaluation_allowed"] is False
    assert result["verified_frames"] == 0
    assert result["annotation_files_found"] == 0
    assert result["predictions_accessed"] is False
    assert result["accuracy_metrics_computed"] is False


def test_subset_evaluation_is_rejected(tmp_path):
    write_csv(tmp_path / "TEST32_INPUT_MANIFEST.csv", [{"challenge_id": "single"}])
    with pytest.raises(ValueError, match="exactly 32"):
        gate.preflight(tmp_path)
