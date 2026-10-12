import copy
import json
import sys
from pathlib import Path

import pytest


COMPLETION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPLETION / "scripts"))
import archive_existing_gt as archive
import verify_completion as delivery


def annotation_fixture():
    return {"imageWidth": 20, "imageHeight": 20, "imagePath": "synthetic_0001.jpg",
            "shapes": [{"label": label, "shape_type": "polygon", "points": points}
                       for label, points in (
                           ("potted_plant", [[2, 2], [17, 2], [17, 17], [2, 17]]),
                           ("plant", [[3, 3], [16, 3], [16, 10], [3, 10]]),
                           ("pot", [[5, 11], [14, 11], [14, 16], [5, 16]]))]}


def frame_fixture():
    return {"challenge_id": "synthetic", "frame_id": "synthetic_0001", "sample": "synthetic",
            "frame": "0001", "width": "20", "height": "20", "rgb_sha256": "synthetic_rgb"}


def test_geometry_reuses_original_gate_without_approval(tmp_path):
    path = tmp_path / "synthetic_0001.json"
    path.write_text(json.dumps(annotation_fixture()))
    result = archive.geometry_record(frame_fixture(), path)
    assert result["geometry_status"] == "VALID"
    assert result["human_blue_cube_confirmation"] == "NOT_VERIFIED"
    assert result["potted_clean_status"] == "NONEMPTY"


def test_linestrips_never_become_polygons_or_gt(tmp_path):
    document = annotation_fixture()
    for shape in document["shapes"]:
        shape["shape_type"] = "linestrip"
    before = copy.deepcopy(document)
    issues, details = archive.inspect_shapes(document, 20, 20)
    assert document == before
    assert sum(issue["code"] == "UNSUPPORTED_SHAPE_TYPE" for issue in issues) == 3
    path = tmp_path / "synthetic_0001.json"
    path.write_text(json.dumps(document))
    source_bytes = path.read_bytes()
    result = archive.geometry_record(frame_fixture(), path)
    assert result["geometry_status"] == "INVALID"
    assert result["containment_status"] == "NOT_EVALUATED"
    assert result["potted_clean_status"] == "NOT_EVALUATED"
    assert "Only original-coordinate human polygons" in result["frozen_geometry_error"]
    assert path.read_bytes() == source_bytes


def test_missing_components_are_reported_individually():
    document = annotation_fixture()
    document["shapes"] = document["shapes"][:1]
    issues, details = archive.inspect_shapes(document, 20, 20)
    assert {issue["label"] for issue in issues if issue["code"] == "MISSING_REQUIRED_LABEL"} == {"plant", "pot"}
    assert len(document["shapes"]) == len(details) == 1


@pytest.mark.parametrize("point", [[-1, 5], [20, 5], [5, 20], [float("nan"), 2]])
def test_illegal_points_include_exact_shape_location(point):
    document = annotation_fixture()
    document["shapes"][1]["points"][0] = point
    issues, details = archive.inspect_shapes(document, 20, 20)
    assert any(issue["shape_index"] == 1 and issue["code"] in ("OUT_OF_BOUNDS", "INVALID_POINTS") for issue in issues)


def test_containment_cannot_be_repaired_by_archive(tmp_path):
    document = annotation_fixture()
    document["shapes"][1]["points"][0] = [1, 1]
    path = tmp_path / "synthetic_0001.json"
    path.write_text(json.dumps(document))
    result = archive.geometry_record(frame_fixture(), path)
    assert result["geometry_status"] == "INVALID"
    assert "extends outside" in result["frozen_geometry_error"]


def test_manual_difficulty_derivation_preserves_frame_differences():
    first = {code: "NO" for code in archive.DIFFICULTIES}
    second = {**first, "D1": "YES", "D12": "YES"}
    assert archive.derive_labels(first) == "none"
    assert archive.derive_labels(second) == "D1;D12"
    assert first == {code: "NO" for code in archive.DIFFICULTIES}
    with pytest.raises(ValueError):
        archive.derive_labels({**first, "D3": ""})


def test_objective_filling_never_overwrites_existing_values():
    row = {"annotation_sha256": "", "reviewer": "human_private_code", "approval": "PENDING"}
    archive.preserve_or_fill(row, "annotation_sha256", "true_source_hash")
    assert row["reviewer"] == "human_private_code"
    assert row["approval"] == "PENDING"
    with pytest.raises(ValueError, match="manual resolution"):
        archive.preserve_or_fill(row, "annotation_sha256", "different_source_hash")
    assert row["annotation_sha256"] == "true_source_hash"


def test_hash_and_derived_labels_cannot_approve_human_gate():
    row = {"annotation_sha256": "true_source_hash", "human_difficulty_labels": "none", "approval": "PENDING"}
    with pytest.raises(ValueError, match="human approval missing"):
        archive.original_gate.validate_human_review(row)
    assert "reviewer" not in row
    assert "reviewed_at" not in row
    assert "difficulty_evidence_json" not in row


def test_non_destructive_symlink_registration(tmp_path):
    source = tmp_path / "rgb/fixture.json"
    source.parent.mkdir()
    source.write_text(json.dumps(annotation_fixture()))
    before = source.read_bytes()
    destination = tmp_path / "annotations/fixture.json"
    archive.original_link(source, destination)
    assert destination.is_symlink()
    assert destination.read_bytes() == source.read_bytes() == before
    archive.original_link(source, destination)
    conflict = tmp_path / "other.json"
    conflict.write_text("{}")
    with pytest.raises(FileExistsError):
        archive.original_link(conflict, destination)
    assert source.read_bytes() == before


def test_missing_metadata_list_does_not_invent_values():
    row = {"reviewer": "real_private_code", "approval": "PENDING", "reviewed_at": ""}
    missing = archive.missing_human_fields(row)
    assert "reviewer" not in missing
    assert "second_reviewer" in missing
    assert "reviewed_at" in missing
    assert "approval" in missing
    assert row == {"reviewer": "real_private_code", "approval": "PENDING", "reviewed_at": ""}


def test_all_existing_manual_cells_are_protected():
    before = [{"frame_id": f"synthetic_{number:04d}", "D1": "YES", "approval": "PENDING",
               "reviewer": "true_private_code", "annotation_sha256": ""} for number in range(32)]
    after = [{**row, "annotation_sha256": "objective_hash", "image_width": "20"} for row in before]
    delivery.check_table_preservation(before, after)
    after[0]["D1"] = "NO"
    with pytest.raises(ValueError, match="Existing record overwritten"):
        delivery.check_table_preservation(before, after)
    after[0]["D1"] = "YES"
    after[0]["reviewer"] = "fake_reviewer"
    with pytest.raises(ValueError, match="Existing record overwritten"):
        delivery.check_table_preservation(before, after)


def test_numeric_zero_is_not_an_allowed_unmeasured_result():
    delivery.require_blank_metrics([{"r0_f1": "", "r1_f1": ""}], ["r0_f1", "r1_f1"])
    with pytest.raises(ValueError, match="unmeasured numeric"):
        delivery.require_blank_metrics([{"r0_f1": "0"}], ["r0_f1"])


def test_subset_review_cannot_be_archived_as_full_set():
    rows = [{"frame_id": f"synthetic_{number:04d}", "D1": "YES"} for number in range(32)]
    with pytest.raises(ValueError, match="same 32 identities"):
        delivery.check_table_preservation(rows, rows[:31])
